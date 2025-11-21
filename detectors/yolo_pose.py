# -*- coding: utf-8 -*-
"""
YOLOv8-Pose (ONNX) で人物BBOX + 骨格（COCO-17）を推定する軽量モジュール。
- detect()              : BBOXとscoreとkptsを返す（描画なし）
- detect_and_draw()     : BBOXと骨格をインプレース描画して返す
- UI側（外見/歩容）     : まずは BBOXのみを外見識別で利用、骨格は将来の歩容で再利用
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import cv2
import math
import logging 

import onnxruntime as ort

log = logging.getLogger("app.yolo_pose")

# --- 設定の相対/絶対フォールバック ---
try:
    from ..config import (
        YOLO_POSE_MODEL_PATH, YOLO_POSE_IMGSZ, YOLO_POSE_CONF, YOLO_POSE_IOU, YOLO_POSE_MAX_DET,
        ORT_PROVIDERS,
        POSE_KPT_CONF_MIN, POSE_HARD_FILTER_ENABLED, POSE_DRAW_ENABLED,
        POSE_DRAW_LANDMARK_BGR, POSE_DRAW_CONNECTION_BGR, POSE_DRAW_THICKNESS, POSE_DRAW_RADIUS,
        FACE_IDXS, EAR_IDXS, ANKLE_IDXS, BBOX_COLOR, 
    )
except Exception:
    # 最低限のフォールバック（config未整備でも動く）
    YOLO_POSE_MODEL_PATH = "models/yolov8l-pose.onnx"
    YOLO_POSE_IMGSZ      = 640
    YOLO_POSE_CONF       = 0.25
    YOLO_POSE_IOU        = 0.50
    YOLO_POSE_MAX_DET    = 300
    ORT_PROVIDERS        = ["CPUExecutionProvider"]
    POSE_KPT_CONF_MIN    = 0.35
    POSE_HARD_FILTER_ENABLED = True
    POSE_DRAW_ENABLED    = True
    POSE_DRAW_LANDMARK_BGR   = (0,255,0)
    POSE_DRAW_CONNECTION_BGR = (0,200,0)
    POSE_DRAW_THICKNESS  = 2
    POSE_DRAW_RADIUS     = 2
    FACE_IDXS  = (0,1,2)
    EAR_IDXS   = (3,4)
    ANKLE_IDXS = (15,16)


# ---- ユーティリティ ----

# COCO-17の簡易エッジ（Ultralytics準拠）
KPT_CONNECTIONS: List[Tuple[int,int]] = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (11,12), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16),
    (0,5), (0,6), (1,2), (1,3), (2,4)
]

def _letterbox(bgr: np.ndarray, new_size: int, color=(114,114,114)):
    oh, ow = bgr.shape[:2]
    s = min(new_size / max(oh,1), new_size / max(ow,1))
    nh, nw = int(round(oh * s)), int(round(ow * s))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top  = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out[top:top+nh, left:left+nw] = resized
    return out, s, left, top, (ow, oh)

def _scale_boxes_xyxy(xyxy_in: np.ndarray, s: float, left: int, top: int, orig_wh: Tuple[int,int]) -> np.ndarray:
    ow, oh = orig_wh
    x1 = (xyxy_in[:,0] - left) / s
    y1 = (xyxy_in[:,1] - top)  / s
    x2 = (xyxy_in[:,2] - left) / s
    y2 = (xyxy_in[:,3] - top)  / s
    x1 = np.clip(x1, 0, ow - 1); y1 = np.clip(y1, 0, oh - 1)
    x2 = np.clip(x2, 1, ow);     y2 = np.clip(y2, 1, oh)
    x2 = np.maximum(x2, x1 + 1); y2 = np.maximum(y2, y1 + 1)
    return np.stack([x1,y1,x2,y2], axis=1).astype(np.float32, copy=False)

def _scale_kpts_xy(kpts_in: np.ndarray, s: float, left: int, top: int, orig_wh: Tuple[int,int]) -> np.ndarray:
    ow, oh = orig_wh
    out = kpts_in.copy()
    out[..., 0] = (out[..., 0] - left) / s
    out[..., 1] = (out[..., 1] - top)  / s
    out[..., 0] = np.clip(out[..., 0], 0, ow - 1)
    out[..., 1] = np.clip(out[..., 1], 0, oh - 1)
    return out.astype(np.float32, copy=False)

def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_th]
    return np.asarray(keep, dtype=np.int32)

def _to_nc(a) -> Optional[np.ndarray]:
    """ONNX出力を (N,C) に正規化（YOLOv8系でよくある (1,C,N) も吸収）。"""
    x = np.asarray(a)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        return None
    # (C,N) → (N,C) の代表パターン
    if x.shape[0] in (56,57,58,85,116,117) and x.shape[1] >= 100:
        x = x.T
    return x.astype(np.float32, copy=False)

def _is_point_valid(k3: np.ndarray, idx: int, conf_min: float) -> bool:
    if idx >= k3.shape[0]:
        return False
    x = float(k3[idx, 0]); y = float(k3[idx, 1])
    c = float(k3[idx, 2]) if k3.shape[1] >= 3 else 1.0
    return np.isfinite(x) and np.isfinite(y) and (abs(x) + abs(y) > 1e-6) and (c >= conf_min)

def _passes_hard_filter(k3: np.ndarray) -> bool:
    if not POSE_HARD_FILTER_ENABLED:
        return True
    conf_min = float(POSE_KPT_CONF_MIN)
    K = k3.shape[0]
    # 1) 足首が両方NGなら除外
    ankles_ok = any(_is_point_valid(k3, idx, conf_min) for idx in ANKLE_IDXS if idx < K)
    if not ankles_ok:
        return False
    # 2) 顔+耳（0..4）が全滅なら除外
    face_ear_any = any(_is_point_valid(k3, idx, conf_min) for idx in (0,1,2,3,4) if idx < K)
    if not face_ear_any:
        return False
    # 3) 顔以外（5..16）の未取得が >=4 なら除外
    notok = 0
    for idx in range(5, min(17, K)):
        if not _is_point_valid(k3, idx, conf_min):
            notok += 1
            if notok >= 4:
                return False
    return True

def _calc_body_yaw_deg_from_kpts(k3: np.ndarray,
                                 conf_min: float = POSE_KPT_CONF_MIN) -> Optional[float]:
    """
    COCO-17 キーポイント (K,3) から「体の横方向ベクトル」を作り、 yaw 角度 [deg] を返す。

    使用点:
      - 左肩(5), 右肩(6)
      - 左腰(11), 右腰(12)

    ベクトル:
      v_shoulders = R_shoulder - L_shoulder
      v_hips      = R_hip      - L_hip
      v           = 0.5 * (v_shoulders + v_hips)

    角度定義:
      yaw_rad = atan2(vx, vy)
      yaw_deg = degrees(yaw_rad) in [-180, 180)
    """
    if k3 is None:
        return None

    k3 = np.asarray(k3, dtype=np.float32)
    if k3.ndim != 2 or k3.shape[1] < 2:
        return None

    K = k3.shape[0]
    idx_ls, idx_rs = 5, 6   # shoulders
    idx_lh, idx_rh = 11, 12 # hips

    def _get_pt(idx: int) -> Optional[np.ndarray]:
        if idx >= K:
            return None
        x, y = float(k3[idx, 0]), float(k3[idx, 1])
        c = float(k3[idx, 2]) if k3.shape[1] >= 3 else 1.0
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if c < conf_min:
            return None
        return np.array([x, y], dtype=np.float32)

    ls = _get_pt(idx_ls)
    rs = _get_pt(idx_rs)
    lh = _get_pt(idx_lh)
    rh = _get_pt(idx_rh)
    if ls is None or rs is None or lh is None or rh is None:
        return None

    v_shoulders = rs - ls
    v_hips      = rh - lh
    v = 0.5 * (v_shoulders + v_hips)

    vx, vy = float(v[0]), float(v[1])
    if not (np.isfinite(vx) and np.isfinite(vy)):
        return None
    if abs(vx) + abs(vy) < 1e-6:
        return None

    # atan2(x, y) なので、0° は「画像の下方向」、+90° は「右方向」あたり
    yaw_rad = math.atan2(vx, vy)
    yaw_deg = math.degrees(yaw_rad)

    # [-180, 180) に正規化
    if yaw_deg <= -180.0 or yaw_deg > 180.0:
        yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0

    return float(yaw_deg)


def _classify_body_dir8(yaw_deg: float) -> str:
    """
    yaw_deg [-180, 180) を 8方向に量子化してラベルを返す。

    ざっくり:
      -22.5 ~ +22.5         : front
      +22.5 ~ +67.5         : front-right
      +67.5 ~ +112.5        : right
      +112.5 ~ +157.5       : back-right
      +157.5~180 / -180~-157.5: back
      -157.5 ~ -112.5       : back-left
      -112.5 ~ -67.5        : left
      -67.5 ~ -22.5         : front-left
    """
    a = float(yaw_deg)
    # いちおう [-180,180) に揃える
    if a <= -180.0 or a > 180.0:
        a = ((a + 180.0) % 360.0) - 180.0

    if -22.5 <= a < 22.5:
        return "front"
    if 22.5 <= a < 67.5:
        return "front-right"
    if 67.5 <= a < 112.5:
        return "right"
    if 112.5 <= a < 157.5:
        return "back-right"
    if a >= 157.5 or a < -157.5:
        return "back"
    if -157.5 <= a < -112.5:
        return "back-left"
    if -112.5 <= a < -67.5:
        return "left"
    # -67.5 ~ -22.5
    return "front-left"



class YOLOPoseDetector:
    """
    infer: BGR -> (boxes(N,4), scores(N,), kpts(N,17,3: x,y,conf))
    - BBOXは外見識別（OSNet等）向け
    - kptsは将来の歩容向けにそのまま渡せる
    """
    def __init__(self,
                 model_path: str = YOLO_POSE_MODEL_PATH,
                 input_size: int = YOLO_POSE_IMGSZ,
                 providers: Optional[list] = None,
                 conf_th: float = YOLO_POSE_CONF,
                 nms_iou_th: float = YOLO_POSE_IOU,
                 max_dets: int = YOLO_POSE_MAX_DET):
        import logging, os
        _log = logging.getLogger("app.yolo_pose")

        self.input_size = int(input_size)
        self.conf_th = float(conf_th)
        self.nms_iou_th = float(nms_iou_th)
        self.max_dets = int(max_dets)

        model_path = str(model_path or YOLO_POSE_MODEL_PATH)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[YOLO-POSE] model not found: {model_path}")

        prov_req = providers or ORT_PROVIDERS
        try:
            self.sess = ort.InferenceSession(model_path, providers=prov_req)
        except Exception as e:
            _log.warning("[YOLO-POSE] session init failed on %s, fallback to CPU: %s", prov_req, e)
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.in_name = self.sess.get_inputs()[0].name

        # 軽いウォームアップ
        S = self.input_size
        _ = self.sess.run(None, {self.in_name: np.zeros((1,3,S,S), np.float32)})

        # ★ ロードログ（必ず出る）
        try:
            prov_used = self.sess.get_providers()
        except Exception:
            prov_used = prov_req
        _log.info("[YOLO-POSE] model loaded: %s (input=%d, providers=%s)", model_path, self.input_size, prov_used)

    # 互換インターフェイス
    def __call__(self, frame_bgr: np.ndarray):
        return self.detect(frame_bgr)

    def detect(self, frame_bgr: np.ndarray):
        """
        描画なしで boxes, scores, kpts を返す。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            z4 = np.zeros((0,4), np.float32)
            return z4, z4[:0], z4.reshape(0,0,3)

        S = self.input_size
        canvas, s, left, top, orig_wh = _letterbox(frame_bgr, S)

        # BGR->RGB, 0..1
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0/255.0, size=(S,S),
                                     mean=(0,0,0), swapRB=True, crop=False)
        out = self.sess.run(None, { self.in_name: blob })

        # 出力正規化
        heads = []
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            nc = _to_nc(o)
            if nc is not None:
                heads.append(nc)
        if not heads:
            z4 = np.zeros((0,4), np.float32)
            return z4, z4[:0], z4.reshape(0,0,3)

        head = max(heads, key=lambda x: x.shape[0])  # Nが最大のもの
        N, C = head.shape
        if N == 0:
            z4 = np.zeros((0,4), np.float32)
            return z4, z4[:0], z4.reshape(0,0,3)

        # C = 5 + (K*3) [+nc] を推定（Kは10〜30程度）
        def guess_layout(C):
            for nc in (0, 1, 80):
                rem = C - 5 - nc
                if rem % 3 == 0 and 10 <= (rem // 3) <= 30:
                    return nc, rem // 3
            return 0, 17
        nc, K = guess_layout(C)

        xywh = head[:, :4]
        obj  = head[:, 4]
        cls  = head[:, 5:5+nc] if nc > 0 else None
        kpts_flat = head[:, 5+nc:]

        # スケール（0..1や0..Sの両対応）
        if float(np.nanmax(np.abs(xywh))) <= 2.0:
            xywh = xywh * float(S)
        if kpts_flat.size > 0 and float(np.nanmax(np.abs(kpts_flat))) <= 2.0:
            kpts_flat = kpts_flat * float(S)

        # スコア
        if nc > 0 and cls is not None and cls.size > 0:
            cls_max = np.max(cls, axis=1)
            scores = (obj * cls_max).astype(np.float32, copy=False)
        else:
            scores = obj.astype(np.float32, copy=False)

        # conf 閾値
        keep = scores >= self.conf_th
        if not np.any(keep):
            z4 = np.zeros((0,4), np.float32)
            return z4, z4[:0], z4.reshape(0,0,3)

        xywh   = xywh[keep]
        scores = scores[keep]
        k_flat = kpts_flat[keep] if kpts_flat.size else np.zeros((0, K*3), np.float32)

        # xywh→xyxy（SxS座標）
        cx, cy, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
        xyxy_in = np.stack([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0], axis=1)

        # kpts（SxS座標）
        kpts = k_flat.reshape(-1, K, 3) if k_flat.size else np.zeros((xyxy_in.shape[0], K, 3), np.float32)

        # NMS
        if xyxy_in.shape[0] > 0:
            keep_idx = _nms_xyxy(xyxy_in, scores, self.nms_iou_th)
            xyxy_in  = xyxy_in[keep_idx]
            scores   = scores[keep_idx]
            kpts     = kpts[keep_idx]

        # 上限
        if xyxy_in.shape[0] > self.max_dets:
            idx = np.argsort(-scores)[:self.max_dets]
            xyxy_in = xyxy_in[idx]; scores = scores[idx]; kpts = kpts[idx]

        # 元画像座標系へ
        boxes = _scale_boxes_xyxy(xyxy_in, s, left, top, orig_wh)
        kpts  = _scale_kpts_xy(kpts,     s, left, top, orig_wh)

        # ハードフィルタ
        if boxes.shape[0] > 0 and POSE_HARD_FILTER_ENABLED:
            keep2 = []
            for i in range(boxes.shape[0]):
                keep2.append(_passes_hard_filter(kpts[i]))
            keep2 = np.asarray(keep2, dtype=bool)
            if np.any(keep2):
                boxes  = boxes[keep2]
                scores = scores[keep2]
                kpts   = kpts[keep2]
            else:
                z4 = np.zeros((0,4), np.float32)
                return z4, z4[:0], z4.reshape(0,0,3)

        return boxes.astype(np.float32, copy=False), scores.astype(np.float32, copy=False), kpts.astype(np.float32, copy=False)

    def detect_and_draw(self,
                        frame_bgr: np.ndarray,
                        show_score: bool = True,
                        debug_dir8: bool = False):
        """
        検出＋骨格/BBOXをインプレース描画。

        debug_dir8=True のとき:
          - 各人物について肩・腰から yaw 角を計算
          - 8方向ラベルに量子化してログ出力のみ行う
        """
        boxes, scores, kpts = self.detect(frame_bgr)

        # ★ 8方向のデバッグログ（普段は OFF / 学習時だけ ON にする想定）
        if debug_dir8 and kpts is not None and kpts.size > 0:
            try:
                n = min(len(boxes), len(kpts))
            except Exception:
                n = 0
            for i in range(n):
                yaw = _calc_body_yaw_deg_from_kpts(kpts[i])
                if yaw is None:
                    continue
                dir8 = _classify_body_dir8(yaw)
                log.info(
                    "[YOLO-POSE][DIR8] idx=%d yaw=%.1f deg -> %s",
                    i, yaw, dir8,
                )

        if POSE_DRAW_ENABLED:
            self._draw_pose(frame_bgr, boxes, scores, kpts, show_score=show_score)
        return boxes, scores, kpts
    
    def infer_candidates(self, frame_bgr: np.ndarray):
        """
        歩容コントローラ（GaitMixerController）互換のためのヘルパー。
        現状は detect_and_draw() と同じ動作で、(boxes, scores, kpts) を返す。
        """
        return self.detect_and_draw(frame_bgr)

    # 骨格＋BBOXの描画
    def _draw_pose(self, frame_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, kpts: np.ndarray, show_score: bool = True):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if boxes is None or boxes.size == 0:
            return

        H, W = frame_bgr.shape[:2]
        thickness = max(1, int(round(max(W, H) / 720.0)) * POSE_DRAW_THICKNESS // 2)
        radius    = max(1, int(round(max(W, H) / 1080.0)) * POSE_DRAW_RADIUS)

        for i in range(min(len(boxes), len(kpts))):
            x1, y1, x2, y2 = map(int, boxes[i])
            # BBOX
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), BBOX_COLOR, 2, cv2.LINE_AA)
            if show_score and scores is not None and i < len(scores):
                label = f"{float(scores[i]):.2f}"
                cv2.putText(frame_bgr, label, (x1, max(15, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, POSE_DRAW_CONNECTION_BGR, 1, cv2.LINE_AA)

            # 骨格
            pts = kpts[i]
            # 線
            for a, b in KPT_CONNECTIONS:
                if a < pts.shape[0] and b < pts.shape[0]:
                    xa, ya, ca = pts[a]
                    xb, yb, cb = pts[b]
                    if ca >= POSE_KPT_CONF_MIN and cb >= POSE_KPT_CONF_MIN:
                        cv2.line(frame_bgr, (int(xa), int(ya)), (int(xb), int(yb)),
                                 POSE_DRAW_CONNECTION_BGR, thickness, cv2.LINE_AA)
            # 点
            for j in range(pts.shape[0]):
                x, y, c = pts[j]
                if c >= POSE_KPT_CONF_MIN:
                    cv2.circle(frame_bgr, (int(x), int(y)), radius, POSE_DRAW_LANDMARK_BGR, -1, cv2.LINE_AA)
