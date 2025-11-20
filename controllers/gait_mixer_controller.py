# src/controllers/gait_mixer_controller.py
# -*- coding: utf-8 -*-
"""
controllers/gait_mixer_controller.py — 歩容(Gait)モード用コントローラ
- YOLO-Pose で BBOX+骨格を取得
- 特徴学習：クリックで選択した1人について 32フレーム分の骨格を蓄積 → GaitMixer で128D埋め込み
- 保存：JSONギャラリー（data/gallery/gait/gait_gallery.json）へ追記（PIDは末尾に -P0001 形式で連番付与）
- 識別：全BBOXをIoUトラッカで追跡しながら、32フレーム貯まった人から順次埋め込み→
        ギャラリーとコサイン類似度で照合 → UIにスコア表示＆トップ枠を強調
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import os, json, time, logging
import numpy as np
import cv2

try:
    from PySide6.QtCore import Slot
    from PySide6.QtWidgets import QInputDialog
except Exception:
    Slot = lambda *a, **k: (lambda f: f)  # フォールバック
    class _Dummy: pass
    QInputDialog = _Dummy

from ..detectors.yolo_pose import YOLOPoseDetector
from ..embedders.gait_mixer import GaitMixerRunner

try:
    from ..config import (
        WINDOW_T,                      # 32 を想定
        GAIT_GALLERY_DIR,              # 例: data/gallery/gait
        SELECTED_BBOX_COLOR,
        TOP_MATCH_BBOX_COLOR,
    )
except Exception:
    WINDOW_T = 32
    GAIT_GALLERY_DIR = "data/gallery/gait"
    SELECTED_BBOX_COLOR = (0, 0, 255)
    TOP_MATCH_BBOX_COLOR = (0, 255, 0)

log = logging.getLogger("app.gait_ctrl")

# ------------------------------ JSONギャラリー ------------------------------
class GaitJsonGallery:
    def __init__(self, base_dir: str = GAIT_GALLERY_DIR, json_name: str = "gait_gallery.json") -> None:
        self.base_dir = base_dir
        self.json_path = os.path.join(base_dir, json_name)
        self.items: List[Dict[str, Any]] = []
        self._pid_index: Dict[str, List[int]] = {}
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        try: os.makedirs(self.base_dir, exist_ok=True)
        except Exception: pass

    def load(self):
        self.items.clear(); self._pid_index.clear()
        if os.path.isfile(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "items" in data: data = data["items"]
                if isinstance(data, list): self.items = data
            except Exception as e:
                log.warning("[GAIT][gallery] load failed: %s", e)
        for i, it in enumerate(self.items):
            pid = str(it.get("pid", ""))
            self._pid_index.setdefault(pid, []).append(i)

    def save(self):
        data = {"items": self.items, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error("[GAIT][gallery] save failed: %s", e)

    def alloc_suffix(self, base_pid: str) -> str:
        base = base_pid.strip()
        used = [str(it.get("pid","")) for it in self.items if str(it.get("pid","")).startswith(base+"-P")]
        taken = []
        for s in used:
            try: taken.append(int(s.rsplit("-P", 1)[1]))
            except Exception: pass
        n = 1
        while n in set(taken): n += 1
        return f"{base}-P{n:04d}"

    def add(self, pid: str, feat: np.ndarray, meta: Dict[str, Any] | None = None):
        if feat is None or feat.ndim != 1:
            raise ValueError("feat must be (D,) vector")
        item = {
            "pid": str(pid),
            "feat": feat.astype(np.float32).tolist(),
            "meta": meta or {},
            "ts": time.time()
        }
        self.items.append(item)
        self._pid_index.setdefault(item["pid"], []).append(len(self.items)-1)

    def feature_matrix(self, pids: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        feats = []; idxs = []
        for i, it in enumerate(self.items):
            if pids and str(it.get("pid","")) not in pids: continue
            fv = np.asarray(it.get("feat", []), dtype=np.float32)
            if fv.ndim != 1 or fv.size == 0: continue
            feats.append(fv); idxs.append(i)
        if not feats: return np.zeros((0,128), np.float32), []
        return np.stack(feats, axis=0).astype(np.float32, copy=False), idxs

# ------------------------------ 本体コントローラ ------------------------------
class GaitMixerController:
    """
    AppController から呼び出される“歩容モード専用”コントローラ。
    - on_frame(np.ndarray)
    - on_detection_toggled(bool)
    - on_feature_learning_toggled(bool)
    - on_identification_toggled(bool)
    - on_frame_clicked(int,int)
    - on_feature_save_requested()
    - set_identify_config(gallery_json_path, target_pid)
    """
    def __init__(self, win) -> None:
        self.win = win
        self.det_on = False; self.feat_on = False; self.id_on = False
        self.detector: Optional[YOLOPoseDetector] = None
        self.gait = GaitMixerRunner()

        self._last_boxes: Optional[np.ndarray] = None
        self._last_kpts: Optional[np.ndarray] = None
        self._click_xy: Optional[Tuple[int,int]] = None

        # 学習用
        self._sel_idx: Optional[int] = None
        self._sel_bbox_prev: Optional[np.ndarray] = None
        self._learn_xy: List[np.ndarray] = []  # (T,17,2)

        # 識別用IoUトラッカ
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_tid: int = 1
        self._last_track_ids: List[int] = []
        self._target_tid: Optional[int] = None

        # ギャラリー
        self.gallery = GaitJsonGallery()
        self._gallery_pids_filter: Optional[List[str]] = None
        
        self._active: bool = False
        
        log.info("[GAIT] controller ready. WINDOW_T=%s, gallery=%s", WINDOW_T, self.gallery.json_path)
        
    def set_mode(self, mode_text: str):
        """
        AppController からのモード変更通知。
        歩容モードのときだけアクティブにする。
        """
        active = (str(mode_text).strip() == "歩容")
        self._active = active
        log.info("[GAIT] set_mode=%s active=%s", mode_text, active)


    # ---------- トグル ----------
    def on_detection_toggled(self, on: bool):
        self.det_on = bool(on)
        if self.det_on and self.detector is None:
            try:
                self.detector = YOLOPoseDetector()
                log.info("[YOLO-POSE] initialized for gait.")
                self._status("人物検出: ON（歩容）")
            except Exception as e:
                log.exception("[YOLO-POSE] init failed: %s", e)
                self.det_on = False
        if not self.det_on:
            self.on_feature_learning_toggled(False)
            self.on_identification_toggled(False)
            self._sel_idx = None; self._sel_bbox_prev = None
            self._last_boxes = None; self._last_kpts = None
            self._tracks.clear(); self._next_tid = 1
            self._status("人物検出: OFF（歩容）")

    def on_feature_learning_toggled(self, on: bool):
        if not self.det_on:
            self._status("検出がOFFです"); return
        self.feat_on = bool(on)
        if self.feat_on:
            self.id_on = False
            self._sel_idx = None; self._sel_bbox_prev = None
            self._learn_xy.clear()
            self._status("特徴抽出: 対象人物のBBOXをクリック")
        else:
            self._status("特徴抽出: OFF")
        self._update_feature_progress()

    def on_identification_toggled(self, on: bool):
        """
        歩容識別トグル
        - 検出OFFのときは何もしない（ステータスだけ）
        - ON にしたらトラッカとギャラリーを初期化
        """
        if not self.det_on:
            self._status("検出がOFFです"); return

        self.id_on = bool(on)
        if self.id_on:
            # 学習は強制OFF
            self.feat_on = False

            # トラッカ系リセット
            self._tracks.clear()
            self._next_tid = 1
            self._last_track_ids = []
            self._target_tid = None

            # ギャラリーをロード
            self._gal_mat = self._get_gallery_matrix()
            if self._gal_mat is None or self._gal_mat.shape[0] == 0:
                self._status("ギャラリーが空です（識別できません）")
            else:
                self._status("歩容識別: ON")
        else:
            self._status("歩容識別: OFF")

    def on_frame_clicked(self, x: int, y: int):
        self._click_xy = (x, y)
        if not self.det_on or self._last_boxes is None or len(self._last_boxes) == 0:
            self._status("対象が検出されていません"); return
        idx = self._find_box_at(x, y, self._last_boxes)
        if idx is None: self._status("BBOX外をクリック"); return
        if self.feat_on:
            self._sel_idx = idx
            self._sel_bbox_prev = self._last_boxes[idx].copy()
            self._learn_xy.clear()
            self._status(f"選択: Person #{idx+1}（32フレーム貯めます）")
            self._update_feature_progress(); return
        if self.id_on and idx < len(self._last_track_ids):
            self._target_tid = int(self._last_track_ids[idx])
            self._status(f"ターゲット track_id={self._target_tid}")

    def on_feature_save_requested(self):
        if len(self._learn_xy) < WINDOW_T:
            self._status("まだ32フレームに達していません"); return
        seq = np.stack(self._learn_xy[:WINDOW_T], axis=0).astype(np.float32)  # (T,17,2)
        try:
            feat = self.gait.embed_xy_with_tta(seq)
        except Exception as e:
            log.exception("[GAIT] embed_xy_with_tta failed: %s", e)
            self._status("特徴抽出に失敗"); return
        if feat is None or feat.ndim != 1:
            self._status("有効な特徴が得られませんでした"); return
        pid_base, ok = QInputDialog.getText(self.win, "保存するID", "ベースID（例: Aoki0001）")
        if not ok or not str(pid_base).strip():
            self._status("保存をキャンセル"); return
        pid = self.gallery.alloc_suffix(str(pid_base).strip())
        self.gallery.add(pid, feat, meta={"source":"gait", "T": int(WINDOW_T)})
        self.gallery.save()
        self._status(f"保存: PID={pid}（{os.path.basename(self.gallery.json_path)}）")

    def set_identify_config(self, gallery_json_path: str, target_pid: Optional[str]):
        try:
            base_dir = gallery_json_path if not os.path.isfile(gallery_json_path) else os.path.dirname(gallery_json_path)
            self.gallery = GaitJsonGallery(base_dir=base_dir)
            self._gallery_pids_filter = [target_pid] if (target_pid and target_pid.strip()) else None

            # ここでキャッシュも更新しておく（None地獄を避ける）
            self._gal_mat = self._get_gallery_matrix()
            ntotal = 0 if (self._gal_mat is None) else int(self._gal_mat.shape[0])

            log.info("[GAIT] identify-config: base_dir=%s, target_pid=%s, ntotal=%d",
                    base_dir, (target_pid or "(all)"), ntotal)
            self._status(f"歩容: ギャラリー={os.path.basename(base_dir)} / PID={target_pid or '全件'}（ntotal={ntotal}）")
        except Exception as e:
            log.exception("[GAIT] set_identify_config failed: %s", e)


    # ---------- 毎フレーム ----------
    def on_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None: return
        out = frame_bgr  # YOLOがインプレース描画
        boxes = kpts = None
        if self.det_on and self.detector is not None:
            try:
                boxes, scores, kpts = self.detector.infer_candidates(out)
            except Exception as e:
                log.exception("[YOLO-POSE] infer failed: %s", e)
                z = np.zeros((0, 4), np.float32)
                boxes, kpts = z, z.reshape(0, 0, 3)
        self._last_boxes = boxes; self._last_kpts = kpts
        det_n = int(len(boxes)) if boxes is not None else 0

        # 学習モード：選択個体だけ32フレーム蓄積
        if self.feat_on and det_n > 0 and self._sel_idx is not None and 0 <= self._sel_idx < det_n:
            self._update_selection_from_iou(boxes)
            try:
                k_one = np.asarray(kpts[self._sel_idx:self._sel_idx+1], dtype=np.float32)  # (1,K,3)
                if k_one.ndim == 4: k = k_one[0, :, :2]
                else:               k = k_one[0, :17, :2] if k_one.shape[1] >= 17 else k_one[0, :, :2]
                if k.shape[0] < 17:
                    pad = np.zeros((17 - k.shape[0], 2), np.float32); k = np.concatenate([k, pad], axis=0)
                self._learn_xy.append(k.astype(np.float32))
                if len(self._learn_xy) > WINDOW_T: self._learn_xy.pop(0)
                self._update_feature_progress()
            except Exception as e:
                log.exception("[GAIT] learn buffer update failed: %s", e)
            try: self._draw_selected_bbox(out, boxes[self._sel_idx])
            except Exception: pass

        # 識別モード：全員をIoUトレース→32フレーム貯まった人を順次判定
        elif self.id_on and det_n > 0:
            if not self._tracks: self._tracks, self._next_tid = {}, 1
            tids = self._trace_people(boxes)
            self._last_track_ids = tids[:]
            scores = [None] * len(tids)
            gal = getattr(self, "_gal_mat", None)
            if gal is None:
                gal = self._get_gallery_matrix()
                self._gal_mat = gal

            if gal is None or gal.shape[0] == 0:
                try: self.win.view.set_bgr(out)
                except Exception: pass
                return

            for i, tid in enumerate(tids):
                try:
                    buf = self._tracks.get(tid, {}).get("kbuf", [])
                    if len(buf) < WINDOW_T:
                        continue

                    # (T,1,17,3) と (T,17,3) の両対応にする
                    seq = np.concatenate(buf[-WINDOW_T:], axis=0)
                    if seq.ndim == 4 and seq.shape[1] == 1:
                        xy = seq[:, 0, :17, :2]
                    elif seq.ndim == 3:
                        xy = seq[:, :17, :2]
                    else:
                        continue

                    q = self.gait.embed_xy_with_tta(xy)
                    if q is None:
                        continue
                    q = self._l2(q.astype(np.float32))
                    s = float(np.max(gal @ q))  # Top-1
                    scores[i] = s

                    # （任意）デバッグ確認用
                    # log.info("[GAIT] ident tid=%d T=%d score=%.4f", tid, len(buf), s)

                except Exception as e:
                    log.exception("[GAIT] identify person failed: %s", e)
                    
            self._draw_similarity_labels(out, boxes, scores)
            self._draw_target_dot(out, boxes, tids)

        try: self.win.view.set_bgr(out)
        except Exception: pass

    # ---------- 補助 ----------
    def _status(self, msg: str):
        try: self.win.statusBar().showMessage(str(msg), 1500)
        except Exception: pass

    def _update_feature_progress(self):
        try:
            cur = int(min(len(self._learn_xy), WINDOW_T)); total = int(WINDOW_T)
            if self.det_on and self.feat_on and hasattr(self.win, "setFeatureProgressText"):
                self.win.setFeatureProgressText(f"buffering {cur}/{total} frames")
            else:
                if hasattr(self.win, "setFeatureProgressText"): self.win.setFeatureProgressText("")
        except Exception: pass

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v)); return v if n < 1e-12 else (v / n)

    def _find_box_at(self, x: int, y: int, boxes: np.ndarray) -> Optional[int]:
        hits = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(((x2-x1)*(y2-y1), i))
        if not hits: return None
        hits.sort(key=lambda t: t[0]); return int(hits[0][1])

    def _update_selection_from_iou(self, boxes: np.ndarray, th: float = 0.1):
        if self._sel_bbox_prev is None or boxes is None or len(boxes) == 0: return
        prev = self._sel_bbox_prev
        x1, y1, x2, y2 = prev; area_prev = max(1.0, (x2-x1)*(y2-y1))
        best, best_i = 0.0, self._sel_idx
        for i, (a1,b1,a2,b2) in enumerate(boxes):
            xx1, yy1 = max(x1,a1), max(y1,b1); xx2, yy2 = min(x2,a2), min(y2,b2)
            w, h = max(0.0, xx2-xx1), max(0.0, yy2-yy1)
            inter = w*h; area_i = max(1.0, (a2-a1)*(b2-b1))
            iou = inter/(area_prev+area_i-inter+1e-6)
            if iou > best: best, best_i = iou, i
        if best >= th:
            self._sel_idx = int(best_i)
            self._sel_bbox_prev = boxes[self._sel_idx].copy()

    def _trace_people(self, boxes: np.ndarray, iou_th: float = 0.2) -> List[int]:
        if boxes is None or len(boxes) == 0:
            for t in list(self._tracks.keys()):
                self._tracks[t]['miss'] += 1
                if self._tracks[t]['miss'] > 30: self._tracks.pop(t, None)
            return []
        track_ids = [-1]*len(boxes); used = set()
        def iou(a,b):
            ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
            xx1,yy1 = max(ax1,bx1), max(ay1,by1); xx2,yy2 = min(ax2,bx2), min(ay2,by2)
            w,h = max(0, xx2-xx1), max(0, yy2-yy1)
            inter = w*h; area_a = max(1.0,(ax2-ax1)*(ay2-ay1)); area_b = max(1.0,(bx2-bx1)*(by2-by1))
            return inter/(area_a+area_b-inter+1e-6)
        for i, box in enumerate(boxes):
            best_t, best_v = None, 0.0
            for tid, st in self._tracks.items():
                if tid in used: continue
                v = iou(box, st['bbox'])
                if v > best_v: best_v, best_t = v, tid
            if best_t is not None and best_v >= iou_th:
                track_ids[i] = best_t; used.add(best_t)
                st = self._tracks[best_t]; st['bbox'] = box.copy(); st['miss'] = 0
        for i in range(len(boxes)):
            if track_ids[i] == -1:
                tid = self._next_tid; self._next_tid += 1
                self._tracks[tid] = {'bbox': boxes[i].copy(), 'kbuf': [], 'miss': 0}
                track_ids[i] = tid
        # kbuf push
        for i, tid in enumerate(track_ids):
            try:
                kk = np.asarray(self._last_kpts[i], dtype=np.float32)[None, ...]  # (1,17,3)
                self._tracks[tid]['kbuf'].append(kk)
                if len(self._tracks[tid]['kbuf']) > 64: self._tracks[tid]['kbuf'].pop(0)
            except Exception: pass
        # miss++
        for tid in list(self._tracks.keys()):
            if tid not in used and not any(tid == x for x in track_ids):
                self._tracks[tid]['miss'] += 1
                if self._tracks[tid]['miss'] > 30: self._tracks.pop(tid, None)
        return [int(t) for t in track_ids]

    def _get_gallery_matrix(self) -> Optional[np.ndarray]:
        mat, _ = self.gallery.feature_matrix(self._gallery_pids_filter)
        if mat is None or mat.size == 0: return None
        g = mat.astype(np.float32, copy=False); n = np.linalg.norm(g, axis=1, keepdims=True) + 1e-6
        return g / n

    def _draw_selected_bbox(self, frame_bgr: np.ndarray, box: np.ndarray):
        x1, y1, x2, y2 = map(int, box)
        thick = max(3, int(round(max(frame_bgr.shape[:2]) / 400)))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), SELECTED_BBOX_COLOR, thick, cv2.LINE_AA)

    def _draw_similarity_labels(self, frame_bgr: np.ndarray, boxes: np.ndarray, scores: List[Optional[float]]):
        if frame_bgr is None or boxes is None or scores is None or len(boxes) == 0: return
        H, W = frame_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_thick = max(1, int(round(max(H, W) / 900)))
        font_scale = max(0.5, min(1.2, max(H, W) / 1400.0))
        best_idx, best_val = None, float("-inf")
        for i, s in enumerate(scores):
            if s is None: continue
            if s > best_val: best_val, best_idx = s, i
        for i, s in enumerate(scores):
            if s is None or i >= len(boxes): continue
            x1,y1,x2,y2 = map(int, boxes[i]); label = f"{s:.4f}"
            (tw, th), bl = cv2.getTextSize(label, font, font_scale, base_thick)
            px, py = x1, min(H-2, y2 + th + 6)
            cv2.rectangle(frame_bgr, (px-2, py-th-6), (px+tw+2, py+2), (0,0,0), -1)
            color = (0,200,0) if s >= 0.6 else ((0,200,200) if s >= 0.4 else (0,165,255))
            cv2.putText(frame_bgr, label, (px,py), font, font_scale, color, base_thick, cv2.LINE_AA)
        if best_idx is not None and best_idx < len(boxes):
            x1,y1,x2,y2 = map(int, boxes[best_idx])
            thick = max(3, int(round(max(H, W) / 400)))
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), TOP_MATCH_BBOX_COLOR, thick, cv2.LINE_AA)

    def _draw_target_dot(self, frame_bgr, boxes, tids):
        if self._target_tid is None or boxes is None or len(boxes) == 0: return
        try: idx = next(i for i, tid in enumerate(tids) if int(tid) == int(self._target_tid))
        except StopIteration: return
        x1,y1,x2,y2 = map(int, boxes[idx])
        cx = max(0, min(frame_bgr.shape[1]-1, x1 + 8))
        cy = max(0, min(frame_bgr.shape[0]-1, y1 + 8))
        cv2.circle(frame_bgr, (cx,cy), 5, (0,0,255), -1, cv2.LINE_AA)