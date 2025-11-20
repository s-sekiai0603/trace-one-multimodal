# -*- coding: utf-8 -*-
"""
OsnetController — 外見（ReID）モードの検出・学習・識別をまとめるコントローラ
- YOLOv8-Pose(ONNX)で人物BBOX検出（骨格は描画しない）
- OSNet(ONNX)で外見特徴の埋め込み（単体/バッチ）
- 学習：クリックで選んだBBOXを切り出して特徴をバッファに蓄積 → 保存ダイアログでギャラリーへ
- 識別：全BBOXを切り出して一括埋め込み → ギャラリー（JSON/JSONL）検索 → ラベルと類似度を描画

AppController から呼ばれる公開メソッド（互換）：
  - set_mode(text)
  - on_detect_toggle(on)
  - on_feature_learning_toggled(on)
  - on_identify_toggle(on)
  - set_identify_config(gallery_json_path, target_pid)
  - on_frame(frame_bgr)
  - on_frame_clicked(x, y)
  - save_feature_buffer_dialog()

依存：
  - src/detectors/yolo_pose.py  … YOLOPoseDetector
  - src/embedders/osnet.py     … Osnet, OsnetJsonGallery
  - src/config.py               … 各種定数（モデルパス・色・TTA設定など）
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import os
import time
import logging
import numpy as np
import cv2
from PySide6.QtWidgets import QInputDialog

log = logging.getLogger("app.osnet_ctrl")

# 相対インポート（フォールバック内蔵で安全）
from ..detectors.yolo_pose import YOLOPoseDetector
from ..embedders.osnet import Osnet, OsnetJsonGallery

# 設定
from ..config import (
    OSNET_MODEL_PATH, OSNET_INPUT_SIZE_HW, REID_TTA_HFLIP, REID_BATCH_SIZE,
    REID_CROP_PAD_RATIO, REID_GALLERY_DIR, REID_TOPK, REID_USE_CENTROID,
    ORT_PROVIDERS, REID_EVERY_N, REID_MAX_BOXES,
    BBOX_COLOR, SELECTED_BBOX_COLOR, TOP_MATCH_BBOX_COLOR
)

try:
    import onnxruntime as ort
except Exception:
    ort = None


class OsnetController:
    def __init__(self, ui_window):
        self.ui = ui_window

        # モード有効/状態フラグ
        self._active: bool = False  # set_mode() で「外見」のとき True
        self._det_on: bool = False
        self._feat_on: bool = False
        self._id_on: bool = False

        # クリック選択（UI側でframeClickedを受けて更新）
        self._click_xy: Optional[Tuple[int, int]] = None
        self._last_boxes: Optional[np.ndarray] = None
        self._last_scores: Optional[np.ndarray] = None

        # 学習用バッファ（1セッションで複数フレーム集める）
        self._feat_buffer: List[np.ndarray] = []

        # ReID（OSNet）推論器の状態
        self._ort_sess = None        # onnxruntime.InferenceSession
        self._ort_in_name = None     # 入力名
        self._reid_dim: Optional[int] = None

        # Osnet（前処理＋embedのラッパ）
        self._reid = Osnet(self._get_state, self._set_output_dim)

        # 検出器（YOLO-Pose）— 骨格は描画しない前提で detect() を使う
        self._det = YOLOPoseDetector()

        # ギャラリー（labels.json / labels.jsonl）
        self._gallery: Optional[OsnetJsonGallery] = None
        self._gallery_pids_filter: Optional[List[str]] = None  # None=全件 / ["P0001"] など

        # Nフレーム間引き用
        self._frame_count: int = 0

        # セッション初期化は遅延実行
        log.info("[OSNET] controller ready. model=%s", OSNET_MODEL_PATH)
        
    def _safe_every_n(self) -> int:
        try:
            return max(1, int(REID_EVERY_N))
        except Exception:
            return 3  # デフォルト

    def _safe_max_boxes(self) -> int:
        try:
            return max(1, int(REID_MAX_BOXES))
        except Exception:
            return 8  # デフォルト

    # ----------- Osnet に渡す状態getter/setter -----------
    def _get_state(self):
        return {
            "session": self._ensure_session(),
            "inp_name": self._ort_in_name,
            "input_size": tuple(OSNET_INPUT_SIZE_HW),  # (H,W)
            "output_dim": self._reid_dim,
        }

    def _set_output_dim(self, d: int):
        self._reid_dim = int(d)

    def _ensure_session(self):
        if self._ort_sess is not None:
            return self._ort_sess
        if ort is None:
            raise RuntimeError("onnxruntime が見つかりません。pip install onnxruntime-gpu / -cpu")

        model_path = str(OSNET_MODEL_PATH or "models/osnet_x1_0.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[OSNET] model not found: {model_path}")

        try:
            so = ort.SessionOptions()
            # 最適化を最大化
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # メモリパターンはケースによりスタッターを誘発するのでOFF
            so.enable_mem_pattern = False
            #（必要に応じて）スレッド数固定
            # so.intra_op_num_threads = 1

            self._ort_sess = ort.InferenceSession(model_path, sess_options=so, providers=ORT_PROVIDERS)
        except Exception as e:
            log.warning("[OSNET] providers=%s で初期化失敗 → CPUにフォールバック: %s", ORT_PROVIDERS, e)
            self._ort_sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        self._ort_in_name = self._ort_sess.get_inputs()[0].name

        # ウォームアップ（GPUの初回“ドン詰まり”を回避）
        H, W = OSNET_INPUT_SIZE_HW
        dummy = np.zeros((1, 3, H, W), np.float32)
        for _ in range(3):
            _ = self._ort_sess.run(None, {self._ort_in_name: dummy})
        used = self._ort_sess.get_providers()
        log.info("[OSNET] session providers=%s (requested=%s)", used, ORT_PROVIDERS)
        return self._ort_sess

    # ----------------- AppController互換API -----------------
    def set_mode(self, text: str):
        self._active = (str(text or "").strip() == "外見")
        log.info("[OSNET] set_mode=%s active=%s", text, self._active)

    def on_detect_toggle(self, on: bool):
        self._det_on = bool(on)
        log.info("[OSNET] DETECT=%s", self._det_on)

    def on_feature_learning_toggled(self, on: bool):
        self._feat_on = bool(on)
        if not self._feat_on:
            # OFFに倒れたらクリック状態は破棄
            self._click_xy = None
        log.info("[OSNET] FEATURE=%s", self._feat_on)

    def on_identify_toggle(self, on: bool):
        self._id_on = bool(on)
        log.info("[OSNET] IDENTIFY=%s", self._id_on)

    def set_identify_config(self, gallery_json_path: str, target_pid: Optional[str]):
        """
        identifyダイアログで選ばれたギャラリー/対象PIDをここで反映
        - gallery_json_path はファイルパスだが、OsnetJsonGallery は base_dir を取るので dirname() を使う
        """
        try:
            base_dir = gallery_json_path if not os.path.isfile(gallery_json_path) else os.path.dirname(gallery_json_path)
            self._gallery = OsnetJsonGallery(base_dir=base_dir)
            self._gallery_pids_filter = [str(target_pid).strip()] if (target_pid and str(target_pid).strip()) else None
            ntotal = int(self._gallery.ntotal or 0)
            log.info("[OSNET] identify-config: base_dir=%s, target_pid=%s, ntotal=%d",
                     base_dir, (target_pid or "(all)"), ntotal)
            try:
                self._status(f"外見: ギャラリー={os.path.basename(base_dir)} / PID={target_pid or '全件'}（サンプル数={ntotal}）")
            except Exception:
                pass
        except Exception as e:
            log.exception("[OSNET] set_identify_config failed: %s", e)

    def on_frame_clicked(self, x: int, y: int):
        # 特徴学習ONのときだけ有効
        self._click_xy = (x, y) if self._feat_on else None
        log.info("[OSNET] click=%s (feat_on=%s)", self._click_xy, self._feat_on)

    # ----------------- メイン：1フレーム処理 -----------------
    def on_frame(self, frame_bgr: np.ndarray):
        if not self._active:
            return  # 他モード
        import time
        t0 = time.perf_counter()
        try:
            out = frame_bgr  # インプレース描画

            boxes = None
            scores = None

            # 検出
            if self._det_on:
                t_det0 = time.perf_counter()
                boxes, scores, _kpts = self._det.detect(out)
                t_det1 = time.perf_counter()
                self._last_boxes = boxes
                self._last_scores = scores
                self._draw_boxes(out, boxes, color=BBOX_COLOR, scores=scores)
            else:
                self._last_boxes = None
                self._last_scores = None
                t_det0 = t_det1 = time.perf_counter()

            # 特徴学習
            t_emb0 = time.perf_counter()
            if self._feat_on and self._det_on and boxes is not None and boxes.size > 0:
                sel_idx = self._select_nearest_box(self._click_xy, boxes) if self._click_xy is not None else None
                if sel_idx is not None:
                    x1, y1, x2, y2 = [int(v) for v in boxes[sel_idx]]
                    crop = self._crop_with_margin(out, (x1, y1, x2, y2), REID_CROP_PAD_RATIO)
                    if crop is not None and crop.size > 0:
                        try:
                            feat = self._reid.embed(crop)  # L2済
                            self._feat_buffer.append(np.asarray(feat, np.float32).reshape(-1))
                            cv2.rectangle(out, (x1, y1), (x2, y2), SELECTED_BBOX_COLOR, 2, cv2.LINE_AA)
                            self._put_text(out, "LEARN", (x1, max(16, y1 - 6)))
                        except Exception as e:
                            log.warning("[OSNET] feature extract failed: %s", e)
            t_emb1 = time.perf_counter()

            # 識別
            t_id0 = time.perf_counter()
            if self._id_on and self._det_on and self._gallery is not None and boxes is not None and boxes.size > 0:
                self._frame_count += 1
                every_n = self._safe_every_n()
                if (self._frame_count % every_n) == 0:
                    try:
                        boxes_id, sims = self._identify_step(out, boxes, scores)
                        if boxes_id is not None and sims is not None and len(sims) > 0:
                            best_i = int(np.argmax(sims))
                            if 0 <= best_i < boxes_id.shape[0]:
                                x1, y1, x2, y2 = [int(v) for v in boxes_id[best_i]]
                                cv2.rectangle(out, (x1, y1), (x2, y2), TOP_MATCH_BBOX_COLOR, 3, cv2.LINE_AA)
                    except Exception as e:
                        log.exception("[OSNET] identify failed: %s", e)
            t_id1 = time.perf_counter()

            # 再描画
            try:
                self.ui.view.set_bgr(out)
            except Exception:
                pass

            # 軽い計測ログ（必要なら間引いてOK）
            log.info("[OSNET][prof] det=%.1fms  emb=%.1fms  id=%.1fms  total=%.1fms",
                     (t_det1 - t_det0) * 1000.0,
                     (t_emb1 - t_emb0) * 1000.0,
                     (t_id1 - t_id0) * 1000.0,
                     (time.perf_counter() - t0) * 1000.0)

        except Exception as e:
            # ここで握っておけば、QTimerが止まる事態を防げます
            log.exception("[OSNET] on_frame failed: %s", e)


    # ----------------- 保存ダイアログ（学習） -----------------
    def save_feature_buffer_dialog(self):
        """
        バッファに貯めた外見特徴（複数フレーム）を1本に平均→L2正規化し、ギャラリーへ追加して保存。
        - 保存先は config.REID_GALLERY_DIR/labels.json[.jsonl]
        - 既存のPIDへ“追加登録”する（複数特徴を持てる仕様）
        """
        feats = list(self._feat_buffer)
        if not feats:
            self._status("外見: 学習バッファが空です（検出ON→特徴学習ON→BBOXクリックの順）", 2500)
            return

        pid, ok = QInputDialog.getText(self.ui, "P-ID を入力", "保存するID（既存IDに追加可）:")
        if not ok or not str(pid).strip():
            return
        pid = str(pid).strip()

        try:
            base_dir = str(REID_GALLERY_DIR or "data/gallery/appearance")
            gal = OsnetJsonGallery(base_dir)
            vec = self._average_l2(feats)
            ok_add = gal.add(pid, vec)
            gal.save()
            self._feat_buffer.clear()
            msg = f"[OSNET] 外見特徴を保存しました → {base_dir}"
            log.info(msg + f" (pid={pid}, add_ok={ok_add})")
            self._status("保存完了: 外見ギャラリーを更新しました", 2200)
        except Exception as e:
            log.error("[OSNET] 保存失敗: %s", e, exc_info=True)
            self._status(f"保存失敗: {e}", 2500)

    # ----------------- 内部処理 -----------------
    def _identify_step(self, frame_bgr: np.ndarray, boxes: np.ndarray, scores: Optional[np.ndarray]):
        """
        全BBOXを切り出して一括埋め込み → ギャラリー検索 → ラベルと類似度を描画。
        返り値: (処理対象に使ったboxes, sims)  ※最良枠ハイライトに利用
        """
        if boxes is None or boxes.size == 0 or self._gallery is None:
            return None, None

        # 上位スコアから上限人数だけを対象にする
        idxs = np.arange(boxes.shape[0], dtype=int)
        if scores is not None and len(scores) == boxes.shape[0]:
            idxs = np.argsort(-scores)  # 降順
        idxs = idxs[: int(max(1, int(REID_MAX_BOXES)))]

        # 切り出し
        crops: List[np.ndarray] = []
        boxes_sel: List[np.ndarray] = []
        for i in idxs:
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            crop = self._crop_with_margin(frame_bgr, (x1, y1, x2, y2), REID_CROP_PAD_RATIO)
            if crop is None or crop.size == 0:
                continue
            crops.append(crop)
            boxes_sel.append(boxes[i])
        if not crops:
            return None, None
        boxes_sel = np.stack(boxes_sel, axis=0)

        # 一括埋め込み
         # 一括埋め込み（計測）
        t_e0 = time.perf_counter()
        feats = self._reid.embed_batch(crops, batch_size=int(REID_BATCH_SIZE or 16))
        t_e1 = time.perf_counter()

        # 検索（計測）
        t_s0 = time.perf_counter()
        labels: List[str] = []
        sims: List[float] = []
        for f in feats:
            if f is None or np.linalg.norm(f) < 1e-12:
                labels.append("(unknown)"); sims.append(0.0); continue
            # ラベル制限（target_pid指定時）
            if self._gallery_pids_filter:
                # 対象PIDのみ計算（簡易：filterを通るものだけ search相当）
                best_lab = "(unknown)"; best_sim = -1.0
                for lab in self._gallery_pids_filter:
                    cen = self._gallery.centroid(lab)  # type: ignore
                    if cen is None:
                        continue
                    s = float(np.dot(f, cen))
                    if s > best_sim:
                        best_sim, best_lab = s, lab
                if best_sim < 0:
                    best_lab, best_sim = "(unknown)", 0.0
                labels.append(best_lab); sims.append(float(best_sim))
            else:
                top = self._gallery.search(f, topk=int(REID_TOPK or 5))  # type: ignore
                lab, sim = (top[0] if top else ("(unknown)", 0.0))
                labels.append(str(lab)); sims.append(float(sim))
                
        t_s1 = time.perf_counter()
        log.info("[OSNET][id] emb=%.1fms  search=%.1fms  n=%d",
                 (t_e1 - t_e0) * 1000.0, (t_s1 - t_s0) * 1000.0, len(crops))
        # 描画
        sims_arr = np.asarray(sims, dtype=np.float32)
        for i, (box, lab, sim) in enumerate(zip(boxes_sel, labels, sims)):
            x1, y1, x2, y2 = [int(v) for v in box]
            self._draw_label(frame_bgr, (x1, y1), f"{lab}  {sim:.3f}")

        return boxes_sel, sims_arr

    @staticmethod
    def _average_l2(vecs: List[np.ndarray]) -> np.ndarray:
        arr = np.asarray([np.asarray(v, np.float32).reshape(-1) for v in vecs], dtype=np.float32)
        v = arr.mean(axis=0)
        n = float(np.linalg.norm(v)) + 1e-12
        return (v / n).astype(np.float32)

    @staticmethod
    def _crop_with_margin(bgr: np.ndarray, box: Tuple[int, int, int, int], pad_ratio: float = 0.05) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        H, W = bgr.shape[:2]
        w = x2 - x1; h = y2 - y1
        px = int(round(w * float(pad_ratio))); py = int(round(h * float(pad_ratio)))
        xx1 = max(0, x1 - px); yy1 = max(0, y1 - py)
        xx2 = min(W, x2 + px); yy2 = min(H, y2 + py)
        if xx2 <= xx1 or yy2 <= yy1:
            return None
        return bgr[yy1:yy2, xx1:xx2].copy()

    @staticmethod
    def _select_nearest_box(pt: Optional[Tuple[int, int]], boxes: np.ndarray) -> Optional[int]:
        if pt is None or boxes is None or boxes.size == 0:
            return None
        x, y = pt
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        return int(np.argmin(d2))

    # ----------- ちょい描画ユーティリティ -----------
    @staticmethod
    def _draw_boxes(img: np.ndarray, boxes: np.ndarray, color=(255, 0, 0), scores: Optional[np.ndarray] = None):
        if boxes is None or boxes.size == 0:
            return
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            if scores is not None and i < len(scores):
                s = float(scores[i])
                cv2.putText(img, f"{s:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @staticmethod
    def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _draw_label(self, img: np.ndarray, org: Tuple[int, int], text: str):
        # ラベルの描画（かぶり回避の複雑な処理はAppController側に寄せる想定）
        self._put_text(img, text, org, color=(255, 255, 255))

    def _status(self, msg: str, ms: int = 1200):
        try:
            self.ui.statusBar().showMessage(msg, ms)
        except Exception:
            pass

    # リソース解放（将来用）
    def close(self):
        pass
