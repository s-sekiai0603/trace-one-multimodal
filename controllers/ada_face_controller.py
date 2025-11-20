# -*- coding: utf-8 -*-
"""
AdaFaceController — 顔モード専用コントローラ
- PlayerWindow のシグナルを購読し、顔モードかつ「検出」ON時に YOLO-Face を実行
- 特徴学習ON時：クリック選択したBBOXのみ AdaFace で埋め込み→一時バッファに蓄積
- 識別ON時：各BBOXを AdaFace で埋め込み、バッファ平均（ギャラリー）との cos 類似度を表示
- 保存機能（JSON/FAISS）は今回は未実装
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import logging
import numpy as np
import cv2
import os
import json

from PySide6.QtCore import QObject, Slot

# 設定（色やデバイス、モード名など）
try:
    from ..config import (
    TORCH_DEVICE,
    YOLO_FACE_MODEL_PATH, YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_FACE_IMGSZ, YOLO_FACE_MAX_DET,
    ADAFACE_CKPT_PATH, ADAFACE_ARCH, ADAFACE_MARGIN, ADAFACE_OUT_SIZE, ADAFACE_LOG_EVERY,
    BBOX_COLOR, SELECTED_BBOX_COLOR,
)
except Exception:
    TORCH_DEVICE = "cpu"
    BBOX_COLOR = (0, 255, 255)
    SELECTED_BBOX_COLOR = (0, 180, 255)

# 検出・埋め込み
from ..detectors.yolo_face import YOLOFaceDetector
from ..embedders.ada_face import AdaFaceEmbedder
from ..utils.ada_face_utils import inflate_boxes_xyxy, smooth_boxes_ema, choose_boxes


FACE_MODE_NAME = "顔"  # Player のドロップダウンでこの文字列を使用している前提


class AdaFaceController(QObject):
    """
    役割：
      - PlayerWindow からのフレームを受け取り、顔モードかつ検出ON時のみ処理
      - 特徴学習ON：クリック選択された一人のみ埋め込み→一時バッファ（平均が疑似ギャラリー）
      - 識別ON：各BBOXに対して埋め込み→ギャラリー平均と cos 類似度 → 枠と数値を描画
    """
    def __init__(self, window) -> None:
        super().__init__(window)
        self.log = logging.getLogger("app.adaface_ctrl")
        self.win = window

        # --- モード＆トグル状態 ---
        self.current_mode: str = FACE_MODE_NAME  # AppControllerが差し替えるなら setter を呼ぶ
        self.det_on: bool = False
        self.feat_on: bool = False
        self.id_on: bool = False
        
        self._last_frame: Optional[np.ndarray] = None
        
        self._gallery_json_path: Optional[str] = None
        self._target_pid: Optional[str] = None
        
        # --- 識別設定 ---
        self._fixed_gallery: Optional[np.ndarray] = None
        
        # --- 推論器 ---
        self.detector: Optional[YOLOFaceDetector] = None
        self.embedder: Optional[AdaFaceEmbedder] = None

        # --- 状態 ---
        self._last_boxes: Optional[np.ndarray] = None
        self._last_scores: Optional[np.ndarray] = None
        self.sel_idx: Optional[int] = None        # 画面上の選択 BBOX index
        self._sel_bbox_prev: Optional[np.ndarray] = None
        self._feat_buf: List[np.ndarray] = []     # 一時ギャラリー（平均を使う）
        self._label_occupied: List[Tuple[int, int, int, int]] = []

        # --- PlayerWindow シグナル購読（旧名互換含む） ---
        # トグル
        if hasattr(self.win, "detectionToggled"):
            self.win.detectionToggled.connect(self.on_detection_toggled)
        if hasattr(self.win, "personDetectionToggled"):
            self.win.personDetectionToggled.connect(self.on_detection_toggled)

        if hasattr(self.win, "featureLearningToggled"):
            self.win.featureLearningToggled.connect(self.on_feature_toggled)
        if hasattr(self.win, "featureExtractionToggled"):
            self.win.featureExtractionToggled.connect(self.on_feature_toggled)

        if hasattr(self.win, "identificationToggled"):
            self.win.identificationToggled.connect(self.on_identification_toggled)
        if hasattr(self.win, "faceIdentificationToggled"):
            self.win.faceIdentificationToggled.connect(self.on_identification_toggled)

        # フレーム・クリック
        self.win.frameAvailable.connect(self.on_frame)
        self.win.frameClicked.connect(self.on_frame_clicked)

        # その他
        if hasattr(self.win, "appClosing"):
            self.win.appClosing.connect(self.on_app_closing)
        if hasattr(self.win, "fileSelected"):
            self.win.fileSelected.connect(self.on_file_selected)

        self.log.info("[AdaFaceController] ready.")

    # ====== 外部からモードを切り替えるための API（AppController から呼ぶ想定） ======
    def set_mode(self, mode_name: str) -> None:
        """ドロップダウンの変更を AppController から通知するための簡単な setter。"""
        self.current_mode = str(mode_name or "").strip()
        # 顔モード以外へ切り替わったら内部状態を軽くリセット
        if self.current_mode != FACE_MODE_NAME:
            self._reset_runtime(light=True)

    # ====== トグル ======
    @Slot(bool)
    def on_detection_toggled(self, on: bool):
        self.det_on = bool(on)
        if not self._is_face_mode():
            return

        if self.det_on and self.detector is None:
            try:
                self.detector = YOLOFaceDetector(
                    model_path=YOLO_FACE_MODEL_PATH,
                    device=TORCH_DEVICE,
                    conf=YOLO_FACE_CONF,
                    iou=YOLO_FACE_IOU,
                    imgsz=YOLO_FACE_IMGSZ,
                    max_det=YOLO_FACE_MAX_DET,
                )
                self._status("顔検出: ON")
                self.log.info("[YOLO-FACE] initialized (%s).", TORCH_DEVICE)
            except Exception as e:
                self.detector = None
                self.det_on = False
                self._status("顔検出 初期化失敗")
                self.log.exception("YOLO-FACE init failed: %s", e)
        elif not self.det_on:
            self._status("顔検出: OFF")
            # 依存状態を整理
            self.feat_on = False
            self.id_on = False
            self.sel_idx = None
            self._last_boxes = None
            self._last_scores = None
            self._sel_bbox_prev = None
            self._feat_buf.clear()
            # OFF直後に最終フレームを“素の状態”で再表示して残像を消す
            try:
                if hasattr(self, "_last_frame") and self._last_frame is not None:
                    self.win.view.set_bgr(self._last_frame)  # ここでは描画しない
            except Exception:
                pass

        # ON直後は現在フレームで即時再処理（停止中でもBBOXを出す）
        if self.det_on:
            self._reprocess_last_frame()
            
        if self.det_on:
            self._reprocess_last_frame()

    @Slot(bool)
    def on_feature_toggled(self, on: bool):
        # 顔モードかつ検出ONでのみ有効
        if not (self._is_face_mode() and self.det_on):
            # Player 側のUIが disable 管理してくれているはずだが念のため非許可
            self._status("特徴学習は“検出ON”で有効です")
            self.feat_on = False
            return

        self.feat_on = bool(on)
        if self.feat_on:
            # 識別と相互排他（仕様上はUIで活性制御済み）
            self.id_on = False
            # 埋め込み器の遅延初期化
            if self.embedder is None:
                try:
                    self.embedder = AdaFaceEmbedder(
                        ckpt_path=ADAFACE_CKPT_PATH,
                        architecture=ADAFACE_ARCH,
                        device=TORCH_DEVICE,
                        margin=ADAFACE_MARGIN,
                        out_size=ADAFACE_OUT_SIZE,
                        log_each=ADAFACE_LOG_EVERY,
                    )
                    self.log.info("[ADAFACE] initialized (%s).", TORCH_DEVICE)
                except Exception as e:
                    self.embedder = None
                    self.feat_on = False
                    self._status("AdaFace初期化に失敗")
                    self.log.exception("AdaFace init failed: %s", e)
                    return
            # 学習バッファを新規に（選択切替に追従させやすい運用）
            self._feat_buf.clear()
            self.sel_idx = None
            self._sel_bbox_prev = None
            self._status("特徴学習: BBOXをクリックして対象を選択")
            
            self._reprocess_last_frame()
        else:
            self.sel_idx = None
            self._sel_bbox_prev = None
            self._status("特徴学習: OFF")

    @Slot(bool)
    def on_identification_toggled(self, on: bool):
        # 顔モードかつ検出ONでのみ有効
        if not (self._is_face_mode() and self.det_on):
            self.id_on = False
            return

        self.id_on = bool(on)
        if self.id_on:
            # 特徴学習と排他（UI側が disable するが念のため）
            self.feat_on = False
            # 埋め込み器の遅延初期化
            if self.embedder is None:
                try:
                    self.embedder = AdaFaceEmbedder(
                        ckpt_path=ADAFACE_CKPT_PATH,
                        architecture=ADAFACE_ARCH,
                        device=TORCH_DEVICE,
                        margin=ADAFACE_MARGIN,
                        out_size=ADAFACE_OUT_SIZE,
                        log_each=ADAFACE_LOG_EVERY,
                    )
                    self.log.info("[ADAFACE] initialized (%s).", TORCH_DEVICE)
                except Exception as e:
                    self.embedder = None
                    self.id_on = False
                    self._status("AdaFace初期化に失敗")
                    self.log.exception("AdaFace init failed: %s", e)
                    return
            self._status("顔識別: ON（バッファ平均と照合）")
            
            self._reprocess_last_frame()
        else:
            self._status("顔識別: OFF")

    # ====== フレーム処理 ======
    @Slot(object)
    def on_frame(self, frame_bgr: np.ndarray):
        """
        1フレーム処理：
          - 顔モード＆検出ONのときだけ YOLO-Face
          - 特徴学習ON：クリック選択した1名のみ AdaFace 埋め込みをバッファへ
          - 識別ON：各BBOXの AdaFace 埋め込みとギャラリー平均の cos 類似度を表示
        """
        
        self._last_frame = frame_bgr.copy()
        self._label_occupied = []
        if not self._is_face_mode():
            return
        if frame_bgr is None:
            return

        out = frame_bgr.copy()

        # --- 検出 ---
        if self.det_on and self.detector is not None:
            try:
                _det_in = frame_bgr.copy()   # 捨てバッファ
                boxes, scores = self.detector.detect(_det_in)
                self._last_boxes = np.array(boxes, dtype=np.float32) if boxes is not None else None
                self._last_scores = np.array(scores, dtype=np.float32) if scores is not None else None
            except Exception as e:
                self._last_boxes = None
                self._last_scores = None
                self.log.exception("[YOLO-FACE] infer failed: %s", e)

        # --- 描画（検出表示 / conf 表示のみ） ---
        if self._last_boxes is not None and len(self._last_boxes) > 0:
            self._draw_boxes(out, self._last_boxes, self._last_scores, color=BBOX_COLOR, show_score_label=not self.id_on)

        # --- 特徴学習（クリック選択された1名のみ） ---
        if self.feat_on and self.embedder is not None:
            if self._last_boxes is not None and len(self._last_boxes) > 0:
                # 過去の選択があれば IoU で追従して sel_idx を更新
                if self.sel_idx is not None:
                    self._update_selection_from_iou(self._last_boxes, iou_th=0.1)

                # 選択済みならその枠のみ埋め込み
                if self.sel_idx is not None and 0 <= self.sel_idx < len(self._last_boxes):
                    sel_box = self._last_boxes[self.sel_idx]
                    try:
                        feat = self.embedder.embed_one(out, sel_box)
                        if feat is not None:
                            self._feat_buf.append(feat.astype(np.float32, copy=False))
                            self.log.info("[ADAFACE] appended feat #%d: norm≈%.3f",
                                          len(self._feat_buf), float(np.linalg.norm(feat)))
                            # 選択枠を太線で強調
                            self._draw_selected_bbox(out, sel_box)
                    except Exception as e:
                        self.log.exception("[ADAFACE] embed failed: %s", e)

        # --- 識別（ギャラリー平均と cos 類似度） ---
        if self.id_on and self.embedder is not None:
            if self._last_boxes is not None and len(self._last_boxes) > 0:
                gal = self._get_gallery_vector()
                sims = None
                if gal is not None:
                    try:
                        feats = self.embedder.embed_many(out, self._last_boxes)  # (N, D)
                        sims = feats @ gal.astype(np.float32)                   # 内積=cos類似度（L2済）
                    except Exception as e:
                        sims = None
                        self.log.exception("[ADAFACE] batch embed failed: %s", e)

                # 上段に sim、下段に conf を描画（最高 sim を緑にするなどは好みで）
                self._draw_sim_and_conf(out, self._last_boxes, self._last_scores, sims)

        # 最後に画面へ反映（AppControllerが別途上書きしない前提）
        try:
            self.win.view.set_bgr(out)
        except Exception:
            pass

    # ====== クリック選択 ======
    @Slot(int, int)
    def on_frame_clicked(self, x: int, y: int):
        if not (self._is_face_mode() and self.det_on):
            return
        if self._last_boxes is None or len(self._last_boxes) == 0:
            self._status("対象が検出されていません")
            return

        idx = self._find_box_at(x, y, self._last_boxes)
        if idx is None:
            self._status("BBOX外をクリック（未選択）")
            return

        self.sel_idx = int(idx)
        self._sel_bbox_prev = self._last_boxes[self.sel_idx].copy()
        if self.feat_on:
            # 学習対象を切り替えたので、バッファはクリアしてその人に積む運用
            self._feat_buf.clear()
            self._status(f"選択: idx={self.sel_idx}（この人物のみ特徴学習）")
        else:
            self._status(f"選択: idx={self.sel_idx}")

    # ====== 付帯イベント ======
    @Slot()
    def on_app_closing(self):
        self._reset_runtime(light=False)

    @Slot(str)
    def on_file_selected(self, path: str):
        # 新規動画で 一旦状態を軽くリセット（選択や検出結果など）
        self._reset_runtime(light=True)
        self._status(f"読み込み完了: {path}")

    # ====== 内部ユーティリティ ======
    def _is_face_mode(self) -> bool:
        return (self.current_mode == FACE_MODE_NAME)

    def _status(self, text: str, ms: int = 1600):
        try:
            self.win.statusBar().showMessage(text, ms)
        except Exception:
            pass

    def _reset_runtime(self, light: bool = True):
        """実行時状態をリセット。light=False で推論器も解放。"""
        self._last_boxes = None
        self._last_scores = None
        self.sel_idx = None
        self._sel_bbox_prev = None
        self._feat_buf.clear()
        if not light:
            self.detector = None
            self.embedder = None

    # --- 描画系 ---
    def _draw_boxes(self, frame_bgr: np.ndarray, boxes: np.ndarray, scores: Optional[np.ndarray], color=(255, 0, 0), show_score_label: bool = True):
        if boxes is None or len(boxes) == 0:
            return
        H, W = frame_bgr.shape[:2]
        t = max(2, int(round(max(H, W) / 600)))
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
            # ★識別ONなら show_score_label=False が渡ってくるので、ここは描かない
            if show_score_label and (scores is not None and i < len(scores)):
                s = float(scores[i])
                label = f"conf={s:.2f}"
                self._draw_label_no_bg_avoid_overlap(
                    frame_bgr,
                    label,
                    (x1, y1, x2, y2),
                    color,                        # 文字色＝BBOX色
                    self._label_occupied,
                    font_scale=1 * max(1.0, (frame_bgr.shape[1] >= 3840 and frame_bgr.shape[0] >= 2160) and 1.25 or 1.0),
                    thickness=max(1, int(round((t - 1) * ( (frame_bgr.shape[1] >= 3840 and frame_bgr.shape[0] >= 2160) and 1.25 or 1.0)))) 
                )

    def _draw_selected_bbox(self, frame_bgr: np.ndarray, box: np.ndarray):
        x1, y1, x2, y2 = map(int, box)
        H, W = frame_bgr.shape[:2]
        t = max(3, int(round(max(H, W) / 400)))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), SELECTED_BBOX_COLOR, t, cv2.LINE_AA)

    def _draw_sim_and_conf(self,
                        frame_bgr: np.ndarray,
                        boxes: np.ndarray,
                        scores: Optional[np.ndarray],
                        sims: Optional[np.ndarray]):
        """
        BBOX 上段＝Sim（類似度）、下段＝Conf（信頼度）を描画。
        - 4K(>=3840x2160)では文字サイズと太さを25%アップ
        - ギャラリー未選択など sims/scores が無い場合でも "Sim ---" / "Conf ---" を表示
        - 最も高い Sim の枠は緑、その他は既定色（BBOX_COLOR or yellow）
        """
        if boxes is None or len(boxes) == 0:
            return

        H, W = frame_bgr.shape[:2]
        t = max(2, int(round(max(H, W) / 600)))  # BBOXの線の太さ
        # 4Kならラベルを少し大きく
        is_4k = (W >= 3840 and H >= 2160)
        scale = 1.25 if is_4k else 1.0
        fs_top = 1 * scale  # 上段(Sim)のフォントスケール
        fs_bot = 1 * scale  # 下段(Conf)のフォントスケール
        tt = max(1, int(round((t - 1) * scale)))  # テキストの太さ

        # 既定色（なければ黄）
        color_default = getattr(self, "BBOX_COLOR", (255, 0, 0))
        
        # ベストSimの枠だけ緑に
        best_idx = None
        if sims is not None:
            try:
                best_idx = int(np.argmax(np.asarray(sims).reshape(-1)))
            except Exception:
                best_idx = None

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b[:4])
            is_best = (best_idx is not None and i == best_idx)
            color = (0, 255, 0) if is_best else color_default

            # BBOX本体
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)

            # -------- 上段：Sim（枠の内側） --------
            sim_txt = f"sim={float(sims[i]):.2f}" if (sims is not None and i < len(sims)) else "sim=--"
            conf_txt = f"conf={float(scores[i]):.2f}" if (scores is not None and i < len(scores)) else "conf=--"
            label = f"{sim_txt}  {conf_txt}"

            self._draw_label_no_bg_avoid_overlap(
                frame_bgr,
                label,
                (x1, y1, x2, y2),
                color,                    # 文字色＝BBOX色（最良simなら色を変えていればその色）
                self._label_occupied,
                font_scale=1 * scale,   # ★ 4Kだと自動で少し大きく
                thickness=tt               # ★ BBOX太さに追随
            )

    # --- 選択補助 ---
    def _find_box_at(self, x: int, y: int, boxes: np.ndarray) -> Optional[int]:
        """クリック点(x,y)を内包するBBOXのうち、最も面積が小さいものを返す（重なり対策）"""
        hits = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                hits.append((area, i))
        if not hits:
            return None
        hits.sort(key=lambda t: t[0])
        return hits[0][1]

    def _update_selection_from_iou(self, boxes: np.ndarray, iou_th: float = 0.1):
        """前フレーム選択BBOXと最大IoUの検出に再アサインし sel_idx を更新"""
        if self._sel_bbox_prev is None or boxes is None or len(boxes) == 0:
            return
        prev = self._sel_bbox_prev
        x1, y1, x2, y2 = prev
        area_prev = max(1.0, (x2 - x1) * (y2 - y1))
        best_iou, best_i = 0.0, self.sel_idx
        for i, (a1, b1, a2, b2) in enumerate(boxes):
            xx1 = max(x1, a1); yy1 = max(y1, b1)
            xx2 = min(x2, a2); yy2 = min(y2, b2)
            w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
            inter = w * h
            area_i = max(1.0, (a2 - a1) * (b2 - b1))
            iou = inter / (area_prev + area_i - inter + 1e-6)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= iou_th:
            self.sel_idx = int(best_i)
            self._sel_bbox_prev = boxes[self.sel_idx].copy()
            
    def get_feature_buffer(self):
        """現在の学習バッファをリストで返す（各要素は np.ndarray）。コピーを返す。"""
        try:
            return list(self._feat_buf)
        except Exception:
            return []

    def clear_feature_buffer(self):
        """学習バッファをクリア。"""
        self._feat_buf.clear()

    # --- ギャラリー（バッファ平均） ---
    def _get_gallery_vector(self) -> Optional[np.ndarray]:
        if not self._feat_buf:
            return None
        gal = np.mean(self._feat_buf, axis=0).astype(np.float32)
        n = np.linalg.norm(gal) + 1e-6
        gal = gal / n
        return gal

    def _reprocess_last_frame(self):
        """
        直近のフレームを1回だけ即時処理する。
        再生中/一時停止中に関係なく、トグルON時に現在の絵から始められる。
        """
        if self._last_frame is None:
            logging.getLogger("app.adaface_ctrl").warning("[FACE] no last frame cached; cannot reprocess.")
            return

        # 既存の on_frame と同じ導線を通すのが一番安全
        try:
            self.on_frame(self._last_frame.copy())
        except Exception as e:
            logging.getLogger("app.adaface_ctrl").error("[FACE] reprocess failed: %s", e, exc_info=True)
            
    def set_identify_config(self, gallery_json_path: str, target_pid: str) -> None:
        """
        Playerの識別ダイアログで選ばれた設定を受け取るだけ（ここではロードしない）。
        実際のロードや照合は on_frame() 内の既存フローに任せる。
        """
        self._gallery_json_path = str(gallery_json_path or "").strip() or None
        self._target_pid = str(target_pid or "").strip() or None
        self.log.info("[FACE] identify-config set: gallery=%s, pid=%s",
                      self._gallery_json_path or "(default)", self._target_pid or "(all)")
        if self._gallery_json_path and self._target_pid:
            self.set_external_gallery(self._gallery_json_path, self._target_pid)


    def set_external_gallery(self, json_path: str, pid: str):
        vec = None
        try:
            with open(os.path.abspath(json_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            for e in (data if isinstance(data, list) else []):
                if str(e.get("pid","")) == str(pid):
                    v = np.asarray(e.get("feat", []), dtype=np.float32).reshape(-1)
                    n = float(np.linalg.norm(v)) + 1e-6
                    vec = v / n
                    break
        except Exception as e:
            self.log.exception("[ID] ギャラリー読込失敗: %s", e)

        self._fixed_gallery = vec
        self._feat_buf.clear()  # 誤学習防止のため内部バッファはクリア

        if vec is not None:
            self.log.info("[ID] ギャラリーをロード: pid=%s (%s)", str(pid), os.path.basename(str(json_path)))
        else:
            self.log.warning("[ID] 指定pidが見つかりません: pid=%s (%s)", str(pid), os.path.basename(str(json_path)))

    def _get_gallery_vector(self) -> Optional[np.ndarray]:
        if self._fixed_gallery is not None:
            return self._fixed_gallery
        if not self._feat_buf:
            return None
        gal = np.mean(self._feat_buf, axis=0).astype(np.float32)
        n = np.linalg.norm(gal) + 1e-6
        return gal / n
    
    def _draw_label_no_bg_avoid_overlap(
        self,
        img: np.ndarray,
        text: str,
        box_xyxy: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        occupied: List[Tuple[int, int, int, int]],
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> None:

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        x1, y1, x2, y2 = map(int, box_xyxy)
        H, W = img.shape[:2]
        # 基本はBBOXの左上に合わせて“上側”優先
        tx = max(0, min(x1, W - tw - 1))
        ty = y1 - 5
        if ty - th < 0:
            ty = min(H - 1, y2 + th + 5)

        def rect_for(txx, tyy):
            return (txx, tyy - th - baseline, txx + tw, tyy + baseline)

        def overlap(a, b):
            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

        # BBOX/既存ラベルと当たればずらす
        r = rect_for(tx, ty)
        tries = 0
        while (overlap(r, (x1, y1, x2, y2)) or any(overlap(r, o) for o in occupied)) and tries < 24:
            # まず下方向へ、下が詰まれば上方向へ
            if r[3] + (th + 6) < H:
                ty += (th + 6)
            else:
                ty = max(th + baseline + 2, ty - (th + 6))
            r = rect_for(tx, ty)
            tries += 1

        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        occupied.append(r)