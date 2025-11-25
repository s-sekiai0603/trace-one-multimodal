# -*- coding: utf-8 -*-
"""
player.py — OpenCV + PySide6(QGraphicsView)
- QMediaPlayer は不使用。OpenCV(cv2.VideoCapture)でフレーム取得
- フレームはそのまま AppController（旧InterfaceController）へ frameAvailable で渡す
- 表示は QGraphicsView/QGraphicsScene に QPixmap で行う（検出描画はコントローラ側）
- 要件対応：
  * 再生/一時停止の単一ボタン化（初期は一時停止状態で再生アイコン表示）
  * 再生/停止ボタン右にドロップダウン（顔/外見/歩容/組合せ）
  * 「検出」「特徴学習」「識別」をトグルボタン化（オン=緑）
  * 「特徴学習」と「識別」の間に「特徴保存」ボタン
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QCheckBox, QMessageBox, QStyle, QComboBox, QDialog, QInputDialog, QLineEdit,
)

from .identify import IdentifyDialog
from .feat_fuse_dialog import FeatFuseDialog
from ..config import VIDEO_ORIENTATION, AUTO_FACE_THRESH_DEFAULT, AUTO_APP_THRESH_DEFAULT, AUTO_GAIT_THRESH_DEFAULT

# --- 設定（DEFAULT_VIDEO_PATH が無い環境でも動くようフォールバック） ---
log = logging.getLogger("app.player")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
try:
    from ..config import DEFAULT_VIDEO_PATH
except Exception:
    DEFAULT_VIDEO_PATH = ""
    log.warning("動画が読み込めませんでした")

# ------------------------- ユーティリティ -------------------------

def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr is None or bgr.size == 0:
        return QImage()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def ms_to_hms(ms: int) -> str:
    s = max(0, int(round(ms / 1000.0)))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


# ------------------------- ビュー -------------------------

class VideoView(QGraphicsView):
    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)
        

        scn = QGraphicsScene(self)
        self._scene = scn
        self.setScene(scn)
        self._pix = QGraphicsPixmapItem()
        scn.addItem(self._pix)

    def set_qimage(self, qimg: QImage) -> None:
        if qimg is None or qimg.isNull():
            return
        self._pix.setPixmap(QPixmap.fromImage(qimg))
        self._fit_view()

    def set_bgr(self, bgr: np.ndarray) -> None:
        self.set_qimage(bgr_to_qimage(bgr))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_view()

    def _fit_view(self):
        pm = self._pix.pixmap()
        if pm.isNull():
            return
        self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, e):
        """
        クリック座標を元フレームのピクセル座標に変換して通知。
        """
        try:
            if self._pix is None or self._pix.pixmap().isNull():
                return super().mousePressEvent(e)

            scene_pos = self.mapToScene(e.position().toPoint())
            local = self._pix.mapFromScene(scene_pos)

            w = self._pix.pixmap().width()
            h = self._pix.pixmap().height()

            if 0 <= local.x() < w and 0 <= local.y() < h:
                self.clicked.emit(int(local.x()), int(local.y()))
        finally:
            return super().mousePressEvent(e)


# ------------------------- 本体 -------------------------

class PlayerWindow(QMainWindow):

    # シグナル
    detectionToggled = Signal(bool)
    featureLearningToggled = Signal(bool)
    identificationToggled = Signal(bool)
    identificationConfigSelected = Signal(str, str)
    featureFuseRequested = Signal(str, str, str, str)  # mode_key, label_a, label_b, base_label

    # その他
    frameAvailable = Signal(object)  # BGR フレーム
    fileSelected = Signal(str)
    frameClicked = Signal(int, int)
    featureBufferClearRequested = Signal()
    featureSaveRequested = Signal()
    modeChanged = Signal(str) 
    appClosing = Signal()
    autoHtmlThresholdChanged = Signal(float, float, float)
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TraceOne")
        self.resize(1280, 720)

        # 再生状態
        self.cap = None
        self._rotate_code = 0       # 0/90/180/270
        self._force_vflip = bool(VIDEO_ORIENTATION.get("FORCE_VFLIP", False))
        self.fps = 0.0
        self.frame_count = 0
        self.w = 0
        self.h = 0
        self.cur_idx = 0
        self.playing = False  # 初期は一時停止（＝再生アイコン表示）
        self.user_scrubbing = False
        self._scrub_was_playing = False
        self._last_frame_bgr = None
        self._last_raw_frame_bgr = None  
        
        self.current_video_path = ""

        # トグル状態
        self._det_on = False
        self._feat_on = False
        self._id_on = False

        # 画面
        self.view = VideoView(self)
        self.view.clicked.connect(self._on_view_clicked)

        # ラベル
        self.lbl_file = QLabel("ファイル未選択", self)
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_time = QLabel("00:00 / 00:00", self)

        # ファイル選択ボタン
        self.btn_open = QPushButton(self.style().standardIcon(QStyle.SP_DialogOpenButton), "開く...", self)
        self.btn_open.clicked.connect(self._on_open_clicked)

        # 再生/一時停止 単一ボタン
        self.btn_playpause = QPushButton(self)
        self._update_playpause_button_icon()  # 初期は「再生」アイコン
        self.btn_playpause.clicked.connect(self._on_playpause_clicked)

        # ドロップダウン：識別モード
        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItems([
            "マルチモーダル",
            "顔",
            "外見",
            "歩容",
        ])
        self.cmb_mode.setCurrentIndex(0)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)

        # トグル（ボタン版）
        self.btn_det = self._make_toggle_button("検出", self._on_det_toggle_clicked, checked=False)
        self.btn_feat = self._make_toggle_button("特徴学習", self._on_feat_toggle_clicked, checked=False)
        self.btn_save_feat = QPushButton("特徴保存", self)
        self.btn_save_feat.clicked.connect(self._on_save_feat_clicked)
        self.btn_id = self._make_toggle_button("識別", self._on_id_toggle_clicked, checked=False)

        # 特徴バッファ破棄ボタン（従来の挙動を維持：任意）
        self.btn_clear_feat = QPushButton("特徴破棄", self)
        self.btn_clear_feat.clicked.connect(self._on_clear_feat_clicked)
        self.btn_clear_feat.setEnabled(False)
        
        # 特徴統合ダイアログ起動ボタン
        self.btn_feat_fuse = QPushButton("特徴統合", self)
        self.btn_feat_fuse.clicked.connect(self._on_feat_fuse_clicked)
        
        self._update_enable_states()
        
        # --- マルチモーダルモード用 閾値テキストボックス（顔/外見/歩容） ---
        validator = QDoubleValidator(0.0, 1.0, 5, self)  # 0.00〜1.00 小数第2位まで

        self.edit_face_thresh = QLineEdit(self)
        self.edit_face_thresh.setFixedWidth(50)
        self.edit_face_thresh.setValidator(validator)
        self.edit_face_thresh.setText(f"{AUTO_FACE_THRESH_DEFAULT:.5f}")

        self.edit_app_thresh = QLineEdit(self)
        self.edit_app_thresh.setFixedWidth(50)
        self.edit_app_thresh.setValidator(validator)
        self.edit_app_thresh.setText(f"{AUTO_APP_THRESH_DEFAULT:.5f}")

        self.edit_gait_thresh = QLineEdit(self)
        self.edit_gait_thresh.setFixedWidth(50)
        self.edit_gait_thresh.setValidator(validator)
        self.edit_gait_thresh.setText(f"{AUTO_GAIT_THRESH_DEFAULT:.5f}")

        # 初期はマルチモーダルモードじゃないので無効にしておく
        self._on_mode_changed(self.cmb_mode.currentText())

        # スライダー
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)

        # レイアウト
        top = QWidget(self)
        v = QVBoxLayout(top); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)
        
        # 上部（動画の上、左端に「開く…」、右にファイル名）
        top_row = QHBoxLayout(); top_row.setSpacing(8)
        top_row.addWidget(self.btn_open)
        top_row.addWidget(self.lbl_file)
        top_row.addStretch(1)
        v.addLayout(top_row)
        
        # 動画ビュー
        v.addWidget(self.view, 1)

        bar = QHBoxLayout(); bar.setSpacing(8)
        bar.addWidget(self.btn_playpause)
        bar.addWidget(self.cmb_mode)  # 再生/一時停止ボタンの右
        bar.addSpacing(16)
        bar.addWidget(self.btn_det)
        bar.addWidget(self.btn_feat)
        bar.addWidget(self.btn_save_feat)  # 特徴学習と識別の間
        bar.addWidget(self.btn_id)
        
        bar.addWidget(QLabel("顔閾値:", self))
        bar.addWidget(self.edit_face_thresh)
        bar.addWidget(QLabel("外見閾値:", self))
        bar.addWidget(self.edit_app_thresh)
        bar.addWidget(QLabel("歩容閾値:", self))
        bar.addWidget(self.edit_gait_thresh)
        
        bar.addSpacing(16)
        bar.addWidget(self.btn_clear_feat)
        bar.addWidget(self.btn_feat_fuse)
        bar.addStretch(1)
        bar.addWidget(self.lbl_time)
        v.addLayout(bar)

        v.addWidget(self.slider)
        self.setCentralWidget(top)

        # タイマー
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.setInterval(33)  # FPS 確定後に調整
        
        self.btn_vflip = self._make_toggle_button("反転", self._on_vflip_clicked, checked=self._force_vflip)
        bar.addWidget(self.btn_vflip)

        v.addWidget(self.slider)
        self.setCentralWidget(top)

        # 起動時のデフォルト動画（停止状態で表示）
        self.initialize_default_video()

    # --------------- ヘルパ：トグルボタン ---------------

    def _make_toggle_button(self, text: str, slot, checked: bool = False) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setCheckable(True)
        btn.setChecked(bool(checked))
        self._apply_toggle_palette(btn, bool(checked))
        btn.clicked.connect(slot)
        return btn

    def _apply_toggle_palette(self, btn: QPushButton, on: bool):
        # オンの時だけ緑、オフはデフォルト
        if on:
            btn.setStyleSheet("QPushButton { background-color: #38a169; color: white; }")
        else:
            btn.setStyleSheet("")
            
    def _update_enable_states(self):
        """
        ボタン活性条件を一元管理：
        - 特徴学習：検出=ON かつ 識別=OFF
        - 識別    ：検出=ON かつ 特徴学習=OFF
        - 特徴保存：検出=ON かつ 特徴学習=ON
        """
        feat_enabled = (self._det_on and (not self._id_on))
        id_enabled   = (self._det_on and (not self._feat_on))
        save_enabled = (self._det_on and self._feat_on)

        self.btn_feat.setEnabled(feat_enabled)
        self.btn_id.setEnabled(id_enabled)
        self.btn_save_feat.setEnabled(save_enabled)


    # --------------- UIイベント ---------------

    def _on_playpause_clicked(self):
        if self.cap is None:
            return
        if self.playing:
            self.pause()
        else:
            self.play()

    def _on_mode_changed(self, text: str):
        try:
            self.modeChanged.emit(str(text))
            is_auto = (text.strip() == "マルチモーダル")
            self.edit_face_thresh.setEnabled(is_auto)
            self.edit_app_thresh.setEnabled(is_auto)
            self.edit_gait_thresh.setEnabled(is_auto)
        except Exception:
            pass

    def _on_det_toggle_clicked(self, on: bool):
        self._det_on = bool(on)
        self._apply_toggle_palette(self.btn_det, self._det_on)

        try:
            self.detectionToggled.emit(self._det_on)
        except Exception:
            pass

        self._update_enable_states()

        # --- 検出ONになったとき：一時停止中なら「このフレームで検出」 ---
        if self._det_on:
            if (not self.playing):
                raw = getattr(self, "_last_raw_frame_bgr", None)
                if raw is not None:
                    try:
                        frame = raw.copy()
                        if getattr(self, "_flip_enabled", False):
                            frame = cv2.flip(frame, -1)

                        self._last_frame_bgr = frame.copy()
                        self.view.set_bgr(self._last_frame_bgr)

                        # ここから先で YOLO-Face / YOLO-Pose が動く
                        self.frameAvailable.emit(frame)
                    except Exception:
                        pass

        # --- 検出OFFになったとき：現在の描画を全部消す ---
        else:
            # 再生中でも一瞬で消してOK
            self._redraw_last_raw_frame()

    def _on_feat_toggle_clicked(self, on: bool):
        self._feat_on = bool(on)
        self._apply_toggle_palette(self.btn_feat, self._feat_on)
        try:
            self.featureLearningToggled.emit(self._feat_on)
        except Exception:
            pass
        self._update_enable_states()

    def _on_save_feat_clicked(self):
        try:
            self.featureSaveRequested.emit()
        except Exception:
            pass
        
    def _on_clear_feat_clicked(self):
        """特徴抽出用の一時バッファを破棄（コントローラへ通知）"""
        try:
            self.featureBufferClearRequested.emit()
        except Exception:
            pass
        
    def _on_feat_fuse_clicked(self):
        """特徴統合ダイアログを開き、ユーザーの選択内容を通知する。"""
        try:
            dlg = FeatFuseDialog(self)
            if dlg.exec() == QDialog.Accepted:
                mode_key = dlg.selected_mode_key()
                label_a, label_b = dlg.selected_labels()
                base_label = dlg.new_label_base()
                log.info(
                    "[FEAT-FUSE] mode=%s, A=%s, B=%s, base=%s",
                    mode_key, label_a, label_b, base_label,
                )
                try:
                    self.featureFuseRequested.emit(mode_key, label_a, label_b, base_label)
                except Exception:
                    log.exception("featureFuseRequested.emit failed")
        except Exception:
            log.exception("FeatFuseDialog failed")

    def _on_id_toggle_clicked(self, on: bool):
        self._id_on = bool(on)
        self._apply_toggle_palette(self.btn_id, self._id_on)

        if self._id_on:
            mode_txt = self.cmb_mode.currentText() if hasattr(self, "cmb_mode") else ""

            # ===== マルチモーダル：文字入力で「名前+連番」を受ける =====
            if mode_txt == "マルチモーダル":
                dlg = IdentifyDialog(self, default_gallery=None, auto_mode=True, auto_labels =None)
                if dlg.exec() == QDialog.Accepted:
                    label = dlg.selected_pid()  # “名前+連番”がここに入る
                    if label:
                        try:
                            def _safe_parse(edit: QLineEdit, default: float) -> float:
                                try:
                                    v = float(edit.text())
                                except Exception:
                                    v = default
                                # 0〜1 にクリップし、小数第2位で表示を揃える
                                v = max(0.0, min(1.0, v))
                                edit.setText(f"{v:.5f}")
                                return v

                            face = _safe_parse(self.edit_face_thresh, AUTO_FACE_THRESH_DEFAULT)
                            app  = _safe_parse(self.edit_app_thresh,  AUTO_APP_THRESH_DEFAULT)
                            gait = _safe_parse(self.edit_gait_thresh, AUTO_GAIT_THRESH_DEFAULT)

                            # AppController → AutoController.set_html_thresholds(...) へ届く
                            try:
                                self.autoHtmlThresholdChanged.emit(face, app, gait)
                            except Exception:
                                pass
                        except Exception:
                            # 閾値通知まわりで何かあっても識別自体は継続
                            try:
                                log.exception("[ID][AUTO] failed to update HTML thresholds")
                            except Exception:
                                pass
                        try: self.identificationToggled.emit(True)
                        except: pass
                        try: self.identificationConfigSelected.emit("", label)  # gpath未使用
                        except: pass
                        try: log.info("[ID][AUTO] target_label=%s", label)
                        except: pass
                    else:
                        self._id_on = False
                        self._apply_toggle_palette(self.btn_id, self._id_on)
                        self._update_enable_states()
                        return
                else:
                    self._id_on = False
                    self._apply_toggle_palette(self.btn_id, self._id_on)
                    self._update_enable_states()
                    return

            # ===== マルチモーダル以外：従来の IdentifyDialog =====
            else:
                dlg = IdentifyDialog(self)
                if dlg.exec() == QDialog.Accepted:
                    gpath = dlg.selected_gallery_path()
                    pid   = dlg.selected_pid()
                    # 1) 設定
                    try:
                        self.identificationConfigSelected.emit(str(gpath), str(pid))
                    except Exception:
                        pass
                    # 2) ログ
                    try:
                        log.info("[ID] start on current video: %s",
                                self.current_video_path or "(none)")
                    except Exception:
                        pass
                    # 3) ON 通知
                    try:
                        self.identificationToggled.emit(True)
                    except Exception:
                        pass
                else:
                    # キャンセル → OFFへ戻す
                    self._id_on = False
                    self._apply_toggle_palette(self.btn_id, self._id_on)
                    self._update_enable_states()
                    return

        else:
            # OFF 通知（共通）
            try:
                self.identificationToggled.emit(False)
            except Exception:
                pass

        self._update_enable_states()



    @Slot()
    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画ファイルを選択", "",
            "Videos (*.mp4 *.mkv *.mov *.avi);;All Files (*)"
        )
        if not path:
            return
        if self._open_video(path, emit_signal=True):
            # 自動再生はしない（先頭フレームだけ表示）
            self.pause()
            if self.timer.isActive():
                self.timer.stop()

    def _on_view_clicked(self, x: int, y: int):
        try:
            self.frameClicked.emit(x, y)
        except Exception:
            pass

    # --------------- 再生制御 ---------------

    def _update_playpause_button_icon(self):
        if self.playing:
            self.btn_playpause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.btn_playpause.setText("一時停止")
        else:
            self.btn_playpause.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.btn_playpause.setText("再生")

    def play(self):
        if self.cap is None:
            return
        self.playing = True
        self._update_playpause_button_icon()
        if not self.timer.isActive():
            self.timer.start()

    def pause(self):
        self.playing = False
        self._update_playpause_button_icon()

    def _seek_frame(self, idx: int):
        if self.cap is None:
            return
        idx = max(0, int(idx))
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if not ok:
            ms = int(round((idx / max(self.fps, 1e-6)) * 1000.0))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        self.cur_idx = idx
        self.slider.blockSignals(True); self.slider.setValue(idx); self.slider.blockSignals(False)
        self._update_time_label()
        
    def _seek_and_preview(self, idx: int):
        """
        指定フレームへシークし、そのフレームを即表示（再生状態は変更しない）。
        ※ read() を1回行うため、内部の cur_idx は実質 idx+1 になる（UIは見た目優先）。
        """
        if self.cap is None:
            return
        idx = max(0, int(idx))
        # 位置指定（フレーム指定が効かないコーデックもあるのでMSEC fallbackは _seek_frame に準拠）
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if not ok:
            ms = int(round((idx / max(self.fps, 1e-6)) * 1000.0))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, ms)

        # 1フレーム読み出して表示（回転補正は _on_tick に合わせる）
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        # 生→補正→表示の順に統一
        self._last_raw_frame_bgr = frame
        oriented = self._apply_orientation(frame)
        self._last_frame_bgr = oriented
        self.view.set_bgr(oriented)

        # UIの数字も反映
        self.cur_idx = idx + 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.cur_idx)
        self.slider.blockSignals(False)
        self._update_time_label(self.cur_idx)

        # コントローラ側にもプレビューを送る（必要なら）
        try:
            self.frameAvailable.emit(self._last_frame_bgr)
        except Exception:
            pass


    # --------------- 状態変更API（コントローラから操作したい時に使用） ---------------

    def setDetectionChecked(self, checked: bool):
        self.btn_det.setChecked(bool(checked))
        self._on_det_toggle_clicked(bool(checked))

    def setFeatureLearningChecked(self, checked: bool):
        self.btn_feat.setChecked(bool(checked))
        self._on_feat_toggle_clicked(bool(checked))

    def setIdentificationChecked(self, checked: bool):
        self.btn_id.setChecked(bool(checked))
        self._on_id_toggle_clicked(bool(checked))

    def setClearFeatEnabled(self, enabled: bool):
        self.btn_clear_feat.setEnabled(bool(enabled))

    # --------------- タイマー駆動 ---------------

    def _on_tick(self):
        if not self.playing or self.cap is None or self.user_scrubbing:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.pause()
            self.timer.stop()
            return

        self.cur_idx += 1

        # 1) 未補正フレームを保持
        self._last_raw_frame_bgr = frame

        # 2) 向き補正（_open_video で決めた self._rotate_code と現在の _force_vflip を適用）
        oriented = self._apply_orientation(frame)

        # 3) 表示用も通知用も『補正後』で統一
        self._last_frame_bgr = oriented
        self.view.set_bgr(oriented)

        # UI
        self.slider.blockSignals(True); self.slider.setValue(self.cur_idx); self.slider.blockSignals(False)
        self._update_time_label()

        # 外部へ：補正後を渡す（描画座標系の一貫性を保つ）
        try:
            self.frameAvailable.emit(oriented)
        except Exception:
            pass

    # --------------- スライダー ---------------

    def _on_slider_pressed(self):
        # ドラッグ開始：今の再生状態を記憶して一時停止（掴みやすく）
        self.user_scrubbing = True
        self._scrub_was_playing = self.playing
        if self.timer.isActive():
            self.timer.stop()
        # 再生中でも一旦停止表示にして“競合”を避ける（押しやすさUP）
        if self.playing:
            self.pause()

    def _on_slider_released(self):
        # ドラッグ終了：その位置へ確定シーク＆プレビュー
        self.user_scrubbing = False
        idx = int(self.slider.value())
        self._seek_and_preview(idx)
        # もともと再生中だったら再開（体験維持）
        if self._scrub_was_playing:
            self.play()
        self._scrub_was_playing = False

    def _on_slider_moved(self, v: int):
        # 一時停止中は“動かすたびに”即プレビュー
        if not self.playing:
            self._seek_and_preview(int(v))
        else:
            # 再生中でもドラッグ中はプレビューしてOK（重い時はここをコメントアウト）
            if self.user_scrubbing:
                self._seek_and_preview(int(v))
            else:
                # 非ドラッグの自動移動時は時間ラベルだけ更新
                self._update_time_label(v)

    def _update_time_label(self, idx: Optional[int] = None):
        if idx is None:
            idx = self.cur_idx
        ms_cur = int(round((idx / max(self.fps, 1e-6)) * 1000.0))
        if self.frame_count > 0:
            ms_tot = int(round((self.frame_count / max(self.fps, 1e-6)) * 1000.0))
            self.lbl_time.setText(f"{ms_to_hms(ms_cur)} / {ms_to_hms(ms_tot)}")
        else:
            self.lbl_time.setText(f"{ms_to_hms(ms_cur)} / 00:00")
            
    def get_position_sec(self) -> float:
        """
        現在の動画位置（秒）を返す。
        AutoController._get_video_time_label から呼ばれる想定。
        """
        return float(self.cur_idx) / max(self.fps or 1.0, 1.0)

    def get_current_frame(self) -> int:
        """
        現在のフレーム番号（0-based）を返す。
        フォールバック用（現状は get_position_sec が優先して使われる）。
        """
        return int(self.cur_idx)

    def get_fps(self) -> float:
        """
        FPS を返す。
        フォールバック用（get_current_frame とセットで使われる）。
        """
        return float(self.fps or 0.0)

    # --------------- ファイルIO ---------------

    def _open_video(self, path: str, emit_signal: bool = False) -> bool:
        # 既存リソースを解放
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "エラー", "この動画を開けませんでした。")
            return False

        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # --- 向きの決定（メタデータ＋設定で確定） ---
        rotate_code = 0

        # 1) メタデータ優先（AUTO_ROTATE=True のときだけ）
        if bool(VIDEO_ORIENTATION.get("AUTO_ROTATE", True)):
            prop_rot = getattr(cv2, "CAP_PROP_ROTATION", None)
            if prop_rot is not None:
                try:
                    v = int(self.cap.get(prop_rot) or -1)
                    if v in (0, 90, 180, 270):
                        rotate_code = v
                except Exception:
                    pass
            if rotate_code == 0:
                prop_orient = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
                if prop_orient is not None:
                    try:
                        val = int(self.cap.get(prop_orient) or -1)
                        # 一部環境の値→角度への緩やかな対応
                        if val in (90, 180, 270):
                            rotate_code = val
                        elif val == 1:
                            rotate_code = 90
                        elif val == 2:
                            rotate_code = 180
                        elif val == 3:
                            rotate_code = 270
                    except Exception:
                        pass

        # 2) 設定による上書き（0以外なら強制）
        conf_deg = int(VIDEO_ORIENTATION.get("ROTATE_DEG", 0) or 0)
        if conf_deg in (90, 180, 270):
            rotate_code = conf_deg

        self._rotate_code = rotate_code
        # 上下反転は __init__ で config を取り込み済み（手動で切り替えるなら別メソッドで変更）

        # 先頭フレーム準備
        self.cur_idx = 0
        self.slider.setRange(0, max(0, self.frame_count - 1))
        self._seek_frame(0)

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._last_raw_frame_bgr = frame
            frame = self._apply_orientation(frame)
            self._last_frame_bgr = frame.copy()
            self.view.set_bgr(self._last_frame_bgr)
            self.cur_idx = 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.cur_idx)
            self.slider.blockSignals(False)
            self._update_time_label()
            self.lbl_file.setText(os.path.basename(path))
            if emit_signal and hasattr(self, "frameAvailable"):
                self.frameAvailable.emit(self._last_frame_bgr)
        else:
            self._last_frame_bgr = None

        return True

    def initialize_default_video(self):
        if getattr(self, "current_video_path", ""):
            log.info("[INIT] 既に動画を開いているため初期動画はスキップ: %s", self.current_video_path)
            return
        log.info("[INIT] DEFAULT_VIDEO_PATH=%r", DEFAULT_VIDEO_PATH)
        if not DEFAULT_VIDEO_PATH:
            log.info("[INIT] 既定の動画パスが設定されていません（スキップ）")
            return
        try:
            if not os.path.exists(DEFAULT_VIDEO_PATH):
                log.warning("[INIT] 既定の動画が見つかりません: %s", DEFAULT_VIDEO_PATH)
                return

            ok = self._open_video(DEFAULT_VIDEO_PATH, emit_signal=True)
            log.info("[INIT] 既定の動画をオープン: %s", "OK" if ok else "NG")

            # 自動再生はしない（先頭フレームのみ表示）
            self.pause()
            if self.timer.isActive():
                self.timer.stop()
        except Exception as e:
            log.exception("[INIT] 既定動画の読み込みで例外: %s", e)


    # --------------- 終了処理 ---------------

    def closeEvent(self, e):
        """
        アプリ終了時：
        - タイマー停止
        - OpenCVリソース解放
        - 特徴バッファ破棄通知
        """
        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        try:
            self.appClosing.emit()
        except Exception:
            pass

        super().closeEvent(e)

    def _apply_orientation(self, frame):
        if frame is None:
            return frame

        code = int(getattr(self, "_rotate_code", -1) or 0)
        if code == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif code == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif code == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ↑回転のあと
        if bool(getattr(self, "_force_vflip", False)):
            frame = cv2.flip(frame, -1)  # ←両軸反転
        return frame
    
    def _on_vflip_clicked(self, on: bool):
        """下部バーの『上下反転』トグルを押した時。生フレームから再適用して即再描画。"""
        self._force_vflip = bool(on)
        # 直近の生フレームから再適用して表示＆通知
        if getattr(self, "_last_raw_frame_bgr", None) is not None:
            oriented = self._apply_orientation(self._last_raw_frame_bgr.copy())
            self._last_frame_bgr = oriented
            self.view.set_bgr(oriented)
            try:
                self.frameAvailable.emit(oriented)
            except Exception:
                pass
            
    def _redraw_last_raw_frame(self):
        """最後に読み込んだ「生フレーム」から表示を描き直す"""
        raw = getattr(self, "_last_raw_frame_bgr", None)
        if raw is None:
            return

        frame = raw.copy()

        # 反転ONなら適用（あなたの実装に合わせて）
        if getattr(self, "_flip_enabled", False):
            frame = cv2.flip(frame, -1)

        # 表示用フレームとして保存 & 表示
        self._last_frame_bgr = frame.copy()
        self.view.set_bgr(self._last_frame_bgr)