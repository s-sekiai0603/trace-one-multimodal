# -*- coding: utf-8 -*-
"""
AppController — プレゼンテーション層（UI）とアプリケーションロジックの橋渡し
- PlayerWindow からのシグナル（再生フレーム／トグル／モード変更／クリック等）を受ける
- 将来の pipelines.realtime や各モデル用 controller を“あれば使う、なければ安全にスキップ”
- 受け取ったフレームに対し、必要に応じて検出結果の描画などを実施して UI へ再表示

【ポイント】
- 依存モジュールは try/except で“オプショナルに”読み込み（未実装でも動く）
- UI とは疎結合：UI ウィジェットへはシグナル経由／描画は set_bgr() を上書きするだけ
- ログは "app.interface" 名で集約（既存ログと整合性を取りやすくする）
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
from PySide6.QtWidgets import QInputDialog, QLineEdit
import logging
import numpy as np
import cv2

from .. import config
from ..db.feature import save_face_features
from ..db.feat_fuse import fuse_face_auto, fuse_app_auto, fuse_gait_auto, fuse_all_auto
from ..utils.preprocess_video import VideoPreprocessor
from .ada_face_controller import AdaFaceController
from .osnet_controller import OsnetController
from .gait_mixer_controller import GaitMixerController
from .auto_controller import AutoController

log = logging.getLogger("app.interface")

# --- オプショナル依存：存在すれば使う ---
RealtimePipeline = None
try:
    from pipelines.realtime import RealtimePipeline  # type: ignore
except Exception:
    pass


# -------------------------------------------------
# ユーティリティ（軽量な描画／凡例）
# -------------------------------------------------

_GREEN = (40, 160, 105)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_GRAY = (180, 180, 180)

def _put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=_WHITE) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def _draw_boxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=_GREEN) -> None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# -------------------------------------------------
# AppController 本体
# -------------------------------------------------

class AppController:
    """
    PlayerWindow からのイベントを受け取り、処理フローを調停するクラス。
    - 検出／特徴学習／識別の ON/OFF
    - モード（顔／外見／歩容／組合せ）の切替
    - フレーム受信 → 必要に応じてパイプライン実行 → 描画して UI へ返す
    """
    def __init__(self, ui_window):
        self.ui = ui_window

        # 状態
        self.detect_on: bool = False
        self.feature_on: bool = False  # 「特徴学習」
        self.identify_on: bool = False
        self.mode_text: str = "マルチモーダル"     # ドロップダウンの現在値
        self._last_raw: Optional[np.ndarray] = None
        self._click_xy: Optional[Tuple[int, int]] = None
        
        self._last_frame: Optional[np.ndarray] = None

        # バッファ（例：顔のギャラリー、特徴保管など）— 将来的にDB/FAISSと繋ぐ
        self._feature_buffer: Dict[str, Any] = {}
        
        # 前回のベースID
        self._pid_hint: Optional[str] = None

        # オプショナル依存の初期化
        self.pipeline = None
        if RealtimePipeline is not None:
            try:
                self.pipeline = RealtimePipeline()
                log.info("[PIPELINE] RealtimePipeline initialized.")
            except Exception as e:
                log.warning("[PIPELINE] RealtimePipeline init failed: %s", e)

        self.ada = None
        if AdaFaceController is not None:
            try:
                self.ada = AdaFaceController(self.ui)
                self.ada.set_mode(self.mode_text)
                log.info("[ADAFACE] controller initialized.")
            except Exception as e:
                log.warning("[ADAFACE] controller init failed: %s", e)

        self.osnet = None
        if OsnetController is not None:
            try:
                self.osnet = OsnetController(self.ui)
                self.osnet.set_mode(self.mode_text)
                log.info("[OSNET] controller initialized.")
            except Exception as e:
                log.warning("[OSNET] controller init failed: %s", e)

        self.gait = None
        if GaitMixerController is not None:
            try:
                self.gait = GaitMixerController(self.ui)
                self.gait.set_mode(self.mode_text)
                log.info("[GAIT] controller initialized.")
            except Exception as e:
                log.warning("[GAIT] controller init failed: %s", e)
                
        self.auto = None
        if AutoController is not None:
            try:
                self.auto = AutoController(self.ui)
                self.auto.set_mode(self.mode_text)
                log.info("[AUTO] controller initialized.")
            except Exception as e:
                log.warning("[AUTO] controller init failed: %s", e)

        # UI シグナル配線
        self._connect_signals()

        # UI 初期状態（ボタン活性など）
        self.ui.setClearFeatEnabled(False)
        
        self._force_ui_initial_state()
        self._sync_enables()
        
        self._preproc = None
        try:
            self._preproc = VideoPreprocessor(config.__dict__)
        except Exception as e:
            log.warning("[PREPROC] disabled: %s", e)

        log.info("[TOGGLE] 検出=%s, 特徴学習=%s, 識別=%s, モード=%s",
                 self.detect_on, self.feature_on, self.identify_on, self.mode_text)

    # ----------------------- signal wiring -----------------------

    def _connect_signals(self):
        # 再生フレーム
        self.ui.frameAvailable.connect(self.on_frame)

        # クリック（人物選択などで利用予定）
        self.ui.frameClicked.connect(self.on_frame_clicked)

        # トグル
        self.ui.detectionToggled.connect(self.on_detection_toggled)
        self.ui.featureLearningToggled.connect(self.on_feature_learning_toggled)
        self.ui.identificationToggled.connect(self.on_identification_toggled)
        self.ui.identificationConfigSelected.connect(self.on_identification_config_selected)

        # モード変更
        self.ui.modeChanged.connect(self.on_mode_changed)

        # 特徴保存要求
        self.ui.featureSaveRequested.connect(self.on_feature_save_requested)

        # 特徴バッファのクリア要求
        self.ui.featureBufferClearRequested.connect(self.on_feature_buffer_clear)
        
        # 特徴統合要求
        if hasattr(self.ui, "featureFuseRequested"):
            self.ui.featureFuseRequested.connect(self.on_feature_fuse_requested)

        # アプリ終了
        self.ui.appClosing.connect(self.on_app_closing)

        # ファイル選択
        self.ui.fileSelected.connect(self.on_file_selected)
        
        if hasattr(self.ui, "autoHtmlThresholdChanged"):
            self.ui.autoHtmlThresholdChanged.connect(self.on_auto_html_threshold_changed)

    # ----------------------- event handlers -----------------------

    def on_mode_changed(self, text: str):
        """
        モード変更時：
        - 各コントローラに set_mode を通知
        - 現在のUIトグル状態（検出 / 特徴学習 / 識別）を
          「新しいモードのコントローラ」にも再適用する
          → 歩容モードに切り替えたときに det_on / id_on がズレないようにする
        """
        self.mode_text = (text or "").strip()

        # 1) コントローラにモードを通知
        for ctrl in (getattr(self, "ada", None),
                     getattr(self, "osnet", None),
                     getattr(self, "gait", None),
                    getattr(self, "auto", None)):
            if ctrl and hasattr(ctrl, "set_mode"):
                ctrl.set_mode(self.mode_text)
        log.info("[MODE] %s", self.mode_text)

        # 2) 現在のトグル状態を UI から取得
        det_on  = self._is_checked("btn_det")
        feat_on = self._is_checked("btn_feat")
        id_on   = self._is_checked("btn_id")

        # 3) 新しいモードに対してトグル状態を再適用
        #    （ここで on_detection_toggled() を呼ぶことで、
        #      歩容モードなら GaitMixerController.on_detection_toggled(det_on)
        #      が必ず一度は実行される）
        self.on_detection_toggled(det_on)

        # 特徴学習・識別も同様に、現在のチェック状態を再適用
        self.on_feature_learning_toggled(feat_on)
        self.on_identification_toggled(id_on)

        # 4) UI活性状態の更新
        self._sync_enables()


    def on_detection_toggled(self, on: bool):
        m = (self.mode_text or "").strip()
        log.info("[TOGGLE] 検出=%s (mode=%s)", on, m)

        if m == "顔" and getattr(self, "ada", None) and hasattr(self.ada, "on_detect_toggle"):
            self.ada.on_detect_toggle(on)
        elif m == "外見" and getattr(self, "osnet", None) and hasattr(self.osnet, "on_detect_toggle"):
            self.osnet.on_detect_toggle(on)
        elif m == "歩容" and getattr(self, "gait", None):
            # ★ GaitMixerController は on_detection_toggled を持っている
            if hasattr(self.gait, "on_detection_toggled"):
                self.gait.on_detection_toggled(on)
        elif m == "マルチモーダル" and getattr(self, "auto", None):
            if hasattr(self.auto, "on_detection_toggled"):
                self.auto.on_detection_toggled(on)
        else:
            log.info("[TOGGLE] 検出=%s (no controller for mode=%s)", on, m)


    def on_feature_learning_toggled(self, on: bool):
        self.feature_on = bool(on)
        self.ui.setClearFeatEnabled(self.feature_on)
        m = (self.mode_text or "").strip()
        log.info("[TOGGLE] 特徴学習=%s (mode=%s)", self.feature_on, m)

        if m == "顔" and getattr(self, "ada", None) and hasattr(self.ada, "on_feature_learning_toggled"):
            self.ada.on_feature_learning_toggled(self.feature_on)

        elif m == "外見" and getattr(self, "osnet", None) and hasattr(self.osnet, "on_feature_learning_toggled"):
            self.osnet.on_feature_learning_toggled(self.feature_on)

        elif m == "歩容" and getattr(self, "gait", None) and hasattr(self.gait, "on_feature_learning_toggled"):
            # ★ 歩容コントローラへ「学習ON/OFF」を通知
            self.gait.on_feature_learning_toggled(self.feature_on)
            
        elif m == "マルチモーダル" and getattr(self, "auto", None) and hasattr(self.auto, "on_feature_toggled"):
            # ★ 歩容コントローラへ「学習ON/OFF」を通知
            self.auto.on_feature_toggled(self.feature_on)

        else:
            log.info("[TOGGLE] 特徴学習=%s (no controller for mode=%s)", self.feature_on, m)

        self._sync_enables()

    def on_identification_toggled(self, on: bool):
        m = (self.mode_text or "").strip()
        log = logging.getLogger("app.interface")
        on = bool(on)

        # 1) 現在モードのコントローラだけ ON にする
        if m == "顔" and getattr(self, "ada", None) and hasattr(self.ada, "on_identify_toggle"):
            self.ada.on_identify_toggle(on)
        elif m == "外見" and getattr(self, "osnet", None) and hasattr(self.osnet, "on_identify_toggle"):
            self.osnet.on_identify_toggle(on)
        elif m == "歩容" and getattr(self, "gait", None):
            self.gait.on_identification_toggled(on)
        elif m == "マルチモーダル" and getattr(self, "auto", None):
            self.auto.on_identification_toggled(on)

        else:
            log.info("[TOGGLE] 識別=%s (no controller for mode=%s)", on, m)

        # 2) 他モードのコントローラは強制OFF（漏れを遮断）
        for name, ctrl in (("顔", getattr(self, "ada", None)),
                        ("外見", getattr(self, "osnet", None)),
                        ("歩容", getattr(self, "gait", None)),
                        ("マルチモーダル", getattr(self, "auto", None))):
            if name == m:
                continue
            if not ctrl:
                continue
            try:
                # ★ ここを修正：identify_toggle を持っていなくても、identification_toggled を持っていればOFFする
                if hasattr(ctrl, "on_identification_toggled"):
                    ctrl.on_identification_toggled(False)
                elif hasattr(ctrl, "on_identify_toggle"):
                    ctrl.on_identify_toggle(False)
            except Exception:
                log.exception("[TOGGLE] 識別OFF伝播失敗: %s", name)

        self._sync_enables()

    def on_feature_save_requested(self):
        """
        顔モードの学習バッファを JSON に保存する。
        - P-ID はダイアログで入力（キャンセルなら中断）
        - 保存成功時はファイルパスを statusBar とログに通知
        """
        # 顔モードかつ AdaFaceController が有効かを確認
        mode = (self.mode_text or "").strip()
        if mode == "顔" and self.ada and hasattr(self.ada, "save_feature_buffer_dialog"):
            self.ada.save_feature_buffer_dialog(); return

        if mode == "外見" and self.osnet and hasattr(self.osnet, "save_feature_buffer_dialog"):
            self.osnet.save_feature_buffer_dialog(); return

        if mode == "歩容" and self.gait:
            if hasattr(self.gait, "on_feature_save_requested"):
                self.gait.on_feature_save_requested(); return
                
        if mode == "マルチモーダル" and getattr(self, "auto", None):
            # ★ 毎回ラベルを確認する（常にダイアログを出す）
            current_label = None
            for attr in ("target_label", "_target_label"):
                if hasattr(self.auto, attr) and getattr(self.auto, attr):
                    current_label = getattr(self.auto, attr)
                    break

            text_default = str(current_label) if current_label else ""

            label, ok = QInputDialog.getText(
                self.ui,
                "保存ラベルを入力",
                "",
                text="",
            )
            if not ok or not str(label).strip():
                log.info("[AUTO][SAVE] キャンセル（ラベル未指定）")
                return

            if hasattr(self.auto, "set_target_label"):
                try:
                    self.auto.set_target_label(str(label).strip())
                except Exception:
                    # フォールバック：属性として持つだけでもOK
                    try:
                        setattr(self.auto, "_target_label", str(label).strip())
                    except Exception:
                        pass

            # 実保存
            if hasattr(self.auto, "on_feature_save_requested"):
                self.auto.on_feature_save_requested()
            return


        # コントローラから学習バッファを取得
        try:
            feats = self.ada.get_feature_buffer()
        except Exception:
            feats = []

        if not feats:
            log.info("[SAVE] バッファが空です。特徴学習をオンにして、対象BBOXをクリックしてから保存してください。")
            try:
                self.ui.statusBar().showMessage("バッファが空です（学習→クリック→保存の順）", 1800)
            except Exception:
                pass
            return

        # P-ID 入力
        pid, ok = QInputDialog.getText(
            self.ui,
            "保存ラベルを入力",
            "",
            text=(self._pid_hint or "")
        )
        if not ok or not str(pid).strip():
            return
        pid = str(pid).strip()
        self._pid_hint = pid

        # JSON 保存
        try:
            json_path = save_face_features(pid=pid, feats=feats, json_path=None)
            msg = f"[SAVE] 特徴を保存しました → {json_path}"
            log.info(msg)
            try:
                self.ui.statusBar().showMessage(f"保存完了: {json_path}", 2500)
            except Exception:
                pass
        except Exception as e:
            log.error("[SAVE] 失敗: %s", e, exc_info=True)
            try:
                self.ui.statusBar().showMessage(f"保存失敗: {e}", 2500)
            except Exception:
                pass

    def on_feature_buffer_clear(self):
        self._feature_buffer.clear()
        log.info("[FEATURE] バッファをクリアしました。")
        
    def on_auto_html_threshold_changed(self, face: float, app: float, gait: float):
        """
        UI から送られてきたマルチモーダルモード用 HTML ランキングしきい値を AutoController に反映。
        """
        if getattr(self, "auto", None) and hasattr(self.auto, "set_html_thresholds"):
            self.auto.set_html_thresholds(face, app, gait)
            
    def on_feature_fuse_requested(
        self,
        mode_key: str,
        label_a: str,
        label_b: str,
        base_label: str,
    ):
        """
        PlayerWindow からの「特徴統合」リクエストを受けて、
        db.feat_fuse のユーティリティを呼び出す。
        
        mode_key: "face" / "appearance" / "gait" / "all"
        label_a, label_b: 統合対象ラベル
        base_label: 新ラベルのベース名（例: "Sasakura-fused"）
        """
        try:
            # ラベルをユニーク化して2本以上あることを確認
            src_labels = [s for s in {label_a, label_b} if s]
            if len(src_labels) < 2:
                log.warning("[FEAT-FUSE] invalid src_labels: %s", src_labels)
                return

            base_label = (base_label or "").strip()
            if not base_label:
                log.warning("[FEAT-FUSE] empty base_label; skip.")
                return

            m = (mode_key or "").strip().lower()
            if m == "face":
                path, fused_label = fuse_face_auto(src_labels, base_label)
                log.info(
                    "[FEAT-FUSE][FACE] fused %s -> %s (path=%s)",
                    src_labels, fused_label, path,
                )

            elif m == "appearance":
                path, fused_label = fuse_app_auto(src_labels, base_label)
                log.info(
                    "[FEAT-FUSE][APP] fused %s -> %s (path=%s)",
                    src_labels, fused_label, path,
                )

            elif m == "gait":
                path, fused_label = fuse_gait_auto(src_labels, base_label)
                log.info(
                    "[FEAT-FUSE][GAIT] fused %s -> %s (path=%s)",
                    src_labels, fused_label, path,
                )

            elif m == "all":
                result = fuse_all_auto(src_labels, base_label)
                log.info(
                    "[FEAT-FUSE][ALL] fused %s -> %s (face=%s, app=%s, gait=%s)",
                    src_labels,
                    result.get("label"),
                    result.get("face_path"),
                    result.get("app_path"),
                    result.get("gait_path"),
                )

            else:
                log.warning("[FEAT-FUSE] unknown mode_key=%s", mode_key)
                return

            # ★ 将来的に:
            # AutoController側に「ギャラリー再読込」などのフックを用意したら、
            # ここで self.auto.reload_galleries() 的な呼び出しを入れるとさらに親切。
        except Exception as e:
            log.exception("[FEAT-FUSE] failed: %s", e)
        
    def on_identification_config_selected(self, gallery_json_path: str, target_pid: str):
        mode = (self.mode_text or "").strip()

        try:
            if mode == "顔" and getattr(self, "ada", None) and hasattr(self.ada, "set_identify_config"):
                self.ada.set_identify_config(gallery_json_path, target_pid)
                log.info("[ID] gallery=%s, target_pid=%s (mode=顔)", gallery_json_path, target_pid or "(all)")
                return

            if mode == "外見":
                if getattr(self, "osnet", None) and hasattr(self.osnet, "set_identify_config"):
                    self.osnet.set_identify_config(gallery_json_path, target_pid)
                    log.info("[ID] gallery=%s, target_pid=%s (mode=外見)", gallery_json_path, target_pid or "(all)")
                else:
                    log.info("[ID] gallery=%s, target_pid=%s (mode=外見; controller lacks set_identify_config)",
                             gallery_json_path, target_pid or "(all)")
                return

            if mode == "歩容":
                if getattr(self, "gait", None) and hasattr(self.gait, "set_identify_config"):
                    # ★ ここで実際に歩容コントローラに設定を渡す
                    self.gait.set_identify_config(gallery_json_path, target_pid)
                    log.info("[ID] gallery=%s, target_pid=%s (mode=歩容)",
                            gallery_json_path, target_pid or "(all)")
                else:
                    log.info("[ID] gallery=%s, target_pid=%s (mode=歩容; controller lacks set_identify_config)",
                            gallery_json_path, target_pid or "(all)")
                return
            
            if mode == "マルチモーダル":
                # target_pid には identify.py（autoモード）が返す「名前+連番」（例: Mizuta-P0001）が入る
                if getattr(self, "auto", None):
                    try:
                        self.auto.set_target_label(target_pid)
                        log.info("[ID][AUTO] target_label=%s", target_pid)
                    except Exception as e:
                        log.exception("[ID][AUTO] set_target_label failed: %s", e)
                return

        except Exception as e:
            log.error("[ID] set_identify_config failed (mode=%s): %s", mode, e, exc_info=True)
                
    def on_file_selected(self, path: str):
        log.info("[FILE] 選択: %s", path)


    def on_app_closing(self):
        # 後片付け（GPU/カメラ/DBなど。今は存在チェックのみ）
        try:
            if self.pipeline and hasattr(self.pipeline, "close"):
                self.pipeline.close()
        except Exception:
            pass
        for ctrl in [self.ada, self.osnet, self.gait]:
            try:
                if ctrl and hasattr(ctrl, "close"):
                    ctrl.close()
            except Exception:
                pass
        log.info("[APP] closing cleanup done.")

    def on_frame_clicked(self, x: int, y: int):
        self._click_xy = (x, y)
        log.info("[CLICK] (%d, %d)", x, y)

        mode = (self.mode_text or "").strip()

        if mode == "顔" and getattr(self, "ada", None) and hasattr(self.ada, "on_frame_clicked"):
            self.ada.on_frame_clicked(x, y)
        elif mode == "外見" and getattr(self, "osnet", None) and hasattr(self.osnet, "on_frame_clicked"):
            self.osnet.on_frame_clicked(x, y)
        elif mode == "歩容" and getattr(self, "gait", None) and hasattr(self.gait, "on_frame_clicked"):
            self.gait.on_frame_clicked(x, y)

    # ----------------------- core: per-frame -----------------------

    def on_frame(self, frame_bgr: np.ndarray):
        # ラベル重なり管理のリセット
        self._label_occupied = []

        # --- 前処理は1回だけ ---
        # if self._preproc and getattr(self._preproc, "enabled", False):
        #     try:
        #         frame_bgr = self._preproc.apply(frame_bgr)
        #     except Exception as e:
        #         log.warning("[PREPROC] apply failed: %s", e)

        # キャッシュ
        self._last_frame = frame_bgr.copy()

        # === 各モードの専用コントローラ優先 ===
        mode = (self.mode_text or "").strip()

        # 顔：AdaFaceController が居れば完全委譲（自身は何もしない）
        if mode == "顔" and self.ada is not None:
            # 顔モード用コントローラにフレームを渡す
            if hasattr(self.ada, "on_frame"):
                self.ada.on_frame(frame_bgr)
            return

        # 外見：OsnetController が居れば委譲
        if mode == "外見" and self.osnet is not None:
            if hasattr(self.osnet, "on_frame"):
                self.osnet.on_frame(frame_bgr)
            return

        # 歩容：GaitMixerController が居れば委譲
        if mode == "歩容" and self.gait is not None:
            if hasattr(self.gait, "on_frame"):
                self.gait.on_frame(frame_bgr)
            return
        
        # マルチモーダル：AutoController が居れば委譲
        if mode == "マルチモーダル" and self.auto is not None:
            if hasattr(self.auto, "on_frame"):
                self.auto.on_frame(frame_bgr)
            return

        # 以降はフォールバック（パイプライン or デモ）
        out = frame_bgr.copy()

        # (A) パイプラインがあれば使用
        used_pipeline = False
        if getattr(self, "pipeline", None) is not None:
            try:
                ctx = {
                    "detect": self.detect_on,
                    "feature": self.feature_on,
                    "identify": self.identify_on,
                    "mode": self.mode_text,
                    "click_xy": getattr(self, "_click_xy", None),
                }
                if hasattr(self.pipeline, "run"):
                    result = self.pipeline.run(out, context=ctx)
                elif hasattr(self.pipeline, "process"):
                    result = self.pipeline.process(out, context=ctx)
                else:
                    result = None

                if isinstance(result, dict) and "frame" in result and isinstance(result["frame"], np.ndarray):
                    out = result["frame"]
                    used_pipeline = True
            except Exception as e:
                log.error("[PIPELINE] 実行でエラー: %s", e, exc_info=True)

        # 画面更新は1回だけ
        try:
            self.ui.view.set_bgr(out)
        except Exception:
            pass

        # ダミーの特徴蓄積（デモ用）も1回だけ
        if self.feature_on:
            self._accumulate_dummy_feature(out)

    # ----------------------- helpers -----------------------

    def _short_mode_text(self) -> str:
        # 表示用の短縮
        mapping = {
            "顔": "face",
            "外見": "osnet",
            "歩容": "gait",
            "マルチモーダル": "auto",
        }
        return mapping.get(self.mode_text, self.mode_text)

    def _accumulate_dummy_feature(self, img: np.ndarray):
        """
        デモ用の簡易特徴（平均色など）をバッファに積む。
        実戦では self.ada / self.osnet / self.gait や pipeline 側で得たベクトルを格納。
        """
        try:
            mean = img.mean(axis=(0, 1))  # BGR の平均
            self._feature_buffer["mean_bgr"] = mean
        except Exception:
            pass
        
    # ========== UI活性制御 ==========
    def _is_checked(self, attr: str) -> bool:
        btn = getattr(self.ui, attr, None)
        try:
            return bool(btn.isChecked())
        except Exception:
            return False

    def _set_enabled(self, attr: str, enabled: bool):
        btn = getattr(self.ui, attr, None)
        try:
            if btn is not None and hasattr(btn, "setEnabled"):
                btn.setEnabled(bool(enabled))
        except Exception:
            pass

    def _safe_uncheck(self, attr: str):
        btn = getattr(self.ui, attr, None)
        try:
            if hasattr(btn, "setChecked"):
                btn.setChecked(False)
        except Exception:
            pass

    def _force_ui_initial_state(self):
        """起動直後の既定（学習/識別/保存は非活性）"""
        self._set_enabled("btn_feat", False)
        self._set_enabled("btn_id", False)
        self._set_enabled("btn_save_feat", False)

    def _force_face_ui_off_and_disable(self):
        """顔モード以外に切り替わったら顔関連UIをOFFかつ非活性に倒す"""
        for attr in ("btn_feat", "btn_id"):
            self._safe_uncheck(attr)
        # 保存ボタンはcheckableでない想定。無害なので同様に扱う
        self._set_enabled("btn_feat", False)
        self._set_enabled("btn_id", False)
        self._set_enabled("btn_save_feat", False)

    def _sync_enables(self):
        """
        全モード共通のボタン活性ルール：
        - 特徴学習 … 検出=ON かつ 識別=OFF
        - 識別     … 検出=ON かつ 学習=OFF
        - 保存     … （顔モードのみ）検出=ON かつ 学習=ON
        """
        det_on = self._is_checked("btn_det")
        feat_on = self._is_checked("btn_feat")
        id_on  = self._is_checked("btn_id")

        enable_feat = False
        enable_id   = False
        enable_save = False

        if det_on:
            enable_feat = not id_on
            enable_id   = not feat_on
            # 顔とマルチモーダルは保存可能
            if (self.mode_text or "") in ("顔", "マルチモーダル"):
                enable_save = feat_on

        self._set_enabled("btn_feat", enable_feat)
        self._set_enabled("btn_id",   enable_id)
        self._set_enabled("btn_save_feat", enable_save)

