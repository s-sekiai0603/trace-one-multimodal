# -*- coding: utf-8 -*-
"""
AutoController — マルチモーダルモード統合コントローラ
- モード「マルチモーダル」時のみ有効
- 検出ON: YOLO-Face + YOLO-Pose の両方を実行
- クリック: PoseのBBOX内に含まれるFaceのBBOXをひも付け、両方を赤表示
- 特徴学習ON: 1名を選択したら、face/osnet/gaitの各埋め込みを
    * 顔: 最初の AUTO_FACE_FRAMES 枚（デフォルト3枚）
    * 外見・歩容: 最初の AUTO_APP_FRAMES / AUTO_GAIT_FRAMES 枚（デフォルト32枚）
  だけバッファに積んで完了（以降は積まない）
- 識別ON: 画像内の全BBOXから特徴を抽出し、
    * 顔/外見は1フレーム目から類似度算出
    * 歩容は AUTO_GAIT_FRAMES に達してから類似度算出
  → UIには描画せず、ログに3つのスコアを書き出す
- 保存: gallery/auto/{face_gallery.json, app_gallery.json, gait_gallery.json} に
  「名前+連番」をキーにして保存（Append）
- 読込（識別設定）: ギャラリーの選択は行わず、「名前+連番」だけ指定して
  そのIDに紐づく3種類のベクトルを内部ロード
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
from datetime import datetime
import os, json, logging, re, math
import numpy as np
from collections import deque
import cv2

from PySide6.QtCore import QObject, Slot

# 遅延import（既に上でimportしていればスキップ）
try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime が見つかりません：pip install onnxruntime-gpu / -cpu") from e

# ==== 設定 ====
try:
    from ..config import (
        # デバイス/色
        TORCH_DEVICE, ORT_PROVIDERS, BBOX_COLOR, SELECTED_BBOX_COLOR, TOP_MATCH_BBOX_COLOR, FPS,
        # 検出モデル
        YOLO_FACE_MODEL_PATH, YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_FACE_IMGSZ, YOLO_FACE_MAX_DET,
        YOLO_POSE_MODEL_PATH, YOLO_POSE_IMGSZ, YOLO_POSE_CONF, YOLO_POSE_IOU, YOLO_POSE_MAX_DET, ORT_PROVIDERS,
        # 各埋め込み器
        ADAFACE_CKPT_PATH, ADAFACE_ARCH, ADAFACE_MARGIN, ADAFACE_OUT_SIZE, ADAFACE_LOG_EVERY,
        # Auto用パラメータ
        AUTO_MODE_NAME, AUTO_FACE_FRAMES, AUTO_APP_FRAMES, AUTO_GAIT_FRAMES,
        AUTO_GALLERY_DIR,
        # 骨格
        POSE_DRAW_LANDMARK_BGR, POSE_DRAW_CONNECTION_BGR, POSE_DRAW_THICKNESS, POSE_DRAW_RADIUS, POSE_KPT_CONF_MIN,
        # OSNet
        OSNET_MODEL_PATH, OSNET_INPUT_SIZE_HW,
        # Gait
        AUTO_GAIT_YAW_ENABLED,
        # ギャラリー
        AUTO_GALLERY_DIR, AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON,
        AUTO_SERIAL_PREFIX, AUTO_SERIAL_WIDTH,
        # ランキングデフォルト閾値
        AUTO_FACE_THRESH_DEFAULT, AUTO_APP_THRESH_DEFAULT, AUTO_GAIT_THRESH_DEFAULT,
        # トラッカー
        BYTE_TRACK_ENABLED,
        BYTE_TRACK_TRACK_THRESH, BYTE_TRACK_HIGH_THRESH,
        BYTE_TRACK_MATCH_THRESH, BYTE_TRACK_MAX_AGE,
        # ランキング
        AUTO_TRACK_RANK_INTERVAL_FRAMES, AUTO_TRACK_RANK_TOPN,
        AUTO_RANK_MODE,AUTO_REPORT_DIR, AUTO_REPORT_SHOW_SIM,       
    )
except Exception:
    # フォールバック（最低限）
    TORCH_DEVICE = "cpu"
    BBOX_COLOR = (0, 255, 255)
    SELECTED_BBOX_COLOR = (0, 180, 255)
    YOLO_FACE_MODEL_PATH = "models/yolov8s-face-lindevs.pt"
    YOLO_FACE_CONF = 0.35; YOLO_FACE_IOU = 0.50; YOLO_FACE_IMGSZ = 640; YOLO_FACE_MAX_DET = 50
    YOLO_POSE_MODEL_PATH = "models/yolov8l-pose.onnx"
    YOLO_POSE_IMGSZ = 640; YOLO_POSE_CONF = 0.25; YOLO_POSE_IOU = 0.50; YOLO_POSE_MAX_DET = 100
    ORT_PROVIDERS = ["CPUExecutionProvider"]
    ADAFACE_CKPT_PATH = "models/adaface_ir101_webface12m.ckpt"
    ADAFACE_ARCH = "ir_101"; ADAFACE_MARGIN = 0.2; ADAFACE_OUT_SIZE = 512; ADAFACE_LOG_EVERY = 0
    AUTO_MODE_NAME = "マルチモーダル"
    AUTO_FACE_FRAMES = 3
    AUTO_APP_FRAMES = 32
    AUTO_GAIT_FRAMES = 32
    AUTO_GALLERY_DIR = "data/gallery/auto"

# ==== 依存 ====
from ..detectors.yolo_face import YOLOFaceDetector
from ..detectors.yolo_pose import YOLOPoseDetector
from ..detectors.tracker import ByteTracker, Track
from ..detectors.media_pipe import MediaPipePose
from ..embedders.ada_face import AdaFaceEmbedder
from ..embedders.osnet import Osnet
from ..embedders.gait_mixer import GaitMixerRunner
from ..utils.auto_ranker import AutoRanker, RankEntry
from ..utils.auto_report_html import AutoHtmlReportBuilder, ReportSnapshot, ReportItem
from ..utils.evidence_capture import EvidenceCapture

# OSNet は Identifier風の実装になっているため薄いラッパが必要な場合があります
from ..embedders.osnet import cosine_sim as cos_sim

class AutoController(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self.log = logging.getLogger("app.auto_ctrl")
        self.win = window

        # 状態
        self.current_mode: str = AUTO_MODE_NAME
        self.det_on = False
        self.feat_on = False
        self.id_on = False

        self._last_frame: Optional[np.ndarray] = None
        self._label_occupied: List[Tuple[int,int,int,int]] = []
        
        self._boxes_face: Optional[np.ndarray] = None
        self._scores_face: Optional[np.ndarray] = None
        self._boxes_pose: Optional[np.ndarray] = None
        self._scores_pose: Optional[np.ndarray] = None
        self._kpts_pose: Optional[np.ndarray] = None

        self.sel_face_idx: Optional[int] = None
        self.sel_pose_idx: Optional[int] = None
        self._sel_pose_prev: Optional[int] = None
        
        # マルチモーダルレポート用状態
        self._auto_ranker = AutoRanker(mode=AUTO_RANK_MODE)
        self._auto_report_builder: Optional[AutoHtmlReportBuilder] = None
        self._auto_report_snapshots: List[ReportSnapshot] = []
        self._auto_report_frame_counter: int = 0
        self._auto_report_session_dir: Optional[str] = None
        
        # ランキング用しきい値
        self._face_thresh: float = float(AUTO_FACE_THRESH_DEFAULT)
        self._app_thresh:  float = float(AUTO_APP_THRESH_DEFAULT)
        self._gait_thresh: float = float(AUTO_GAIT_THRESH_DEFAULT)

        # 検出器
        self.face_det: Optional[YOLOFaceDetector] = None
        self.pose_det: Optional[YOLOPoseDetector] = None
        self.mp_pose: Optional[MediaPipePose] = None

        # 埋め込み器
        self.face_embed: Optional[AdaFaceEmbedder] = None
        self.app_embed: Optional[Osnet] = None
        self.gait_embed: Optional[GaitMixerRunner] = None

        # 検出結果バッファ
        self._boxes_face: Optional[np.ndarray] = None
        self._scores_face: Optional[np.ndarray] = None
        self._boxes_pose: Optional[np.ndarray] = None
        self._scores_pose: Optional[np.ndarray] = None
        self._kpts_pose: Optional[np.ndarray] = None  # (N,17,3)
        
        self._buf_gait_front: List[np.ndarray] = []
        self._buf_gait_back:  List[np.ndarray] = []
        self._buf_gait_side:  List[np.ndarray] = []
        
        # MediaPipe で推定した向き情報
        self._yaw_pose: Optional[np.ndarray] = None         # (N,) yaw[deg], 取れないところは NaN
        self._view_pose: Optional[List[Optional[str]]] = None  # ["front","side","back"] or None

        # 選択中
        self.sel_pose_idx: Optional[int] = None
        self.sel_face_idx: Optional[int] = None
        self._sel_pose_prev: Optional[np.ndarray] = None

        # 特徴バッファ（学習用：保存時に使う）
        self._buf_face: List[np.ndarray] = []
        self._buf_app: List[np.ndarray] = []
        self._buf_gait: List[np.ndarray] = []  # (1フレーム=17*2D) ではなく、モデルに合う形/個数で確保
        
        # 顔用：姿勢別（front / look-down / right / left / right-look-down / left-look-down）
        # 1 ID あたり最大 6 個の特徴ベクトルを保持
        self._buf_face_by_pose: Dict[str, np.ndarray] = {}
        # いま連続フレームを数えている姿勢ラベル（None ならリセット状態）
        self._face_pose_run_type: Optional[str] = None
        # 上記 run_type で連続して確保中の AdaFace ベクトル
        self._face_pose_run_vecs: List[np.ndarray] = []
        
        self._hi_face_idx: Optional[int] = None
        self._hi_app_idx:  Optional[int] = None
        self._hi_gait_idx: Optional[int] = None
        
        # 類似度（描画用）
        self._sim_face: Optional[np.ndarray] = None
        self._sim_app:  Optional[np.ndarray] = None
        self._sim_gait: Optional[np.ndarray] = None

        # 識別対象（名前+連番 → 3種ベクトル）
        self._id_label: Optional[str] = None
        self._gallery_face: Optional[np.ndarray] = None  # (D,)
        self._gallery_app: Optional[np.ndarray] = None   # (D,)
        self._gallery_gait: Optional[np.ndarray] = None  # (D,)
        
        self._gallery_face_by_pose: Dict[str, np.ndarray] = {}
        
        self._app_ort_sess = None          # onnxruntime.InferenceSession
        self._app_ort_in_name = None       # 入力名
        self._app_reid_dim = None          # 出力次元（初回埋め込みで確定）
        
        self._gait_api = None                  # seq -> vec を行う呼び出し関数をキャッシュ
        self._gait_api_missing_logged = False  # 未対応エラーを一度だけ出すためのフラグ
        
        self._target_label_base = None   # 例: "Yamada"
        self._target_label      = None   # 例: "Yamada-P0001"（完全形） 
        
        self._face_frames = 0
        self._app_frames  = 0
        self._gait_frames = 0
        
        self._last_face_boxes: Optional[np.ndarray] = None     # 顔BBOX（on_frame内で同期）
        self._last_person_boxes: Optional[np.ndarray] = None   # 人物BBOX（同上）
        self._last_kpts: Optional[np.ndarray] = None           # 骨格（同上）
        
        self._gseq_buffer = deque(maxlen=32)    # 上限は config(AUTO_GAIT_N_FRAMES) に合わせてもOK
        
        self._gait_tracks: dict[int, deque] = {}
        
        self._tracker: Optional[ByteTracker] = None
        self._track_ids_pose: Optional[np.ndarray] = None  # Pose BBOXごとの track_id
        self._track_ids_face: Optional[np.ndarray] = None  # Face BBOX ごとの track_id
        
        # アクティブな TrackID セット（tracker.update() 時に更新）
        self._active_track_ids: set[int] = set()

        # TrackIDごとの類似度統計（sum / count / avg / last）
        self._track_stats_face: dict[int, dict] = {}
        self._track_stats_app: dict[int, dict] = {}
        self._track_stats_gait: dict[int, dict] = {}

        # フレームカウンタ（ランキング出力用）
        self._frame_counter: int = 0
        self._interval_cleared_frame = -1

        # ラベル採番用の保持
        self._target_label_base = None   # 例: "Yamada"
        self._target_label      = None   # 例: "Yamada-P0001"（完全形）
        self._label_prefix      = None   # 例: "P"（configのAUTO_SEQ_PREFIX）
        self._label_seq         = None   # 例: 1（int。AUTO_SEQ_WIDTHでゼロ詰め）
        
        self._fps = FPS
        
        # ★ 2秒ごとのキャプチャ用
        #   FPS=30 の場合 → 2秒 = 60フレームごとに保存
        interval_frames = int(self._fps * 2)
        self._evidence_capture = EvidenceCapture(
            start_after=interval_frames,      # 最初の保存は2秒経過時
            interval=interval_frames,         # 以後2秒ごと
            save_root=os.path.join("data", "evidence", "capture"),
            file_prefix="capture_",
            ext=".png",
        )
        self._capture_frame_idx = 0  # 識別ON中のフレームカウンタ

        # シグナル購読
        self.win.frameAvailable.connect(self.on_frame)
        self.win.frameClicked.connect(self.on_frame_clicked)
        if hasattr(self.win, "detectionToggled"):
            self.win.detectionToggled.connect(self.on_detection_toggled)
        if hasattr(self.win, "featureLearningToggled"):
            self.win.featureLearningToggled.connect(self.on_feature_toggled)
        if hasattr(self.win, "identificationToggled"):
            self.win.identificationToggled.connect(self.on_identification_toggled)
        if hasattr(self.win, "featureSaveRequested"):
            self.win.featureSaveRequested.connect(self.on_feature_save_requested)
        if hasattr(self.win, "modeChanged"):
            self.win.modeChanged.connect(self.set_mode)

        self.log.info("[AUTO] controller ready.")
        
    def set_target_label(self, label: str):
        """
        入力:
        - 'Mizuta-P0003' のような「名前+連番」 → そのまま採用
        - 'Mizuta' のような「名前のみ」       → gallery/auto の3ファイルを横断して
                                                既存の最大連番+1 を自動採番（-P0001 形式）
        処理:
        - self._target_label / _label_base / _label_prefix / _label_seq を更新
        - face/app/gait の各ギャラリーから該当ラベルのベクトルを読み込みキャッシュ
        """
        try:
            s = (label or "").strip()
            if not s:
                raise ValueError("empty label")

            import re, json, os
            # 1) ラベル解析： base + [ - <prefix?><digits> ]
            #    例:
            #      'Mizuta-P0003' → base='Mizuta', prefix='P', num='0003'
            #      'Mizuta-0007'  → base='Mizuta', prefix='',  num='0007'
            #      'Mizuta'       → base='Mizuta'（自動採番へ）
            pat = r"^(?P<base>.+?)(?:-(?P<prefix>[A-Za-z]?)(?P<num>\d{2,}))?$"
            m = re.match(pat, s)
            if not m:
                s2 = s.replace("ー", "-").replace("−", "-").replace("―", "-")
                m = re.match(pat, s2)
                if not m:
                    raise ValueError(f"invalid label format: {label!r}")

            base   = m.group("base").strip()
            # まだ _label_prefix が存在しない可能性があるので getattr で安全に
            prefix = m.group("prefix") or getattr(self, "_label_prefix", None) or AUTO_SERIAL_PREFIX
            numstr = m.group("num")    # None → 自動採番

            # 2) 自動採番（num が無いときだけ）
            if not numstr:
                want_re = re.compile(rf"^{re.escape(base)}-{re.escape(prefix)}(\d+)$")

                def _scan_labels_from_json(path: str) -> list[int]:
                    seqs: list[int] = []
                    if not (path and os.path.isfile(path)):
                        return seqs
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:
                        return seqs

                    def _try_push(sval: str):
                        sval = (sval or "").strip()
                        m2 = want_re.match(sval)
                        if m2:
                            try:
                                seqs.append(int(m2.group(1)))
                            except Exception:
                                pass

                    if isinstance(data, dict):
                        # 代表的なマップ系
                        for k in ("label_to_vecs", "pid_to_vecs", "name_to_vecs",
                                "label_to_centroid", "pid_to_centroid", "name_to_centroid"):
                            if k in data and isinstance(data[k], dict):
                                for key in data[k].keys():
                                    _try_push(str(key))
                        # records 配列を内包するタイプ
                        if "records" in data and isinstance(data["records"], list):
                            for rec in data["records"]:
                                if isinstance(rec, dict):
                                    for key in ("pid", "label", "name", "id"):
                                        if key in rec:
                                            _try_push(str(rec[key]))
                                            break
                        # それ以外は top-level のキー群を一応スキャン
                        for key in list(data.keys()):
                            _try_push(str(key))
                    elif isinstance(data, list):
                        for rec in data:
                            if isinstance(rec, dict):
                                for key in ("pid", "label", "name", "id"):
                                    if key in rec:
                                        _try_push(str(rec[key]))
                                        break
                            elif isinstance(rec, str):
                                _try_push(rec)
                    return seqs

                seqs: list[int] = []
                for p in (AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON):
                    seqs.extend(_scan_labels_from_json(p))

                next_seq = (max(seqs) + 1) if seqs else 1
                width = int(AUTO_SERIAL_WIDTH) if AUTO_SERIAL_WIDTH else 4
                numstr = str(next_seq).zfill(width)

            # 3) フルラベル確定
            full = f"{base}-{prefix}{numstr}" if prefix else f"{base}-{numstr}"

            # 4) メンバー更新
            self._label_base   = base
            self._label_prefix = prefix
            self._label_seq    = int(numstr)
            self._target_label = full

            # 5) ギャラリー読み込み（face/app/gait）
            try:
                self._gallery_face = self._load_label_vecs_from_json(AUTO_FACE_JSON, full)
            except Exception:
                self._gallery_face = None
            try:
                self._gallery_app  = self._load_label_vecs_from_json(AUTO_APP_JSON,  full)
            except Exception:
                self._gallery_app  = None
            try:
                self._gallery_gait = self._load_label_vecs_from_json(AUTO_GAIT_JSON, full)
            except Exception:
                self._gallery_gait = None
                
            self._gallery_face_by_pose = self._load_face_gallery_by_pose(full)

            # 6) ログ
            ok_face = "OK" if (self._gallery_face is not None) else "None"
            ok_app  = "OK" if (self._gallery_app  is not None) else "None"
            ok_gait = "OK" if (self._gallery_gait is not None) else "None"
            self.log.info("[AUTO] target_label=%s  (base=%s, prefix=%s, seq=%s)", full, base, prefix, numstr)
            self.log.info("[AUTO][GALLERY] face=%s, appearance=%s, gait=%s", ok_face, ok_app, ok_gait)

        except Exception as e:
            self.log.exception("[AUTO] set_target_label failed: %s", e)
            
    def _app_get_state(self):
        # embedders/osnet.py が要求するキーを返す
        return {
            "session": self._ensure_app_session(),
            "inp_name": self._app_ort_in_name,
            "input_size": tuple(OSNET_INPUT_SIZE_HW),  # (H, W)
            "output_dim": self._app_reid_dim,
        }

    def _app_set_output_dim(self, d: int):
        # 初回embedで決まる出力次元を受け取って保持
        try:
            self._app_reid_dim = int(d)
        except Exception:
            self._app_reid_dim = None

    def _ensure_app_session(self):
        # 既に作っていればそれを返す
        if self._app_ort_sess is not None:
            return self._app_ort_sess

        model_path = str(OSNET_MODEL_PATH or "models/osnet_x1_0.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[OSNET] model not found: {model_path}")

        so = ort.SessionOptions()
        # 最適化/安定化
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False

        try:
            self._app_ort_sess = ort.InferenceSession(model_path, sess_options=so, providers=ORT_PROVIDERS)
        except Exception as e:
            self.log.warning("[AUTO][OSNET] providers=%s で初期化失敗 → CPUへフォールバック: %s", ORT_PROVIDERS, e)
            self._app_ort_sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        # 入力名
        self._app_ort_in_name = self._app_ort_sess.get_inputs()[0].name

        # 軽いウォームアップ（初回のカクつき回避）
        H, W = OSNET_INPUT_SIZE_HW
        dummy = np.zeros((1, 3, H, W), np.float32)
        for _ in range(3):
            _ = self._app_ort_sess.run(None, {self._app_ort_in_name: dummy})

        used = self._app_ort_sess.get_providers()
        self.log.info("[AUTO][OSNET] session providers=%s (requested=%s)", used, ORT_PROVIDERS)
        return self._app_ort_sess
    
    def _ensure_mediapipe(self) -> None:
        """MediaPipe PoseLandmarker の遅延初期化"""
        if self.mp_pose is not None:
            return
        try:
            self.mp_pose = MediaPipePose()
            self.log.info("[AUTO][MP] MediaPipePose initialized.")
        except Exception as e:
            self.log.warning("[AUTO][MP] MediaPipePose init failed: %s", e)
            self.mp_pose = None


    # ---------------- 外部API ----------------
    def set_mode(self, mode_text: str):
        self.current_mode = (mode_text or "").strip()
        if self.current_mode != AUTO_MODE_NAME:
            # 軽いリセット（描画や学習バッファ・選択解除）
            self._reset_runtime(light=True)

    @Slot(bool)
    def on_detection_toggled(self, on: bool):
        if self.current_mode != AUTO_MODE_NAME:
            return
        self.det_on = bool(on)
        if self.det_on:
            # 遅延初期化
            if self.face_det is None:
                try:
                    self.face_det = YOLOFaceDetector(
                        model_path=YOLO_FACE_MODEL_PATH,
                        device=TORCH_DEVICE,
                        conf=YOLO_FACE_CONF, iou=YOLO_FACE_IOU,
                        imgsz=YOLO_FACE_IMGSZ, max_det=YOLO_FACE_MAX_DET,
                    )
                    self.log.info("[AUTO] YOLO-Face ready.")
                except Exception as e:
                    self.log.exception("[AUTO] face detector init failed: %s", e)
                    self.face_det = None
            if self.pose_det is None:
                try:
                    self.pose_det = YOLOPoseDetector(
                        model_path=YOLO_POSE_MODEL_PATH,
                        input_size=YOLO_POSE_IMGSZ,
                        providers=ORT_PROVIDERS,
                        conf_th=YOLO_POSE_CONF,
                        nms_iou_th=YOLO_POSE_IOU,
                        max_dets=YOLO_POSE_MAX_DET
                    )
                    self.log.info("[AUTO] YOLO-Pose ready.")
                except Exception as e:
                    self.log.exception("[AUTO] pose detector init failed: %s", e)
                    self.pose_det = None
            # 停止中も即時反映
            self._reprocess_last_frame()
        else:
            self._boxes_face = self._scores_face = None
            self._boxes_pose = self._scores_pose = self._kpts_pose = None
            self.sel_face_idx = self.sel_pose_idx = None
            self._buf_face.clear(); self._buf_app.clear(); self._buf_gait.clear()
            self._buf_gait_front.clear(); self._buf_gait_back.clear(); self._buf_gait_side.clear()
            self._buf_face_by_pose.clear()
            self._face_pose_run_type = None
            self._face_pose_run_vecs = []
            self._buf_face.clear(); self._buf_app.clear(); self._buf_gait.clear()
            self._redraw_plain()

    @Slot(bool)
    def on_feature_toggled(self, on: bool):
        if self.current_mode != AUTO_MODE_NAME:
            return
        if not self.det_on:
            self.feat_on = False
            return

        self.feat_on = bool(on)

        if self.feat_on:
            # 排他
            self.id_on = False
            # 埋め込み器遅延初期化
            self._ensure_embedders()
            # クリック対象を積み直すので既存バッファはクリア
            self._buf_face.clear()
            self._buf_app.clear()
            self._buf_gait.clear()
            self.sel_face_idx = self.sel_pose_idx = None
            self._reprocess_last_frame()
        else:
            # ★ 特徴学習OFFになったときは選択状態をクリアして赤枠を消す
            self.sel_face_idx = self.sel_pose_idx = None
            try:
                self._reprocess_last_frame()
            except Exception:
                # 再描画に失敗しても致命的ではないので握りつぶし
                pass

    @Slot(bool)
    def on_identification_toggled(self, on: bool):
        if self.current_mode != AUTO_MODE_NAME:
            return
        if not self.det_on:
            self.id_on = False
            return

        self.id_on = bool(on)
        if self.id_on:
            if hasattr(self, "_evidence_capture") and self._evidence_capture is not None:
                try:
                    session_dir = self._evidence_capture.start_new_session()
                    self._capture_frame_idx = 0
                    self.log.info("[AUTO][CAPTURE] new session: %s", session_dir)
                except Exception as e:
                    self.log.exception("[AUTO][CAPTURE] start_new_session failed: %s", e)
                    
            # 排他
            self.feat_on = False
            self._ensure_embedders()
            try:
                if hasattr(self, "_gseq_buffer"):
                    self._gseq_buffer.clear()
            except Exception:
                pass
            try:
                self._buf_gait.clear()
            except Exception:
                pass
            self._sim_gait = None
            self._hi_gait_idx = None
            self.log.info("[AUTO] 識別: ON（UI描画なし・ログ出力のみ）")

            # ★ HTMLレポート用のセッションを初期化
            self._auto_report_frame_counter = 0
            self._auto_report_snapshots = []
            try:
                self._auto_report_session_dir = self._create_report_session_dir()
            except Exception:
                # 失敗しても動作は止めない
                self._auto_report_session_dir = AUTO_REPORT_DIR

            self._auto_report_builder = AutoHtmlReportBuilder(
                output_dir=self._auto_report_session_dir,
                show_sim=AUTO_REPORT_SHOW_SIM,
            )
            self._auto_report_builder.set_video_name(self._get_video_filename())

            self._reprocess_last_frame()

        else:
            # ★ OFFになった瞬間にHTMLレポート確定
            self.log.info("[AUTO][REPORT] 識別OFF → レポートを確定します")
            self._capture_frame_idx = 0
            try:
                self._finalize_auto_report()
            except Exception as e:
                self.log.exception("[AUTO][REPORT] finalize failed: %s", e)
                
    @Slot(object)
    def on_frame(self, frame_bgr: np.ndarray):
        # 直近フレームはコピーで保持（後段の再処理で破壊されないように）
        self._last_frame = frame_bgr.copy() if frame_bgr is not None else None
        if self.current_mode != AUTO_MODE_NAME or frame_bgr is None:
            return

        out = frame_bgr.copy()

        # --- 検出: Face / Pose 両方（det_on のときだけ更新） ---
        if self.det_on:
            # Face
            if self.face_det is not None:
                try:
                    boxes_f, scores_f = self.face_det.detect(frame_bgr)
                    self._boxes_face  = np.asarray(boxes_f,  np.float32) if boxes_f  is not None else None
                    self._scores_face = np.asarray(scores_f, np.float32) if scores_f is not None else None
                except Exception as e:
                    self._boxes_face = self._scores_face = None
                    self.log.exception("[AUTO] face detect failed: %s", e)

            # Pose
            if self.pose_det is not None:
                try:
                    boxes_p, scores_p, kpts = self.pose_det.detect(frame_bgr)
                    self._boxes_pose  = np.asarray(boxes_p,  np.float32) if boxes_p  is not None else None
                    self._scores_pose = np.asarray(scores_p, np.float32) if scores_p is not None else None
                    self._kpts_pose   = np.asarray(kpts,     np.float32) if kpts    is not None else None
                except Exception as e:
                    self._boxes_pose = self._scores_pose = self._kpts_pose = None
                    self.log.exception("[AUTO] pose detect failed: %s", e)

                #  歩容バッファに積む（32Fリングバッファ）
                if hasattr(self, "_gseq_buffer") and self._kpts_pose is not None:
                    try:
                        # 形状を壊さないよう shallow copy
                        self._gseq_buffer.append(self._kpts_pose.copy())
                    except Exception:
                        pass
                    
            # ByteTrack 風トラッカー更新（Face + Pose 両方）
            if BYTE_TRACK_ENABLED:
                self._update_tracker_and_track_ids()

            # クリック追従（IOU）と顔⇄人物リンク更新
            if self.sel_pose_idx is not None and self._boxes_pose is not None and len(self._boxes_pose) > 0:
                self._update_selection_pose_iou(iou_th=0.1)
                self._update_face_link_for_selected_pose()

            # 全Poseに対するFaceリンク（類似度描画用）
            try:
                self._update_face_links_all()
            except Exception:
                pass
        if AUTO_GAIT_YAW_ENABLED:
            if self._boxes_pose is not None and self._last_frame is not None:
                try:
                    if not hasattr(self, "_mp_pose") or self._mp_pose is None:
                        self._mp_pose = MediaPipePose()
                    self._pose_yaws = self._mp_pose.estimate_yaw_for_boxes(
                        self._last_frame, self._boxes_pose
                    )
                except Exception as e:
                    self.log.warning("[AUTO][YAW] failed: %s", e)
                    self._pose_yaws = None

        # --- 埋め込み器の準備（必要時のみ） ---
        if (self.feat_on or self.id_on):
            try:
                self._ensure_embedders()
            except Exception:
                pass

        # --- 識別：先に回して“ハイライト用インデックス”を確定 ---
        if self.id_on:
            try:
                self._do_identification(out)      # ← この中でログ出力も行う実装に統一
            except Exception as e:
                self.log.exception("[AUTO] identification failed: %s", e)

            # --- TrackIDごとの類似度統計を更新 ---
            try:
                self._update_similarity_track_stats()
            except Exception as e:
                self.log.exception("[AUTO] update_similarity_track_stats failed: %s", e)

            # --- Sフレームごとのランキングログ出力（任意） ---
            try:
                self._log_topN_track_stats_periodically()
            except Exception as e:
                self.log.exception("[AUTO] log_topN_track_stats failed: %s", e)

            # --- HTMLレポート用キャプチャ（一定フレームごと） ---
            self._auto_report_frame_counter += 1
            try:
                if (
                    AUTO_TRACK_RANK_INTERVAL_FRAMES > 0
                    and (self._auto_report_frame_counter % AUTO_TRACK_RANK_INTERVAL_FRAMES) == 0
                ):
                    self._capture_topN_snapshot()
            except Exception as e:
                self.log.exception("[AUTO] capture_topN_snapshot failed: %s", e)

        # --- 特徴学習（選択ありのときのみ） ---
        if self.feat_on and self._has_selection():
            try:
                self._do_feature_learning(out)
            except Exception as e:
                self.log.exception("[AUTO] feature_learning failed: %s", e)

        # --- 描画（識別で確定したハイライトを反映） ---
        self._draw_detections(out)
        
        # 2秒ごとのキャプチャ保存（識別ON中のみ）
        if hasattr(self, "_evidence_capture") and self._evidence_capture is not None:
            if self.id_on:
                # 識別ON中だけフレームカウンタを進める
                self._capture_frame_idx += 1
                try:
                    saved = self._evidence_capture.save_if_needed(
                        frame_idx=self._capture_frame_idx,
                        active=True,           # id_on中だけ True
                        bgr_frame=out,         # 描画済みの動画部分
                    )
                    if saved:
                        self.log.info("[AUTO][CAPTURE] saved: %s", saved)
                except Exception as e:
                    self.log.exception("[AUTO][CAPTURE] save_if_needed failed: %s", e)
            else:
                # 識別OFF中はカウンタリセット（ONになったらまた start_after から）
                self._capture_frame_idx = 0

        # 反映
        try:
            self.win.view.set_bgr(out)
        except Exception:
            pass

    @Slot(int, int)
    def on_frame_clicked(self, x: int, y: int):
        # マルチモーダルモード以外、検出OFF、または特徴学習OFFのときはクリック無効
        if self.current_mode != AUTO_MODE_NAME or not self.det_on or not self.feat_on:
            return
        if self._boxes_pose is None or len(self._boxes_pose) == 0:
            return

        # Pose側のインデックスを選択
        idx = self._find_box_at(x, y, self._boxes_pose)
        if idx is None:
            return

        self.sel_pose_idx = int(idx)
        self._sel_pose_prev = self._boxes_pose[self.sel_pose_idx].copy()

        # このPoseに内包されるFaceをひも付け
        self._update_face_link_for_selected_pose()

        # 学習ON中はバッファをリセット（先頭フレームから積む仕様のため）
        # （ここは self.feat_on が True でしか来ないので実質そのままでOK）
        self._buf_face.clear()
        self._buf_app.clear()
        self._buf_gait.clear()


    # UIの「特徴保存」ボタン
    def on_feature_save_requested(self):
        if self.current_mode != AUTO_MODE_NAME or not self._has_any_feat():
            self.log.info("[AUTO][SAVE] バッファが空です。")
            return
        os.makedirs(AUTO_GALLERY_DIR, exist_ok=True)
        # label 入力は AppController 側でダイアログ対応させる想定
        label = self._ensure_full_label()
        if not label:
            self.log.info("[AUTO][SAVE] 保存先ラベル（名前+連番）が未指定です。識別設定ダイアログ（マルチモーダル用）で指定してください。")
            return
        try:
            face_path = os.path.join(AUTO_GALLERY_DIR, "face_gallery.json")
            
            self.log.info(
                "[AUTO][SAVE] start: label=%r, face_by_pose_keys=%s, face_vecs=%d, path=%s",
                label,
                sorted(list(self._buf_face_by_pose.keys())) if hasattr(self, "_buf_face_by_pose") else "(none)",
                0 if self._buf_face is None else len(self._buf_face),
                face_path,
            )

            # ラベル空は保存しない（見落としチェック）
            if not label or not str(label).strip():
                self.log.warning("[AUTO][SAVE] label is empty -> skip saving.")
                return

            # 姿勢別バッファがあればそちらを優先して保存
            if self._buf_face_by_pose:
                # 例: {"front": vec_front, "right": vec_right, ...}
                for pose_type, v in self._buf_face_by_pose.items():
                    self._append_face_pose_json(face_path, label, pose_type, v)
            elif self._buf_face:
                # 姿勢別が無い場合は、従来通りの形式で保存（互換用）
                for v in self._buf_face:
                    self._append_json(face_path, label, v)
            else:
                # 顔特徴が 1 本も取れていない場合はスキップ相当
                self._append_json(face_path, label, None)

            # 外見・歩容は従来通り
            self._append_json(
                os.path.join(AUTO_GALLERY_DIR, "app_gallery.json"),
                label,
                self._mean(self._buf_app),
            )

            # 歩容: 向き別バッファがあれば view ごとに保存、無ければ従来通り
            gait_path = os.path.join(AUTO_GALLERY_DIR, "gait_gallery.json")

            if AUTO_GAIT_YAW_ENABLED:
                # 安全のため getattr で取得（まだ属性がなければ空リスト扱い）
                buf_front = getattr(self, "_buf_gait_front", []) or []
                buf_back  = getattr(self, "_buf_gait_back",  []) or []
                buf_side  = getattr(self, "_buf_gait_side",  []) or []

                if buf_front or buf_back or buf_side:
                    # front/back/side それぞれ平均をとって、存在するものだけ保存
                    def _mean_or_none(buf):
                        return self._mean(buf) if buf else None

                    front_vec = _mean_or_none(buf_front)
                    back_vec  = _mean_or_none(buf_back)
                    side_vec  = _mean_or_none(buf_side)

                    if front_vec is not None:
                        self._append_gait_view_json(gait_path, label, "front", front_vec)
                    if back_vec is not None:
                        self._append_gait_view_json(gait_path, label, "back", back_vec)
                    if side_vec is not None:
                        self._append_gait_view_json(gait_path, label, "side", side_vec)

                else:
                    # まだ向き別学習をしていない古いデータ用に、従来形式も維持
                    self._append_json(
                        gait_path,
                        label,
                        self._mean(self._buf_gait),
                    )

                self.log.info(
                    "[AUTO][SAVE] 保存完了: %s (face/app/gait, face_vecs=%d)",
                    label, len(self._buf_face),
                )
                
            else:
                # yaw OFF のときは常に従来形式
                self._append_json(gait_path, label, self._mean(self._buf_gait))
        except Exception as e:
            self.log.exception("[AUTO][SAVE] JSON 追記に失敗しました: %s", e)

    # 片付け
    def close(self):
        self._reset_runtime(light=False)

    # ---------------- 内部処理 ----------------
    def _ensure_embedders(self):
        # 顔
        if self.face_embed is None:
            try:
                self.face_embed = AdaFaceEmbedder(
                    ckpt_path=ADAFACE_CKPT_PATH, architecture=ADAFACE_ARCH,
                    device=TORCH_DEVICE, margin=ADAFACE_MARGIN,
                    out_size=ADAFACE_OUT_SIZE, log_each=ADAFACE_LOG_EVERY
                )
            except Exception as e:
                self.log.exception("[AUTO] AdaFace init failed: %s", e); self.face_embed = None
        # 外見（Osnet は既存の Identifier形状に依存。ここでは単純化、embed(person_crop) を持つ想定）
        if self.app_embed is None:
            try:
                self.app_embed = Osnet(self._app_get_state, self._app_set_output_dim)
                # ORTセッションはAuto側で構築して get_state() から渡す
                self._ensure_app_session()         # ← OK: Auto側のセッションを必ず作る
                self.log.info("[AUTO] OSNet ready.")
            except Exception as e:
                self.log.exception("[AUTO] OSNet init failed: %s", e)
                self.app_embed = None
                        
        # 歩容
        if self.gait_embed is None:
            try:
                # 1) 位置引数（パス, デバイス）で試す
                try:
                    self.gait_embed = GaitMixerRunner("models/GaitMixer.pt", TORCH_DEVICE)
                except TypeError:
                    # 2) 別名キーワード（weights_path/ckpt_path）で試す
                    try:
                        self.gait_embed = GaitMixerRunner(weights_path="models/GaitMixer.pt", device=TORCH_DEVICE)
                    except TypeError:
                        try:
                            self.gait_embed = GaitMixerRunner(ckpt_path="models/GaitMixer.pt", device=TORCH_DEVICE)
                        except TypeError:
                            # 3) 引数なし→あとで load() を探す
                            self.gait_embed = GaitMixerRunner()
                            if hasattr(self.gait_embed, "load"):
                                try:
                                    self.gait_embed.load("models/GaitMixer.pt", device=TORCH_DEVICE)
                                except TypeError:
                                    # load(path) だけのケース
                                    self.gait_embed.load("models/GaitMixer.pt")
                self.log.info("[AUTO] GaitMixer ready.")
            except Exception as e:
                self.log.exception("[AUTO] GaitMixer init failed: %s", e)
                self.gait_embed = None

    def _do_feature_learning(self, frame: np.ndarray):
        # 選択中のPose/Face indexで、先頭 N 枚だけ積む
        if self.sel_pose_idx is None: 
            return
        # 顔：YOLO-Pose の顔キーポイントから向きを判定し、
        #     向きごとに連続 AUTO_FACE_FRAMES 枚ぶんを 1 ベクトルに集約
        self._update_face_learning_with_pose(frame)
        
        # 外見
        if self.app_embed is not None and len(self._buf_app) < int(AUTO_APP_FRAMES):
            # セッション保険（まだならこのタイミングで必ず作る）
            try:
                if self._app_ort_sess is None:
                    self._ensure_app_session()
            except Exception:
                pass

            pbox = self._boxes_pose[self.sel_pose_idx]
            crop = self._crop(frame, pbox)
            if crop is not None:
                try:
                    v = self.app_embed.embed(crop)
                except RuntimeError:
                    # まれに初回だけ未初期化のエラーが走る実装に対処
                    try:
                        self.app_embed._ensure_session()
                        v = self.app_embed.embed(crop)
                    except Exception:
                        v = None
                if v is not None:
                    self._buf_app.append(np.asarray(v, np.float32).reshape(-1))
        # 歩容
        if self.gait_embed is not None:
            k = None
            # ★ sel_pose_idx と _kpts_pose の両方を安全にチェック
            if self._kpts_pose is not None and self.sel_pose_idx is not None:
                idx = int(self.sel_pose_idx)
                if 0 <= idx < len(self._kpts_pose):
                    k = self._kpts_pose[idx]
                else:
                    # BBOX 数が減った／検出ゼロになったなどでインデックスが無効になった場合はリセット
                    self.sel_pose_idx = None

            if k is not None:
                if not hasattr(self, "_gait_seq"):
                    from collections import deque
                    self._gait_seq = deque(maxlen=int(AUTO_GAIT_FRAMES))
                self._gait_seq.append(np.asarray(k, np.float32))  # (17,3)

                # 事前にAPI解決（未解決なら一度だけログ）
                if self._gait_api is None:
                    self._resolve_gait_api()

                # 32フレームそろっていて、まだ歩容バッファが空のときだけ 1 本学習
                if len(self._gait_seq) == int(AUTO_GAIT_FRAMES) and len(self._buf_gait) == 0:
                    seq = np.stack(list(self._gait_seq), axis=0)  # (T,17,3)
                    gv = self._gait_embed_from_seq(seq)          # (D,) or None
                    if gv is not None:
                        gv = np.asarray(gv, np.float32).reshape(-1)

                        # ① 従来どおり全体バッファにも入れる（既存保存ロジックとの互換性確保）
                        self._buf_gait.append(gv)
                        
                        if AUTO_GAIT_YAW_ENABLED:
                            # ② MediaPipe の yaw から front/back/side を決定し、向き別バッファにも格納
                            yaw = None
                            yaw_list = getattr(self, "_pose_yaws", None)
                            if yaw_list is not None and self.sel_pose_idx is not None:
                                idx = int(self.sel_pose_idx)
                                if 0 <= idx < len(yaw_list):
                                    yaw = yaw_list[idx]

                            # デフォルトは front 扱い
                            view = "front"
                            if yaw is not None:
                                a = abs(float(yaw))
                                # ざっくり:
                                #   0°±45° → front
                                #   180°±45° → back
                                #   それ以外 → side（90°近辺）
                                if a <= 45.0:
                                    view = "front"
                                elif a >= 135.0:
                                    view = "back"
                                else:
                                    view = "side"

                            if view == "front":
                                self._buf_gait_front.append(gv)
                            elif view == "back":
                                self._buf_gait_back.append(gv)
                            else:
                                self._buf_gait_side.append(gv)

                            self.log.info(
                                "[AUTO][GAIT] learned 1 seq (T=%d, view=%s, yaw=%s)",
                                len(self._gait_seq),
                                view,
                                f"{yaw:.1f}" if yaw is not None else "None"
                            )

                    # 1本学習したら、このセッション用のシーケンスはリセット
                    if hasattr(self, "_gait_seq"):
                        self._gait_seq.clear()

    def _do_identification_and_log(self, frame: np.ndarray):
        # ラベルが指定されていなければ読み込まない
        if self._gallery_face is None and self._gallery_app is None and self._gallery_gait is None:
            # 必要ならここで _target_label から読み直す
            if getattr(self, "_target_label", None):
                self._load_auto_label_vectors(self._target_label)
            if self._gallery_face is None and self._gallery_app is None and self._gallery_gait is None:
                return

        # 顔（そのフレームで全Face BBOX）
        face_sims: List[float] = []
        if self.face_embed is not None and self._gallery_face is not None and self._boxes_face is not None:
            try:
                feats = self.face_embed.embed_many(frame, self._boxes_face)  # (N,D)
                sims = feats @ self._gallery_face.astype(np.float32)
                face_sims = [float(s) for s in sims]
            except Exception:
                pass

        # 外見（全Pose BBOXで crop → embed）
        app_sims: List[float] = []
        if self.app_embed is not None and self._gallery_app is not None and self._boxes_pose is not None:
            # セッション保険
            try:
                if self._app_ort_sess is None:
                    self._ensure_app_session()
            except Exception:
                pass

            sims = []
            for b in self._boxes_pose:
                crop = self._crop(frame, b)
                if crop is None:
                    continue
                try:
                    v = self.app_embed.embed(crop)
                except RuntimeError:
                    try:
                        self.app_embed._ensure_session()
                        v = self.app_embed.embed(crop)
                    except Exception:
                        v = None
                if v is None:
                    continue
                v = np.asarray(v, np.float32).reshape(-1)
                v /= (np.linalg.norm(v) + 1e-12)
                sims.append(float(np.dot(v, self._gallery_app)))
            app_sims = sims

        # 歩容（32フレームない場合はスキップ）
        gait_sims: List[float] = []
        if self.gait_embed is not None and self._gallery_gait is not None:
            # クリック選択者のみ評価（全員分の時系列を管理していないため簡易）
            if len(self._buf_gait) >= int(AUTO_GAIT_FRAMES):
                gv = self._mean(self._buf_gait)  # ここでは平均で近似。実装に合わせて時系列→128Dを算出してください
                gait_sims = [float(np.dot(gv, self._gallery_gait))]

        # === ログ出力（UIへは描画しない） ===
        # 顔/外見: 1フレーム目から、歩容: 32フレーム揃ってから
        if face_sims:
            self.log.info("[AUTO][SIM][face] %s", ["%.3f" % s for s in face_sims])
        if app_sims:
            self.log.info("[AUTO][SIM][appearance] %s", ["%.3f" % s for s in app_sims])
        if gait_sims:
            self.log.info("[AUTO][SIM][gait] %s", ["%.3f" % s for s in gait_sims])
    
    def _do_identification(self, frame_bgr: np.ndarray):
        """
        AUTO識別（UIは描画せず、ハイライト用インデックスとログだけ）
        - 顔/外見: 現フレームの全BBOX → 埋め込み → ギャラリーのセントロイドと cos 類似
        - 歩容   : 直近 T=32 フレームのキーポイント系列から人ごとに埋め込み → 類似
        互換ポイント:
        * AdaFace: embed_many(frame, boxes) -> (N,D) を前提（既存コードと同じ）
        * OSNet  : embed(crop) を個別呼び出し（バッチ不要）
        * Gait   : _gseq_buffer の人数最小列で人インデックスを揃えて評価
        """
        # 前回ハイライトをクリア
        self._hi_face_idx = self._hi_app_idx = self._hi_gait_idx = None
        self._sim_face = self._sim_app = self._sim_gait = None

        # ターゲット未ロードならここで読む
        if (self._gallery_face is None and self._gallery_app is None and self._gallery_gait is None
            and getattr(self, "_target_label", None)):
            self._load_auto_label_vectors(self._target_label)

        # ギャラリーのセントロイドを作る（(N,D) でも (D,) でもOK）
        def _centroid(vecs: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if vecs is None:
                return None
            v = np.asarray(vecs, np.float32)
            if v.ndim == 2 and v.shape[0] > 1:
                v = v.mean(axis=0)
            if v.ndim != 1:
                return None
            v /= (np.linalg.norm(v) + 1e-12)
            return v.astype(np.float32)

        g_face = _centroid(self._gallery_face)
        g_app  = _centroid(self._gallery_app)
        g_gait = _centroid(self._gallery_gait)

        any_hit = False

        # ---- 顔（AdaFace; embed_many は (N,D) を返す前提）----
        try:
            if self.face_embed is not None and self._boxes_face is not None and len(self._boxes_face) > 0:
                feats = self.face_embed.embed_many(frame_bgr, self._boxes_face)  # (N,D)
                if feats is not None and len(feats) > 0:
                    feats = np.asarray(feats, np.float32)
                    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
                    feats = feats / norms

                    # pose別ギャラリー（records対応）と、非pose別のセントロイド（互換用）
                    pose_map = getattr(self, "_gallery_face_by_pose", {}) or {}

                    g_face_centroid = None
                    try:
                        g_face_centroid = self._gallery_face
                        if g_face_centroid is not None and g_face_centroid.ndim == 2 and g_face_centroid.shape[0] > 1:
                            g_face_centroid = g_face_centroid.mean(axis=0)
                        if g_face_centroid is not None:
                            g_face_centroid = (g_face_centroid / (np.linalg.norm(g_face_centroid) + 1e-12)).astype(np.float32)
                    except Exception:
                        g_face_centroid = None

                    sims = np.full((feats.shape[0],), np.nan, dtype=np.float32)
                    best_idx, best_val = None, None

                    for i in range(feats.shape[0]):
                        f = feats[i]
                        pose_i = self._pose_for_face_index(i)  # そのBBOXの向き
                        g_i = self._pick_gallery_vec_for_pose(pose_i, pose_map, g_face_centroid)
                        if g_i is None or g_i.ndim != 1 or g_i.size != f.size:
                            continue
                        s = float(np.dot(f, g_i))
                        sims[i] = s
                        if (best_val is None) or (s > best_val):
                            best_val = s
                            best_idx = i

                    self._sim_face = sims
                    self._hi_face_idx = None if best_idx is None else int(best_idx)
                    if best_val is not None:
                        self.log.info("[AUTO][SIM][face]=%.4f (idx=%d)", best_val, self._hi_face_idx)
                        any_hit = True
        except Exception:
            pass

        # ---- 外見（OSNet; 個別crop→embed）----
        try:
            if self.app_embed is not None and g_app is not None and self._boxes_pose is not None and len(self._boxes_pose) > 0:
                best, best_idx = None, None
                #  各Pose BBOXごとの類似度（描画用）
                app_sims = np.full(len(self._boxes_pose), np.nan, dtype=np.float32)

                for i, box in enumerate(self._boxes_pose):
                    crop = self._crop(frame_bgr, box)
                    if crop is None:
                        continue
                    try:
                        v = self.app_embed.embed(crop)
                    except RuntimeError:
                        # たまに初回だけ session未初期化の実装に耐性
                        try:
                            self.app_embed._ensure_session()
                            v = self.app_embed.embed(crop)
                        except Exception:
                            v = None
                    if v is None:
                        continue
                    v = np.asarray(v, np.float32).reshape(-1)
                    v /= (np.linalg.norm(v) + 1e-12)
                    s = float(np.dot(v, g_app))

                    app_sims[i] = s  #  Pose i の類似度を記録

                    if best is None or s > best:
                        best, best_idx = s, i

                #  全Pose BBOX分を保存
                self._sim_app = app_sims

                if best_idx is not None:
                    self._hi_app_idx = int(best_idx)
                    self.log.info("[AUTO][SIM][appearance]=%.4f (idx=%d)", best, best_idx)
                    any_hit = True
        except Exception:
            pass

        # ---- 歩容（GaitMixer; 直近T=32の共通列だけ評価）----
        try:
            if self.gait_embed is not None and hasattr(self, "_gseq_buffer"):
                # API解決（未解決のままならスキップ）
                if self._gait_api is None:
                    self._resolve_gait_api()

                if self._gait_api is not None:
                    # 現フレームの kpts_pose / track_ids_pose から
                    # track_id -> キーポイント列 を更新
                    self._update_gait_tracks()

                    track_ids_pose = getattr(self, "_track_ids_pose", None)
                    if track_ids_pose is None or len(track_ids_pose) == 0:
                        # Pose BBOX がなければ歩容は何もしない
                        pass
                    else:
                        n_pose = len(track_ids_pose)
                        gait_sims = np.full(n_pose, np.nan, dtype=np.float32)

                        best, best_j = None, None

                        # ギャラリー側：view 別（front/back/side）＋デフォルト（viewなし）
                        gallery_view_map = getattr(self, "_gallery_gait_by_view", {}) or {}
                        # 上で _centroid() した g_gait を「ビューなしのデフォルト」として使う
                        gallery_default  = g_gait

                        for j in range(n_pose):
                            try:
                                tid = int(track_ids_pose[j])
                            except Exception:
                                continue
                            if tid < 0:
                                # トラックが付いていない Pose はスキップ
                                continue

                            # この track_id の歩容シーケンスを取得
                            dq = getattr(self, "_gait_tracks", {}).get(tid)
                            if not dq or len(dq) < int(AUTO_GAIT_FRAMES):
                                # まだ十分フレームがたまっていない
                                continue

                            # deque[(17,3)] → (T,17,3)
                            try:
                                seq = np.stack(list(dq), axis=0).astype(np.float32)
                            except Exception:
                                continue

                            gv = self._gait_embed_from_seq(seq)
                            if gv is None:
                                continue
                            gv = np.asarray(gv, np.float32).reshape(-1)
                            gv /= (np.linalg.norm(gv) + 1e-12)

                            # --- この人物の「必要な向き」を決める（Poseインデックス j ベース） ---
                            need_view = None
                            if getattr(self, "_pose_yaws", None) is not None:
                                if 0 <= j < len(self._pose_yaws):
                                    yaw = self._pose_yaws[j]
                                    if yaw is not None:
                                        need_view = self._yaw_to_view(yaw)  # "front"/"back"/"side"

                            # --- 向きに応じて、ギャラリー中から優先度順に view を選ぶ ---
                            gvec = self._pick_gallery_vec_for_view(
                                need_view,
                                gallery_view_map,
                                gallery_default,
                            )
                            if gvec is None:
                                continue
                            gvec = np.asarray(gvec, np.float32).reshape(-1)
                            if gvec.shape != gv.shape:
                                continue

                            s = float(np.dot(gv, gvec))
                            gait_sims[j] = s

                            if best is None or s > best:
                                best, best_j = s, j

                        # 有効な値が1つでもあれば保存（全部 NaN なら捨てる）
                        if np.any(~np.isnan(gait_sims)):
                            # インデックスは「Pose BBOX のインデックス」に揃う
                            self._sim_gait = gait_sims

                        if best_j is not None:
                            self._hi_gait_idx = int(best_j)
                            self.log.info("[AUTO][SIM][gait]=%.4f (idx=%d)", best, best_j)
                            any_hit = True
        except Exception:
            pass

        if not any_hit:
            self.log.info("[AUTO][SIM] 類似度なし（ギャラリー未ロード or BBOX無し or 歩容はフレーム不足）")
            
    def _accumulate_track_stat(self, stats: dict[int, dict], tid: int, sim: float) -> None:
        """
        1つの TrackID について、類似度を1件追加する。
        stats[tid] = {"sum":..., "count":..., "avg":..., "last":...}
        """
        if np.isnan(sim):
            return
        if tid not in stats:
            stats[tid] = {"sum": 0.0, "count": 0, "avg": 0.0, "last": 0.0}
        rec = stats[tid]
        rec["sum"] += float(sim)
        rec["count"] += 1
        rec["avg"] = rec["sum"] / max(rec["count"], 1)
        rec["last"] = float(sim)
        
    def _reset_interval_stats(self) -> None:
        """60フレーム区間の平均だけを使うため、統計をクリアして次の区間を開始する。"""
        self._track_stats_face.clear()
        self._track_stats_app.clear()
        self._track_stats_gait.clear()

    def _update_similarity_track_stats(self) -> None:
        """
        現フレームの TrackID / 類似度から、
        TrackIDごとの統計（Face/App/Gait）を更新する。

        ★ ポイント：
        すべて「Pose側のTrackID」をキーとして集約する。
        顔SIMも Poseインデックス -> Pose TrackID に紐付ける。
        """
        # アクティブなTrackがなければ何もしない
        if not getattr(self, "_active_track_ids", None):
            return

        # 類似度配列
        sim_face = getattr(self, "_sim_face", None)
        sim_app  = getattr(self, "_sim_app",  None)
        sim_gait = getattr(self, "_sim_gait", None)

        # Pose側TrackIDが無いとどうしようもない
        if self._track_ids_pose is None or self._boxes_pose is None:
            return
        if len(self._track_ids_pose) == 0:
            return

        n_pose = len(self._track_ids_pose)

        for i in range(n_pose):
            tid = int(self._track_ids_pose[i])
            if tid < 0:
                continue
            if tid not in self._active_track_ids:
                continue

            # ---- Face: Poseインデックス i に対応する顔SIMを Pose TID で積む ----
            if sim_face is not None:
                f_sim = None
                try:
                    # _get_similarity_for_pose(i) は
                    #   (face_sim, app_sim, gait_sim) のタプルを返す想定
                    f_sim, _, _ = self._get_similarity_for_pose(i)
                except Exception:
                    f_sim = None

                if f_sim is not None:
                    try:
                        fs = float(f_sim)
                        if not np.isnan(fs):
                            self._accumulate_track_stat(self._track_stats_face, tid, fs)
                    except Exception:
                        pass

            # ---- Appearance: これまで通り Poseインデックス i ベース ----
            if sim_app is not None and i < len(sim_app):
                try:
                    s = float(sim_app[i])
                    if not np.isnan(s):
                        self._accumulate_track_stat(self._track_stats_app, tid, s)
                except Exception:
                    pass

            # ---- Gait: これまで通り Poseインデックス i ベース ----
            if sim_gait is not None and i < len(sim_gait):
                try:
                    s = float(sim_gait[i])
                    if not np.isnan(s):
                        self._accumulate_track_stat(self._track_stats_gait, tid, s)
                except Exception:
                    pass
                    
    def _log_topN_track_stats_periodically(self) -> None:
        """
        フレームカウンタを進め、Sフレームごとに
        Face/App/Gait の TrackID別 類似度平均の上位N件をログ出力する。
        """
        self._frame_counter += 1
        if AUTO_TRACK_RANK_INTERVAL_FRAMES <= 0:
            return
        if (self._frame_counter % AUTO_TRACK_RANK_INTERVAL_FRAMES) != 0:
            return

        topN = AUTO_TRACK_RANK_TOPN

        def _collect_top(stats: dict[int, dict], label: str):
            rows = []
            for tid, rec in stats.items():
                if tid not in self._active_track_ids:
                    # 途切れた Track はランキング対象外
                    continue
                c = rec.get("count", 0)
                if c <= 0:
                    continue
                avg = float(rec.get("avg", 0.0))
                last = float(rec.get("last", 0.0))
                rows.append((tid, avg, last, c))
            if not rows:
                return
            rows.sort(key=lambda r: r[1], reverse=True)  # avg で降順
            for tid, avg, last, c in rows[:topN]:
                self.log.info(
                    "[AUTO][RANK][%s] tid=%d avg=%.5f last=%.5f count=%d",
                    label, tid, avg, last, c
                )

        # Face / Appearance / Gait それぞれログ出力
        _collect_top(self._track_stats_face, "face")
        _collect_top(self._track_stats_app,  "appearance")
        _collect_top(self._track_stats_gait, "gait")
        
        if self._interval_cleared_frame != self._frame_counter:
            self._reset_interval_stats()
            self._interval_cleared_frame = self._frame_counter


    # ------ 補助 ------
    def _reprocess_last_frame(self):
        if self._last_frame is not None:
            self.on_frame(self._last_frame)

    def _redraw_plain(self):
        if self._last_frame is not None:
            try:
                self.win.view.set_bgr(self._last_frame)
            except Exception:
                pass

    def _draw_detections(self, img: np.ndarray):
        id_on = bool(getattr(self, "id_on", False))

        # ---- Face BBOX（最良一致だけ緑、そのほかは標準色）----
        if self._boxes_face is not None:
            for i, box in enumerate(self._boxes_face):
                color = TOP_MATCH_BBOX_COLOR if (id_on and getattr(self, "_hi_face_idx", None) == i) else BBOX_COLOR
                self._draw_box(img, box, color, thick=False)

        # ---- Pose BBOX + Skeleton ----
        if self._boxes_pose is not None:
            # BBOX（最良一致だけ緑）
            for i, box in enumerate(self._boxes_pose):
                color = TOP_MATCH_BBOX_COLOR if (id_on and getattr(self, "_hi_app_idx", None) == i) else BBOX_COLOR
                self._draw_box(img, box, color, thick=False)
                
                if id_on:
                    face_sim, app_sim, gait_sim = self._get_similarity_for_pose(i)
                    self._draw_similarity_text(img, box, face_sim, app_sim, gait_sim)
                    
                # ★ MediaPipe Pose の yaw から向きラベルを描画（属性名を _pose_yaws に統一）
                # yaw_deg_list = getattr(self, "_pose_yaws", None)
                # view_txt = None
                # if yaw_deg_list is not None and i < len(yaw_deg_list):
                #     yaw = yaw_deg_list[i]
                #     if yaw is not None:
                #         a = abs(float(yaw))
                #         # ざっくり：
                #         #   0°±45° → 正面
                #         #   90°±45° → 側面
                #         #   180°±45° → 背面
                #         if a <= 45:
                #             view_txt = "Front"    # 0°
                #         elif a >= 135:
                #             view_txt = "Back"    # 180°
                #         else:
                #             view_txt = "Side"    # 90° 付近

                # if view_txt:
                #     x1, y1, x2, y2 = map(int, box[:4])
                #     H, W = img.shape[:2]
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     base_thick = max(1, int(round(max(H, W) / 900)))
                #     font_scale = max(0.5, min(1.0, max(H, W) / 1400.0))
                #     # BBOXの少し上に描画
                #     px, py = x1, max(15, y1 - 8)
                #     cv2.putText(
                #         img, view_txt, (px, py), font,
                #         font_scale, (0, 255, 255),
                #         base_thick + 1, cv2.LINE_AA
                #     )

            # 骨格（最良一致の人物 idx だけ：ジョイント=緑、接続線=白）
            if self._kpts_pose is not None and len(self._kpts_pose) == len(self._boxes_pose):
                hi_idx = getattr(self, "_hi_gait_idx", None) if id_on else None
                self._draw_skeletons(img, self._kpts_pose, hi_index=hi_idx)

        # ---- 選択中は上塗りで赤太線（クリック追従の視覚化）----
        if (
            self.sel_pose_idx is not None
            and self._boxes_pose is not None
        ):
            idx = int(self.sel_pose_idx)
            if 0 <= idx < len(self._boxes_pose):
                self._draw_box(img, self._boxes_pose[idx], SELECTED_BBOX_COLOR, thick=True)
            else:
                # BBOX がなくなった場合は選択もリセット
                self.sel_pose_idx = None

        if (
            self.sel_face_idx is not None
            and self._boxes_face is not None
        ):
            idx_f = int(self.sel_face_idx)
            if 0 <= idx_f < len(self._boxes_face):
                self._draw_box(img, self._boxes_face[idx_f], SELECTED_BBOX_COLOR, thick=True)
            else:
                self.sel_face_idx = None
                
    def _get_similarity_for_pose(self, pose_idx: int):
        """Poseインデックスに対応する Face/App/Gait 類似度を返す（存在しなければ None）。"""
        f_sim = a_sim = g_sim = None

        # Face: Poseに含まれるFace BBOXがあれば、そのFaceの類似度を使用
        if self._sim_face is not None and self._pose_face_index is not None:
            if 0 <= pose_idx < len(self._pose_face_index):
                fi = self._pose_face_index[pose_idx]
                if fi is not None and 0 <= fi < len(self._sim_face):
                    try:
                        val = float(self._sim_face[fi])
                        if not np.isnan(val):
                            f_sim = val
                    except Exception:
                        pass

        # Appearance: Poseインデックスと1:1で対応
        if self._sim_app is not None and 0 <= pose_idx < len(self._sim_app):
            try:
                val = float(self._sim_app[pose_idx])
                if not np.isnan(val):
                    a_sim = val
            except Exception:
                pass

        # Gait: 先頭 min_n 人ぶんだけ self._sim_gait が入っている
        if self._sim_gait is not None and 0 <= pose_idx < len(self._sim_gait):
            try:
                val = float(self._sim_gait[pose_idx])
                if not np.isnan(val):
                    g_sim = val
            except Exception:
                pass

        return f_sim, a_sim, g_sim

    def _draw_similarity_text(self, img: np.ndarray, box: np.ndarray,
                              face_sim: Optional[float],
                              app_sim: Optional[float],
                              gait_sim: Optional[float]):
        # 3つとも None なら描かない
        if face_sim is None and app_sim is None and gait_sim is None:
            return

        H, W = img.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])

        # 文字サイズは画像サイズに応じて調整
        t = max(1, int(round(max(H, W) / 900)))
        font_scale = 0.4 * t
        thickness  = max(1, t)

        line_h = int(18 * t)
        # BBOXの少し上に3行ぶん確保。はみ出しそうならBBOX内上部に描画
        base_y = y1 - 5 - 2 * line_h
        if base_y < 10:
            base_y = y1 + 5

        def _fmt(v):
            return f"{v:.5f}" if v is not None else "----"

        lines = [
            f"Face_sim: {_fmt(face_sim)}",
            f"App_sim:  {_fmt(app_sim)}",
            f"Gait_sim: {_fmt(gait_sim)}",
        ]

        for i, text in enumerate(lines):
            org = (x1, base_y + i * line_h)
            cv2.putText(
                img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )


    def _draw_skeletons(self, img: np.ndarray, all_kpts: np.ndarray, hi_index: int | None = None):
        """
        all_kpts: (N, 17, 3)  形式（x,y,score）
        hi_index: 最良一致の人物インデックス（その人だけ配色を変更）
        """
        H, W = img.shape[:2]
        KPT_CONNECTIONS: List[Tuple[int,int]] = [
            (5,6), (5,7), (7,9), (6,8), (8,10),
            (11,12), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16),
            (0,5), (0,6), (1,2), (1,3), (2,4)
        ]

        t = max(1, int(round(max(H, W) / 800)))
        line_thick    = max(1, int(POSE_DRAW_THICKNESS * t))
        circle_radius = max(1, int(POSE_DRAW_RADIUS   * t))

        for idx, kpts in enumerate(all_kpts):  # (17,3)
            #  色のルール：
            #   - 検出のみ or 非トップ  : 点=POSE_DRAW_LANDMARK_BGR, 線=POSE_DRAW_CONNECTION_BGR
            #   - 類似度トップの人のみ: 点=POSE_DRAW_CONNECTION_BGR, 線=TOP_MATCH_BBOX_COLOR
            if hi_index is not None and idx == hi_index:
                # 類似度が一番高い人
                kpt_bgr  = POSE_DRAW_CONNECTION_BGR   # ランドマーク
                line_bgr = TOP_MATCH_BBOX_COLOR       # コネクション
            else:
                # それ以外（検出だけ/非トップ）
                kpt_bgr  = POSE_DRAW_LANDMARK_BGR
                line_bgr = POSE_DRAW_CONNECTION_BGR

            # 接続線
            for a, b in KPT_CONNECTIONS:
                if a >= len(kpts) or b >= len(kpts):
                    continue
                xa, ya, sa = kpts[a]
                xb, yb, sb = kpts[b]
                if sa >= POSE_KPT_CONF_MIN and sb >= POSE_KPT_CONF_MIN:
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), line_bgr, line_thick, cv2.LINE_AA)

            # キーポイント（ジョイント）
            for x, y, s in kpts:
                if s >= POSE_KPT_CONF_MIN:
                    cv2.circle(img, (int(x), int(y)), circle_radius, kpt_bgr, -1, cv2.LINE_AA)

    def _draw_box(self, img: np.ndarray, box: np.ndarray, color, thick=False):
        if box is None: return
        H, W = img.shape[:2]
        t = max(3 if thick else 2, int(round(max(H,W)/600)))
        x1,y1,x2,y2 = map(int, box[:4])
        cv2.rectangle(img, (x1,y1),(x2,y2), color, t, cv2.LINE_AA)

    def _find_box_at(self, x: int, y: int, boxes: np.ndarray) -> Optional[int]:
        if boxes is None: return None
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = map(int, b[:4])
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def _update_selection_pose_iou(self, iou_th=0.1):
        """
        前フレームの選択Pose BBOX (_sel_pose_prev) とIOUが最大の
        現在のPose BBOXを探し、しきい値以上なら sel_pose_idx を更新する。
        マッチしない場合は選択をクリアする。
        """
        if self._sel_pose_prev is None or self._boxes_pose is None:
            return

        best = None
        best_iou = 0.0
        for i, b in enumerate(self._boxes_pose):
            iou = self._iou(self._sel_pose_prev, b)
            if iou > best_iou:
                best_iou = iou
                best = i

        if best is not None and best_iou >= float(iou_th):
            self.sel_pose_idx = int(best)
            self._sel_pose_prev = self._boxes_pose[int(best)].copy()
        else:
            # IOUしきい値未満の場合は選択を解除しておく
            self.sel_pose_idx = None
            self._sel_pose_prev = None

    def _update_face_link_for_selected_pose(self):
        """
        sel_pose_idx で選択中のPose BBOXに内包されるFace BBOXインデックスを1つだけ紐付ける。
        検出数の変化で sel_pose_idx が範囲外になっていた場合は選択をクリアして終了する。
        """
        self.sel_face_idx = None
        if self.sel_pose_idx is None or self._boxes_pose is None or self._boxes_face is None:
            return

        # ★ 検出数の変化などで sel_pose_idx が範囲外なら選択をリセット
        idx = int(self.sel_pose_idx)
        if idx < 0 or idx >= len(self._boxes_pose):
            self.sel_pose_idx = None
            self._sel_pose_prev = None
            return

        pb = self._boxes_pose[idx]
        for i, fb in enumerate(self._boxes_face):
            if self._is_inside(fb, pb):
                self.sel_face_idx = i
                break
            
    def _update_face_links_all(self):
        """各Pose BBOXに含まれるFace BBOXインデックスを記録（類似度描画用）"""
        self._pose_face_index = None
        if self._boxes_pose is None or self._boxes_face is None:
            return

        links: List[Optional[int]] = [None] * len(self._boxes_pose)
        for pi, pb in enumerate(self._boxes_pose):
            for fi, fb in enumerate(self._boxes_face):
                if self._is_inside(fb, pb):
                    links[pi] = fi
                    break

        self._pose_face_index = links

    @staticmethod
    def _is_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
        x1,y1,x2,y2 = inner[:4]; X1,Y1,X2,Y2 = outer[:4]
        return (x1>=X1 and y1>=Y1 and x2<=X2 and y2<=Y2)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1,ay1,ax2,ay2 = a[:4]; bx1,by1,bx2,by2 = b[:4]
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        w, h = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = w*h
        area = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
        return float(inter / area)

    @staticmethod
    def _crop(img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        if img is None or box is None: return None
        h,w = img.shape[:2]
        x1,y1,x2,y2 = [int(max(0,min(v, (w if i%2==0 else h)-1))) for i,v in enumerate(box[:4])]
        if x2<=x1+1 or y2<=y1+1: return None
        return img[y1:y2, x1:x2].copy()

    @staticmethod
    def _mean(arr: List[np.ndarray]) -> Optional[np.ndarray]:
        if not arr: return None
        m = np.mean(np.stack(arr, axis=0), axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        return m.astype(np.float32)
    
    @staticmethod
    def _yaw_to_view(yaw_deg: float) -> str:
        """
        MediaPipe の yaw[deg] を
          - 0° 近辺      → "front"
          - ±90° 近辺    → "side"
          - ±180° 近辺   → "back"
        にざっくり量子化。
        """
        a = abs(float(yaw_deg))

        # ざっくり 0±45° を正面扱い
        if a <= 45.0:
            return "front"
        # 135°〜180° を背面
        if a >= 135.0:
            return "back"
        # それ以外は側面（45°〜135°）
        return "side"

    def _has_selection(self) -> bool:
        """
        sel_pose_idx が有効な BBOX を指しているかどうかをチェック。
        検出がゼロ件になったフレームでは False にする。
        """
        if self.sel_pose_idx is None:
            return False
        if self._boxes_pose is None:
            return False
        idx = int(self.sel_pose_idx)
        return 0 <= idx < len(self._boxes_pose)

    def _has_any_feat(self) -> bool:
        return bool(self._buf_face or self._buf_app or self._buf_gait)

    def _append_json(self, path: str, label: str, vec: Optional[np.ndarray]):
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        label = str(label)
        if vec is not None:
            v = np.asarray(vec, np.float32).reshape(-1).tolist()
            data.setdefault("label_to_vecs", {}).setdefault(label, []).append(v)
            # 次元も保存（推奨）
            data["dim"] = len(v)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    def _append_face_pose_json(
        self,
        path: str,
        label: str,
        pose: str,
        vec: Optional[np.ndarray],
    ):
        """
        マルチモーダルモードの顔ギャラリー用:
        - label_to_vecs は従来通り更新
        - 追加で records に {label, pose, vec} を追記
        """
        if vec is None:
            return

        # 既存データ読み込み
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        label = str(label or "").strip()
        pose  = str(pose or "").strip()
        if not label or not pose:
            # ラベル or 姿勢が空なら何もしない
            return

        # ベクトルを Python の list に
        v = np.asarray(vec, np.float32).reshape(-1).tolist()

        # --- 従来形式: label_to_vecs も更新して互換性維持 ---
        if not isinstance(data, dict):
            data = {}
        lab2 = data.setdefault("label_to_vecs", {})
        if not isinstance(lab2, dict):
            lab2 = {}
            data["label_to_vecs"] = lab2
        lab2.setdefault(label, []).append(v)

        # --- 新形式: records 配列に姿勢付きで追記 ---
        recs = data.setdefault("records", [])
        if not isinstance(recs, list):
            recs = []
            data["records"] = recs

        recs.append({
            "label": label,
            "pose": pose,
            "vec":  v,
        })

        data["dim"] = len(v)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self.log.info("[AUTO][SAVE] wrote: label=%s pose=%s dim=%d -> %s", label, pose, len(v), path)
        
    def _append_gait_view_json(
        self,
        path: str,
        label: str,
        view: str,
        vec: Optional[np.ndarray],
    ):
        """
        マルチモーダルモードの歩容ギャラリー用:
        - label_to_vecs は従来通り更新
        - 追加で records に {label, view, vec} を追記
        """
        if vec is None:
            return

        # 既存データ読み込み
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        label = str(label or "").strip()
        view  = str(view  or "").strip()
        if not label or not view:
            # ラベル or view が空なら何もしない
            return

        # ベクトルを Python の list に
        v = np.asarray(vec, np.float32).reshape(-1).tolist()

        # --- 従来形式: label_to_vecs も更新して互換性維持 ---
        if not isinstance(data, dict):
            data = {}
        lab2 = data.setdefault("label_to_vecs", {})
        if not isinstance(lab2, dict):
            lab2 = {}
            data["label_to_vecs"] = lab2
        lab2.setdefault(label, []).append(v)

        # --- 新形式: records 配列に view 付きで追記 ---
        recs = data.setdefault("records", [])
        if not isinstance(recs, list):
            recs = []
            data["records"] = recs

        recs.append({
            "label": label,
            "view":  view,
            "vec":   v,
        })

        data["dim"] = len(v)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.log.info(
            "[AUTO][SAVE][GAIT] wrote: label=%s view=%s dim=%d -> %s",
            label, view, len(v), path,
        )

    def _load_auto_label_vectors(self, label: str) -> None:
        def _norm(v):
            v = np.asarray(v, np.float32)
            if v.ndim == 2:  # 複数なら平均
                v = v.mean(axis=0)
            n = float(np.linalg.norm(v) + 1e-12)
            return (v / n).astype(np.float32)

        def _pick_from_record(rec: dict, want_label: str):
            """
            1レコードからベクトルを取り出すユーティリティ。
            label/pid/name のいずれか一致→ vec/feature/emb/centroid/vecs/features から取り出し。
            """
            key = str(rec.get("label") or rec.get("pid") or rec.get("name") or "")
            if key != want_label:
                return None
            # 単体ベクトル候補
            for k in ("vec", "feature", "emb", "centroid"):
                if k in rec and rec[k] is not None:
                    return _norm(rec[k])
            # 複数ベクトル候補（平均化）
            for k in ("vecs", "features", "embs"):
                if k in rec and rec[k]:
                    return _norm(rec[k])
            return None

        def _read_one_vec(json_path: str, want_label: str):
            try:
                if not json_path or not os.path.exists(json_path):
                    return None
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 1) 辞書スタイルの代表的スキーマ
                if isinstance(data, dict):
                    # label_to_vecs / pid_to_vecs
                    for map_key in ("label_to_vecs", "pid_to_vecs", "name_to_vecs"):
                        if map_key in data and isinstance(data[map_key], dict):
                            vecs = data[map_key].get(want_label)
                            if vecs is not None:
                                return _norm(vecs)

                    # *_to_centroid
                    for map_key in ("label_to_centroid", "pid_to_centroid", "name_to_centroid"):
                        if map_key in data and isinstance(data[map_key], dict):
                            vec = data[map_key].get(want_label)
                            if vec is not None:
                                return _norm(vec)

                    # records配列を内包する形 {"records":[{...}, ...]}
                    if "records" in data and isinstance(data["records"], list):
                        for rec in data["records"]:
                            if isinstance(rec, dict):
                                v = _pick_from_record(rec, want_label)
                                if v is not None:
                                    return v

                # 2) 配列スタイル [{label/pid: "...", vec/feature/...: ...}, ...]
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            v = _pick_from_record(rec, want_label)
                            if v is not None:
                                return v

                return None
            except Exception as e:
                self.log.exception("[AUTO][GALLERY] load failed: %s (%s)", json_path, e)
                return None

        face_json = AUTO_FACE_JSON
        app_json  = AUTO_APP_JSON
        gait_json = AUTO_GAIT_JSON

        gf = _read_one_vec(face_json,  label)
        ga = _read_one_vec(app_json,   label)
        gg = _read_one_vec(gait_json,  label)

        self._gallery_face = gf
        self._gallery_app  = ga
        self._gallery_gait = gg

        # 歩容：view 別のギャラリーベクトルも読み込み
        if AUTO_GAIT_YAW_ENABLED:
            try:
                self._gallery_gait_by_view = self._load_gait_gallery_by_view(label)
            except Exception:
                self._gallery_gait_by_view = {}
        else:
            self._gallery_gait_by_view = {}

        self.log.info(
            "[AUTO][GALLERY] loaded: face=%s, appearance=%s, gait=%s, gait_views=%s",
            "OK" if gf is not None else "None",
            "OK" if ga is not None else "None",
            "OK" if gg is not None else "None",
            sorted(list(self._gallery_gait_by_view.keys())) if self._gallery_gait_by_view else "(none)",
        )

    def _reset_runtime(self, light: bool = True):
        self._boxes_face = self._scores_face = None
        self._boxes_pose = self._scores_pose = self._kpts_pose = None
        self.sel_face_idx = self.sel_pose_idx = None
        self._sel_pose_prev = None
        self._buf_face.clear(); self._buf_app.clear(); self._buf_gait.clear()
        self._buf_face_by_pose.clear()
        self._face_pose_run_type = None
        self._face_pose_run_vecs = []
        if not light:
            self.face_det = None; self.pose_det = None
            self.face_embed = None; self.app_embed = None; self.gait_embed = None
            
    def _load_face_gallery_by_pose(self, label: str) -> Dict[str, np.ndarray]:
        """
        保存形式（records: [{label, pose, vec}]）から、指定labelのpose別ベクトル辞書を作る。
        同一(label, pose)が複数ある場合は平均→L2正規化。
        """
        label = (label or "").strip()
        path = AUTO_FACE_JSON
        pose_map: Dict[str, list] = {}

        try:
            if not (path and os.path.isfile(path)):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}

        recs = []
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            recs = data["records"]
        elif isinstance(data, list):
            recs = data
        else:
            # 従来形式のみ（label_to_vecs だけ）の場合は pose別は作れない
            return {}

        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("label", "")).strip() != label:
                continue
            pose = str(rec.get("pose", "")).strip()
            vec  = rec.get("vec")
            if not pose or vec is None:
                continue
            v = np.asarray(vec, np.float32).reshape(-1)
            if v.size == 0:
                continue
            pose_map.setdefault(pose, []).append(v)

        out: Dict[str, np.ndarray] = {}
        for pose, arrs in pose_map.items():
            m = np.mean(np.stack(arrs, axis=0), axis=0)
            m /= (np.linalg.norm(m) + 1e-12)
            out[pose] = m.astype(np.float32)
        return out
    
    def _load_gait_gallery_by_view(self, label: str) -> Dict[str, np.ndarray]:
        """
        gait_gallery.json の records から、
        指定 label の view 別（front/back/side）平均ベクトルを作る。
        """
        label = (label or "").strip()
        path = AUTO_GAIT_JSON
        view_map: Dict[str, list] = {}

        try:
            if not (path and os.path.isfile(path)):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}

        # records を取り出し（dict でも list でも対応）
        recs = []
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            recs = data["records"]
        elif isinstance(data, list):
            recs = data
        else:
            return {}

        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("label", "")).strip() != label:
                continue

            view = str(rec.get("view", "")).strip().lower()
            if not view:
                continue

            v = rec.get("vec")
            if v is None:
                continue
            v = np.asarray(v, np.float32).reshape(-1)
            n = float(np.linalg.norm(v) + 1e-12)
            v = v / n
            view_map.setdefault(view, []).append(v)

        # 各 view ごとに平均して np.ndarray に
        out: Dict[str, np.ndarray] = {}
        for view, vecs in view_map.items():
            vv = np.stack(vecs, axis=0).mean(axis=0)
            n = float(np.linalg.norm(vv) + 1e-12)
            out[view] = (vv / n).astype(np.float32)

        return out

    def _gait_embed_from_seq(self, seq_kpts: np.ndarray):
        """
        seq_kpts: (T, 17, 3) float32
        return: (D,) float32 or None
        """
        if self.gait_embed is None or seq_kpts is None:
            return None
        
        if not hasattr(self, "_gait_api"):
            self._gait_api = None
        if not hasattr(self, "_gait_api_missing_logged"):
            self._gait_api_missing_logged = False
        
        # まずはキャッシュを参照、無ければ解決
        if self._gait_api is None:
            self._resolve_gait_api()
        if self._gait_api is None:
            return None
        try:
            return self._gait_api(seq_kpts)
        except Exception as e:
            # 例外が出たら一度だけ API を再解決してワンモアトライ
            self.log.warning("[AUTO][GAIT] cached API failed (%s), retry resolving...", e)
            self._gait_api = None
            self._resolve_gait_api()
            if self._gait_api is None:
                return None
            try:
                return self._gait_api(seq_kpts)
            except Exception:
                return None
    
    def _resolve_gait_api(self):
        """
        GaitMixerRunner のAPI名に合わせて一度だけ解決→キャッシュ。
        返す callable: np.ndarray (T,17,3) -> np.ndarray (D,)
        """
        if self.gait_embed is None:
            return None

        # 1) 一括推論：まずは embed_sequence(seq) を最優先
        fn = getattr(self.gait_embed, "embed_sequence", None)
        if callable(fn):
            def _call(seq):
                # まず (T,17,3) で試し、ダメなら (T,17,2)
                try:
                    v = fn(seq)
                except TypeError:
                    v = fn(seq[:, :, :2])
                if v is None:
                    return None
                v = np.asarray(v, np.float32).reshape(-1)
                n = float(np.linalg.norm(v) + 1e-12)
                return v / n
            self._gait_api = _call
            self.log.info("[AUTO][GAIT] API resolved: embed_sequence(seq)")
            return self._gait_api

        # 2) 逐次バッファ型：consume_kpts(...) → build_gallery_from_buffer()
        fn_push = getattr(self.gait_embed, "consume_kpts", None)
        fn_build = getattr(self.gait_embed, "build_gallery_from_buffer", None)
        fn_clear = getattr(self.gait_embed, "clear_buffer", None)
        if callable(fn_push) and callable(fn_build):
            def _call(seq):
                try:
                    # まず (17,3) を逐次投入、失敗したら (17,2) で再投入
                    try:
                        for t in range(seq.shape[0]):
                            fn_push(seq[t])          # (17,3) 想定
                    except TypeError:
                        for t in range(seq.shape[0]):
                            fn_push(seq[t, :, :2])   # (17,2) フォールバック
                    v = fn_build()                    # バッファから最終特徴へ
                    if v is None:
                        return None
                    v = np.asarray(v, np.float32).reshape(-1)
                    n = float(np.linalg.norm(v) + 1e-12)
                    return v / n
                finally:
                    if callable(fn_clear):
                        try: fn_clear()
                        except Exception: pass
            self._gait_api = _call
            self.log.info("[AUTO][GAIT] API resolved: consume_kpts -> build_gallery_from_buffer")
            return self._gait_api

        # 3) そのほかの汎用候補（保険）
        for name in ["embed_seq", "embed", "forward", "infer_sequence", "infer", "run"]:
            fn = getattr(self.gait_embed, name, None)
            if callable(fn):
                def _call(seq, _fn=fn):
                    try:
                        v = _fn(seq)
                    except TypeError:
                        v = _fn(seq[:, :, :2])
                    if v is None:
                        return None
                    v = np.asarray(v, np.float32).reshape(-1)
                    n = float(np.linalg.norm(v) + 1e-12)
                    return v / n
                self._gait_api = _call
                self.log.info("[AUTO][GAIT] API resolved: %s(seq)", name)
                return self._gait_api

        # 4) 解決不可：一度だけ一覧を出して沈黙
        if not getattr(self, "_gait_api_missing_logged", False):
            self._gait_api_missing_logged = True
            try:
                names = [x for x in dir(self.gait_embed) if not x.startswith("_")]
            except Exception:
                names = []
            self.log.error("[AUTO][GAIT] No compatible API found in GaitMixerRunner. exports=%s", names[:80])
        self._gait_api = None
        return None
    
    def _ensure_auto_dirs(self):
        try:
            os.makedirs(AUTO_GALLERY_DIR, exist_ok=True)
        except Exception as e:
            self.log.warning("[AUTO][SAVE] make dir failed: %s", e)

    def _calc_next_serial_for_base(self, base: str) -> int:
        """
        base: "Mizuta" のようなベース名
        既存の auto/ 各 JSON を横断して、 base-<PREFIX><NNNN> の N の最大値+1 を返す
        見つからなければ 1
        """
        max_n = 0
        pat = re.compile(rf"^{re.escape(base)}-{re.escape(AUTO_SERIAL_PREFIX)}(\d+)$", re.IGNORECASE)

        def _scan_json(path: str):
            nonlocal max_n
            if not os.path.isfile(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return
            # list/dict どちらにも耐性を持たせる
            if isinstance(data, list):
                it = data
            elif isinstance(data, dict):
                # 例: {"label_to_vecs": {"Mizuta-P0001":[...], ...}}
                it = []
                if "label_to_vecs" in data and isinstance(data["label_to_vecs"], dict):
                    for k in data["label_to_vecs"].keys():
                        it.append({"pid": k})
                else:
                    # その他のdict形式は総当たり
                    for k, v in data.items():
                        it.append({k: v})
            else:
                it = []

            for rec in it:
                # 候補キーから pid/label を抽出
                val = None
                if isinstance(rec, dict):
                    for key in ("pid", "label", "name", "id"):
                        if key in rec:
                            val = str(rec[key])
                            break
                    # dictのキーそのものが PID になっている場合も拾う
                    if val is None and len(rec) == 1:
                        val = str(next(iter(rec.keys())))
                elif isinstance(rec, str):
                    val = rec

                if not val:
                    continue
                m = pat.match(val.strip())
                if m:
                    try:
                        n = int(m.group(1))
                        if n > max_n:
                            max_n = n
                    except Exception:
                        pass

        for p in (AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON):
            _scan_json(p)

        return (max_n + 1) if max_n > 0 else 1

    def _ensure_full_label(self) -> str | None:
        """
        _target_label（完全）を持っていなければ、_target_label_base から自動採番して生成。
        生成に成功したら _target_label を更新して返す。
        """
        if self._target_label and re.search(r"-[A-Za-z]?(\d{2,})$", self._target_label):
            return self._target_label
        base = (self._target_label_base or "").strip()
        if not base:
            return None
        self._ensure_auto_dirs()
        n = self._calc_next_serial_for_base(base)
        full = f"{base}-{AUTO_SERIAL_PREFIX}{n:0{AUTO_SERIAL_WIDTH}d}"
        self._target_label = full
        self.log.info("[AUTO] target_label(auto-assigned)=%s (base=%s)", full, base)
        return full
    
    def _load_label_vecs_from_json(path: str, label: str) -> np.ndarray | None:
        """
        保存形式が list/dict/{"label_to_vecs":{...}} のどれでも拾えるように耐性あり。
        返り値: (N, D) の numpy 配列（無ければ None）
        """
        if not (path and os.path.isfile(path)):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        vecs = []
        # パターン1: {"label_to_vecs": {"Mizuta-P0001":[[...],[...]], ...}}
        if isinstance(data, dict) and "label_to_vecs" in data and isinstance(data["label_to_vecs"], dict):
            arr = data["label_to_vecs"].get(label)
            if isinstance(arr, list):
                for v in arr:
                    a = np.asarray(v, dtype=np.float32)
                    if a.ndim == 1:
                        vecs.append(a)
                    elif a.ndim == 2:
                        vecs.extend([np.asarray(r, dtype=np.float32) for r in a])
        # パターン2: list of dicts [{"pid":"Mizuta-P0001","vec":[...]}, ...]
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    pid = str(rec.get("pid") or rec.get("label") or rec.get("name") or rec.get("id") or "").strip()
                    if pid == label and "vec" in rec:
                        a = np.asarray(rec["vec"], dtype=np.float32)
                        if a.ndim == 1:
                            vecs.append(a)
                        elif a.ndim == 2:
                            vecs.extend([np.asarray(r, dtype=np.float32) for r in a])
        # パターン3: その他（dictでキー=pid の場合など）
        elif isinstance(data, dict):
            if label in data:
                a = np.asarray(data[label], dtype=np.float32)
                if a.ndim == 1:
                    vecs.append(a)
                elif a.ndim == 2:
                    vecs.extend([np.asarray(r, dtype=np.float32) for r in a])

        if not vecs:
            return None
        # L2正規化前提のコサイン類似なので、ここで一応正規化
        vs = np.stack(vecs, axis=0)
        n = np.linalg.norm(vs, axis=1, keepdims=True) + 1e-12
        return vs / n
    
    def _cosine_best(q: np.ndarray, gallery: np.ndarray | None) -> float | None:
        """
        q: (D,), gallery: (N,D)
        返り値: 最大コサイン類似度（無ければ None）
        """
        if gallery is None or gallery.size == 0:
            return None
        q = q.astype(np.float32, copy=False)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (gallery @ q)  # (N,)
        return float(np.max(sims))
    
    def _update_tracker_and_track_ids(self) -> None:
        """
        Face + Pose の検出結果を ByteTracker に渡し、
        返ってきた Track 一覧から Pose / Face それぞれに track_id を割り当てる。
        """
        # 検出が何もなければ TrackID はリセット
        if (self._boxes_pose is None or len(self._boxes_pose) == 0) and \
           (self._boxes_face is None or len(self._boxes_face) == 0):
            self._track_ids_pose = None
            self._track_ids_face = None
            return

        dets_list = []

        # Pose の検出を (N,5) [x1,y1,x2,y2,score] 形式に
        if self._boxes_pose is not None and self._scores_pose is not None and len(self._boxes_pose) > 0:
            dets_pose = np.concatenate(
                [self._boxes_pose, self._scores_pose.reshape(-1, 1)],
                axis=1
            )
            dets_list.append(dets_pose)

        # Face の検出を (M,5) で追加
        if self._boxes_face is not None and self._scores_face is not None and len(self._boxes_face) > 0:
            dets_face = np.concatenate(
                [self._boxes_face, self._scores_face.reshape(-1, 1)],
                axis=1
            )
            dets_list.append(dets_face)

        if not dets_list:
            self._track_ids_pose = None
            self._track_ids_face = None
            return

        dets = np.vstack(dets_list).astype(np.float32)  # (K,5)

        # トラッカー本体を初期化（まだの場合）
        if self._tracker is None:
            self._tracker = ByteTracker(
                track_thresh=BYTE_TRACK_TRACK_THRESH,
                high_thresh=BYTE_TRACK_HIGH_THRESH,
                match_thresh=BYTE_TRACK_MATCH_THRESH,
                max_age=BYTE_TRACK_MAX_AGE,
            )

        try:
            tracks = self._tracker.update(dets)
        except Exception as e:
            self.log.exception("[AUTO] tracker.update failed: %s", e)
            self._track_ids_pose = None
            self._track_ids_face = None
            self._active_track_ids = set()
            return

        # 今フレーム時点での「アクティブな TrackID」を保存
        self._active_track_ids = {t.track_id for t in tracks}

        # Track → Pose / Face それぞれに ID を振り分け
        self._track_ids_pose = self._assign_track_ids(self._boxes_pose, tracks)
        self._track_ids_face = self._assign_track_ids(self._boxes_face, tracks)

    def _assign_track_ids(self, boxes: Optional[np.ndarray], tracks: List[Track]) -> Optional[np.ndarray]:
        """
        任意の BBOX 群に対して、IoU が最大の Track をひとつ対応付ける。
        IoU < 0.3 の場合は track_id = -1 とする。
        """
        if boxes is None or len(boxes) == 0 or not tracks:
            return None

        boxes = np.asarray(boxes, dtype=np.float32)
        n = boxes.shape[0]

        track_boxes = np.stack([t.tlbr for t in tracks], axis=0).astype(np.float32)

        # IoU 計算（(N,T)）
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        tx1, ty1, tx2, ty2 = np.split(track_boxes, 4, axis=1)

        inter_x1 = np.maximum(x1, tx1.T)
        inter_y1 = np.maximum(y1, ty1.T)
        inter_x2 = np.minimum(x2, tx2.T)
        inter_y2 = np.minimum(y2, ty2.T)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_b = (x2 - x1) * (y2 - y1)
        area_t = (tx2 - tx1) * (ty2 - ty1)
        union = area_b + area_t.T - inter
        iou = np.where(union > 0, inter / union, 0.0)

        track_ids = np.full(n, -1, dtype=np.int32)
        for i in range(n):
            j = int(iou[i].argmax())
            if iou[i, j] >= 0.3:
                track_ids[i] = tracks[j].track_id

        return track_ids

        # ---------------- マルチモーダルレポート用ユーティリティ ----------------

    def _get_video_filename(self) -> str:
        """動画ファイル名だけを返す（取れない場合は空文字）"""
        try:
            player = getattr(self.win, "player", None)
            if player is None:
                return ""
            # 環境に合わせて属性名は調整してください
            for attr in ("current_path", "video_path", "source_path"):
                if hasattr(player, attr):
                    path = getattr(player, attr)
                    if path:
                        return os.path.basename(str(path))
        except Exception:
            pass
        return ""

    def _get_video_time_label(self) -> str:
        """
        現在の動画位置を "MM:SS" 形式の文字列で返す。
        取得できない場合は "--:--"
        """
        sec = None
        try:
            player = getattr(self.win, "player", None)
            if player is not None:
                if hasattr(player, "get_position_sec"):
                    sec = float(player.get_position_sec())
                elif hasattr(player, "get_current_frame") and hasattr(player, "get_fps"):
                    idx = int(player.get_current_frame())
                    fps = float(player.get_fps() or 30.0)
                    sec = idx / max(fps, 1.0)
        except Exception:
            sec = None

        if sec is None or sec < 0:
            return "--:--"

        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    def _create_report_session_dir(self) -> str:
        """
        HTML と 人物キャプチャ画像を保存するディレクトリ。
        今回の要望に合わせて、セッション用サブフォルダは作らず、
        data/evidence/ranking 配下をそのまま使う。
        """
        os.makedirs(AUTO_REPORT_DIR, exist_ok=True)
        return AUTO_REPORT_DIR

    def _capture_topN_snapshot(self) -> None:
        """
        現在フレームの BBOX 群から、
        TrackID ごとの“平均類似度”にもとづいて上位N件を選び、
        クロップ画像＋類似度をスナップショットとして保持する。
        """
        if self._boxes_pose is None or self._track_ids_pose is None:
            return
        n_boxes = int(self._boxes_pose.shape[0])
        if n_boxes <= 0:
            return


        # T-IDごとの統計から「平均類似度」をBBOX配列に展開
        sim_face_avg: list[Optional[float]] = [None] * n_boxes
        sim_app_avg:  list[Optional[float]] = [None] * n_boxes
        sim_gait_avg: list[Optional[float]] = [None] * n_boxes

        for i in range(n_boxes):
            tid = int(self._track_ids_pose[i])
            if tid < 0:
                continue  # 未割り当て

            # Face 平均
            rec_f = self._track_stats_face.get(tid)
            if rec_f and rec_f.get("count", 0) > 0:
                sim_face_avg[i] = float(rec_f.get("avg", 0.0))

            # Appearance 平均
            rec_a = self._track_stats_app.get(tid)
            if rec_a and rec_a.get("count", 0) > 0:
                sim_app_avg[i] = float(rec_a.get("avg", 0.0))

            # Gait 平均
            rec_g = self._track_stats_gait.get(tid)
            if rec_g and rec_g.get("count", 0) > 0:
                sim_gait_avg[i] = float(rec_g.get("avg", 0.0))

        # ---- ランキング作成（各モダリティの topN を union → 閾値判定）----
        entries: list[RankEntry] = []

        def _valid(v: Optional[float]) -> Optional[float]:
            """None / 変換不可 / NaN を弾いて float を返すヘルパー。"""
            if v is None:
                return None
            try:
                f = float(v)
            except Exception:
                return None
            if np.isnan(f):
                return None
            return f

        # それぞれの識別方法ごとに
        #   1. 閾値未満を除外
        #   2. 類似度の高い順に AUTO_TRACK_RANK_TOPN 件まで取得
        maxN = AUTO_TRACK_RANK_TOPN if (AUTO_TRACK_RANK_TOPN and AUTO_TRACK_RANK_TOPN > 0) else n_boxes

        scored_face: list[tuple[int, float]] = []
        scored_app:  list[tuple[int, float]] = []
        scored_gait: list[tuple[int, float]] = []

        for i in range(n_boxes):
            f = _valid(sim_face_avg[i])
            if f is not None and f > float(self._face_thresh):
                scored_face.append((i, f))

            a = _valid(sim_app_avg[i])
            if a is not None and a > float(self._app_thresh):
                scored_app.append((i, a))

            g = _valid(sim_gait_avg[i])
            if g is not None and g > float(self._gait_thresh):
                scored_gait.append((i, g))

        scored_face.sort(key=lambda t: t[1], reverse=True)
        scored_app.sort(key=lambda t: t[1], reverse=True)
        scored_gait.sort(key=lambda t: t[1], reverse=True)

        if maxN > 0:
            scored_face = scored_face[:maxN]
            scored_app  = scored_app[:maxN]
            scored_gait = scored_gait[:maxN]

        # TrackID（または index）ごとに重複を除外するためのキー
        used_track_keys: set[int] = set()

        def _track_key_for_index(idx: int) -> int:
            if self._track_ids_pose is None:
                return idx
            try:
                tid = int(self._track_ids_pose[idx])
            except Exception:
                return idx
            return tid if tid >= 0 else idx

        def _make_entry(idx: int, used_modality: str) -> RankEntry:
            f = _valid(sim_face_avg[idx])
            a = _valid(sim_app_avg[idx])
            g = _valid(sim_gait_avg[idx])
            return RankEntry(
                index=idx,
                rank=0,  # このあとブロック順に 1,2,3... を振る
                used_modality=used_modality,
                face_sim=f,
                app_sim=a,
                gait_sim=g,
            )

        # 顔ブロック → 外見ブロック → 歩容ブロック の順で entries を構成。
        # すでに entries に入っている TrackID はスキップ（重複削除）。
        for idx, _ in scored_face:
            key = _track_key_for_index(idx)
            if key in used_track_keys:
                continue
            used_track_keys.add(key)
            entries.append(_make_entry(idx, "face"))

        for idx, _ in scored_app:
            key = _track_key_for_index(idx)
            if key in used_track_keys:
                continue
            used_track_keys.add(key)
            entries.append(_make_entry(idx, "app"))

        for idx, _ in scored_gait:
            key = _track_key_for_index(idx)
            if key in used_track_keys:
                continue
            used_track_keys.add(key)
            entries.append(_make_entry(idx, "gait"))

        if not entries:
            return

        # 最終的な上限数（理論上は最大 3 * AUTO_TRACK_RANK_TOPN 件）
        if AUTO_TRACK_RANK_TOPN and AUTO_TRACK_RANK_TOPN > 0:
            max_final = 3 * int(AUTO_TRACK_RANK_TOPN)
            entries = entries[:max_final]

        # ここから下は、これまでと同じでOK（キャプチャ＋HTML用データ作成）
        snapshots_for_frame: list[dict] = []
        for rank, ent in enumerate(entries, start=1):
            idx = ent.index  # BBOXインデックス
            box = self._boxes_pose[idx]
            crop = self._crop(self._last_frame, box)
            if crop is None:
                continue

            # 画像ファイル名（例: t000060_r1.jpg）
            fname = f"t{self._auto_report_frame_counter:06d}_r{rank}.jpg"
            img_path = os.path.join(self._auto_report_session_dir, fname)
            cv2.imwrite(img_path, crop)

            # ★ ent.face_sim / app_sim / gait_sim は
            #   いま渡した「平均類似度」に対応する
            snapshots_for_frame.append(
                {
                    "rank": rank,
                    "track_id": int(self._track_ids_pose[idx]),
                    "image_path": fname,  # HTMLでは basename で参照
                    "face_sim": ent.face_sim,
                    "app_sim":  ent.app_sim,
                    "gait_sim": ent.gait_sim,
                    "frame_index": self._auto_report_frame_counter,
                }
            )

        if not snapshots_for_frame:
            return
        
        label_time = self._get_video_time_label()

        # 何分何秒か（秒→mm:ss）などは従来通り
        if not label_time or label_time == "--:--":
            t_sec = (self._auto_report_frame_counter / max(self._fps or 30.0, 1.0))
            mm = int(t_sec // 60)
            ss = int(t_sec % 60)
            label_time = f"{mm:02d}:{ss:02d}"

        self._auto_report_snapshots.append(
            {
                "time_label": label_time,
                "records": snapshots_for_frame,
            }
        )
        
        self._reset_interval_stats()
        self._interval_cleared_frame = self._frame_counter

    def _finalize_auto_report(self) -> None:
        """
        識別OFFになったタイミングで、メモリ上のスナップショットからHTMLを生成。
        HTMLファイル名は data/evidence/ranking/ranking_日時.html とする。
        日時は「識別OFFにした瞬間の現在日時」。
        """
        # 状態ログ
        self.log.info(
            "[AUTO][REPORT] finalize called. builder=%s, snapshots=%d",
            type(self._auto_report_builder).__name__ if self._auto_report_builder else "None",
            len(self._auto_report_snapshots) if isinstance(self._auto_report_snapshots, list) else -1,
        )

        # ① builder が None の場合：すでに確定済み or そもそも何もしていない
        if self._auto_report_builder is None:
            self.log.info(
                "[AUTO][REPORT] builder=None のためレポート生成はスキップします（既に確定済みの可能性）。"
            )
            # ★ cleanup は下の finally で必ず実行されるので、ここでは return だけ
            return

        # ② スナップショットが本当に 0 件の場合だけこのメッセージを出す
        if not self._auto_report_snapshots:
            self.log.info("[AUTO][REPORT] スナップショットがないためHTMLは生成しません。")
            return

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_name = f"ranking_{ts}.html"

            # --- dict → ReportSnapshot / ReportItem に変換 ---
            snapshots_dc: List[ReportSnapshot] = []
            for snap in self._auto_report_snapshots:
                time_label = snap.get("time_label", "")
                items_dc: List[ReportItem] = []
                for rec in snap.get("records", []):
                    items_dc.append(
                        ReportItem(
                            rank=int(rec.get("rank", 0)),
                            image_path=rec.get("image_path", ""),
                            face_sim=rec.get("face_sim"),
                            app_sim=rec.get("app_sim"),
                            gait_sim=rec.get("gait_sim"),
                        )
                    )
                if items_dc:
                    snapshots_dc.append(
                        ReportSnapshot(
                            time_label=time_label,
                            items=items_dc,
                        )
                    )

            if not snapshots_dc:
                self.log.info("[AUTO][REPORT] dataclass snapshots が空のためHTMLは生成しません。")
                return

            html_path = self._auto_report_builder.build_from_snapshots(
                snapshots=snapshots_dc,
                basename=html_name,
            )
            self.log.info("[AUTO][REPORT] HTML generated: %s", html_path)

        except Exception as e:
            self.log.exception("[AUTO][REPORT] build failed: %s", e)

        finally:
            # === ここで状態を完全リセット ===
            self._auto_report_builder = None
            self._auto_report_snapshots = []
            self._auto_report_session_dir = None
            self._auto_report_frame_counter = 0

            # ★ ランキング用の類似度統計をリセット
            self._track_stats_face.clear()
            self._track_stats_app.clear()
            self._track_stats_gait.clear()
            self._frame_counter = 0

            # ★ 歩容識別・学習用の32フレームバッファもリセット
            try:
                self._gseq_buffer.clear()
            except Exception:
                pass
            try:
                if hasattr(self, "_gait_seq"):
                    self._gait_seq.clear()
            except Exception:
                pass

            self._buf_gait.clear()
            self._sim_gait = None
            self._hi_gait_idx = None

    def _classify_face_pose_from_kpts(self, kpts: np.ndarray) -> Optional[str]:
        """
        YOLO-Pose(17 keypoints) の顔キーポイントから顔の向きラベルを判定する。

        戻り値:
          "front", "look-down",
          "right", "left",
          "right-look-down", "left-look-down"
          または None（判定不可）

        前提（YOLOv8-Pose の COCO 形式）:
          0: nose
          1: left_eye
          2: right_eye
          3: left_ear
          4: right_ear
        """
        if kpts is None:
            return None
        kpts = np.asarray(kpts, np.float32)
        if kpts.ndim != 2 or kpts.shape[0] < 5:
            return None

        def _has(idx: int) -> bool:
            if idx >= kpts.shape[0]:
                return False
            x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
            if not np.isfinite(x) or not np.isfinite(y):
                return False
            # 3列目があれば信頼度として扱う
            if kpts.shape[1] >= 3:
                conf = float(kpts[idx, 2])
                return conf >= float(POSE_KPT_CONF_MIN)
            return True

        has_nose = _has(0)
        has_leye = _has(1)
        has_reye = _has(2)
        has_lear = _has(3)
        has_rear = _has(4)

        # 目がそろっていないときは判定をやめる
        if not (has_nose and has_leye and has_reye):
            return None

        def _y(idx: int) -> float:
            return float(kpts[idx, 1])

        # 1) 目2 + 鼻 + 耳2 がそろっている → front / look-down
        if has_lear and has_rear:
            dy_left  = _y(3) - _y(1)  # 左耳 - 左目
            dy_right = _y(4) - _y(2)  # 右耳 - 右目
            # 画像座標では y が大きいほど「下」
            # 「目の方が耳より上 or 同じ」  → eye_y <= ear_y → dy >= 0
            # 「目の方が耳より下」            → eye_y >  ear_y → dy < 0
            if dy_left >= 0.0 and dy_right >= 0.0:
                # 目が耳より上か同じ → front
                return "front"
            if dy_left < 0.0 and dy_right < 0.0:
                # 目が耳より下 → look-down
                return "look-down"
            # 左右でばらつくときは平均で判定
            return "front" if (dy_left + dy_right) / 2.0 >= 0.0 else "look-down"

        # 2) 目2 + 鼻 + 右耳 → right / right-look-down
        if has_rear:
            dy_right = _y(4) - _y(2)
            # dy >= 0 → 目が耳より上 or 同じ → right
            # dy <  0 → 目が耳より下         → right-look-down
            return "right-look-down" if dy_right < 0.0 else "right"

        # 3) 目2 + 鼻 + 左耳 → left / left-look-down
        if has_lear:
            dy_left = _y(3) - _y(1)
            # dy >= 0 → 目が耳より上 or 同じ → left
            # dy <  0 → 目が耳より下         → left-look-down
            return "left-look-down" if dy_left < 0.0 else "left"

        return None

    def _update_face_learning_with_pose(self, frame: np.ndarray) -> None:
        """
        マルチモーダルモードの「顔特徴学習」処理。

        - YOLO-Pose の顔キーポイントから向きを判定
        - 向きごとに連続 AUTO_FACE_FRAMES 枚ぶんの AdaFace ベクトルを平均して 1 本に集約
        - front / look-down / right / left / right-look-down / left-look-down
          の最大 6 パターンを _buf_face / _buf_face_by_pose に格納する
        """
        # 学習対象の Face がなければ何もしない
        if self.sel_face_idx is None or self.face_embed is not None and self._boxes_face is None:
            # ↑ すこしややこしいので分けたほうが読みやすいなら分割してもOK
            pass

        if self.sel_face_idx is None or self.face_embed is None:
            return
        if self._boxes_face is None or len(self._boxes_face) == 0:
            return

        # Pose キーポイントを取得（選択中の Pose BBOX に対応）
        kpts = None
        if self._kpts_pose is not None and self.sel_pose_idx is not None:
            idx = int(self.sel_pose_idx)
            if 0 <= idx < len(self._kpts_pose):
                kpts = self._kpts_pose[idx]

        # Pose が取れない場合は「旧仕様（先頭 AUTO_FACE_FRAMES枚だけ）」にフォールバック
        if kpts is None:
            if len(self._buf_face) < int(AUTO_FACE_FRAMES):
                box = self._boxes_face[self.sel_face_idx]
                feat = self.face_embed.embed_one(frame, box)
                if feat is not None:
                    self._buf_face.append(np.asarray(feat, np.float32))
            return

        # 姿勢別バッファの初期化（念のため）
        if not hasattr(self, "_buf_face_by_pose"):
            self._buf_face_by_pose = {}
        if not hasattr(self, "_face_pose_run_type"):
            self._face_pose_run_type = None
        if not hasattr(self, "_face_pose_run_vecs"):
            self._face_pose_run_vecs = []
        if not hasattr(self, "_face_pose_finalized"):
            self._face_pose_finalized = set()

        # すでに 6 パターンすべて埋まっていれば何もしない
        if len(self._buf_face_by_pose) >= 6:
            return

        pose_type = self._classify_face_pose_from_kpts(kpts)
        if pose_type is None:
            # 条件を満たさないフレーム → 連続カウンタをリセット
            self._face_pose_run_type = None
            self._face_pose_run_vecs = []
            return

        # 既にこの向きは保存済み → 次の向き待ち
        if pose_type in self._face_pose_finalized:
            return

        # このフレームの顔特徴を計算
        box = self._boxes_face[self.sel_face_idx]
        feat = self.face_embed.embed_one(frame, box)
        if feat is None:
            return
        feat = np.asarray(feat, np.float32)

        need_frames = int(AUTO_FACE_FRAMES) if AUTO_FACE_FRAMES > 0 else 3

        if self._face_pose_run_type == pose_type:
            self._face_pose_run_vecs.append(feat)
        else:
            # 新しい向きで連続カウント開始
            self._face_pose_run_type = pose_type
            self._face_pose_run_vecs = [feat]

        cnt = len(self._face_pose_run_vecs)
        use_n = min(cnt, need_frames)

        try:
            v = np.mean(
                np.stack(self._face_pose_run_vecs[-use_n:], axis=0),  # 直近 use_n 枚
                axis=0,
            )
        except Exception:
            return

        n = float(np.linalg.norm(v) + 1e-12)
        v = (v / n).astype(np.float32)

        # 暫定/最終にかかわらず、常に最新の平均で上書き
        self._buf_face_by_pose[pose_type] = v
        self._buf_face.append(v)

        if cnt >= need_frames:
            # ★ AUTO_FACE_FRAMES 枚そろったので、この向きは「最終版」として確定
            self._face_pose_finalized.add(pose_type)
            self.log.info(
                "[AUTO][LEARN][face] pose=%s collected (final: %d frames, now %d types).",
                pose_type, use_n, len(self._buf_face_by_pose),
            )
            # 次の向き用に連続カウンタをリセット
            self._face_pose_run_type = None
            self._face_pose_run_vecs = []
        else:
            # ★ 暫定版のログ（何枚目か分かるように）
            self.log.info(
                "[AUTO][LEARN][face] pose=%s collected (temp: %d/%d frames, now %d types).",
                pose_type, cnt, need_frames, len(self._buf_face_by_pose),
            )

    def _pick_gallery_vec_for_pose(self, need_pose: Optional[str],
                                   pose_map: Dict[str, np.ndarray],
                                   default_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        need_poseに最も近い向きのギャラリー特徴を返す。
        フォールバック規則（ご指定どおり）を実装。
        """
        if pose_map is None or len(pose_map) == 0:
            return default_vec

        def has(p): return (p in pose_map)

        # 便宜上の正規化
        p = (need_pose or "").strip().lower()

        # 候補順序マップ
        order: Dict[str, list] = {
            "right": [
                "right",
                "front",
                "left",
                "right-look-down",   # look-down の同じ向き
                "look-down",
                "left-look-down",
            ],
            "left": [
                "left",
                "front",
                "right",
                "left-look-down",    # look-down の同じ向き
                "look-down",
                "right-look-down",
            ],
            "right-look-down": [
                "right-look-down",
                "left-look-down",    # look-down の別の向き
                "look-down",
                "front",
                "right",
                "left",
            ],
            "left-look-down": [
                "left-look-down",
                "right-look-down",   # look-down の別の向き
                "look-down",
                "front",
                "left",
                "right",
            ],
            "front": [
                "front",
                "right",
                "left",
                "right-look-down",
                "left-look-down",
                "look-down",
            ],
            "look-down": [
                "look-down",
                "front",
                "right",
                "left",
                "right-look-down",
                "left-look-down",
            ],
        }

        candidates = order.get(p, ["front", "right", "left", "look-down", "right-look-down", "left-look-down"])
        for c in candidates:
            if has(c):
                return pose_map[c]

        return default_vec
    
    def _pick_gallery_vec_for_view(self,
                                   need_view: Optional[str],
                                   view_map: Dict[str, np.ndarray],
                                   default_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        歩容の向き（front/back/side）に応じて、
        ギャラリーから一番優先度の高い view のベクトルを返す。

        優先ルール:
          - 正面の場合: front > back > side
          - 背面の場合: back  > front > side
          - 側面の場合: side  > front > back
        """
        if not view_map:
            return default_vec

        v = (need_view or "").strip().lower()

        def has(k: str) -> bool:
            return (k in view_map and view_map[k] is not None)

        if v == "front":
            order = ["front", "back", "side"]
        elif v == "back":
            order = ["back", "front", "side"]
        elif v == "side":
            order = ["side", "front", "back"]
        else:
            # 不明な場合は front 優先
            order = ["front", "back", "side"]

        for k in order:
            if has(k):
                return view_map[k]

        return default_vec
    
    def _pose_for_face_index(self, idx_face: int) -> Optional[str]:
        """
        顔BBOXに最も対応しそうなPose（人物）を探して、_classify_face_pose_from_kpts() で向きを返す。
        なければ None。
        """
        try:
            if self._boxes_face is None or self._kpts_pose is None or self._boxes_pose is None:
                return None
            if not (0 <= idx_face < len(self._boxes_face)):
                return None

            f = self._boxes_face[idx_face]
            fx1, fy1, fx2, fy2 = map(float, f[:4])
            cx, cy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5

            # 顔中心が内包されるPose bboxを優先。複数あれば最小中心距離を採用。
            best_j, best_d = None, None
            for j, pb in enumerate(self._boxes_pose or []):
                x1, y1, x2, y2 = map(float, pb[:4])
                if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                    pcx, pcy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    d = (pcx - cx) ** 2 + (pcy - cy) ** 2
                    if best_d is None or d < best_d:
                        best_d, best_j = d, j

            if best_j is None:
                return None

            kpts = self._kpts_pose[best_j]   # (17,3)
            pose_type = self._classify_face_pose_from_kpts(kpts)
            return pose_type
        except Exception:
            return None

    def set_html_thresholds(self, face: float, app: float, gait: float) -> None:
        """
        マルチモーダルモード HTML ランキング用のしきい値を更新する（0.0〜1.0）。
        UI から呼び出される想定。
        """
        # 一旦 float キャスト → 0〜1 にクリップ
        try:
            f = float(face)
        except Exception:
            f = AUTO_FACE_THRESH_DEFAULT
        try:
            a = float(app)
        except Exception:
            a = AUTO_APP_THRESH_DEFAULT
        try:
            g = float(gait)
        except Exception:
            g = AUTO_GAIT_THRESH_DEFAULT

        self._face_thresh = float(np.clip(f, 0.0, 1.0))
        self._app_thresh  = float(np.clip(a, 0.0, 1.0))
        self._gait_thresh = float(np.clip(g, 0.0, 1.0))

        self.log.info(
            "[AUTO][THRESH] face=%.5f app=%.5f gait=%.5sf",
            self._face_thresh, self._app_thresh, self._gait_thresh
        )
        
    def _update_gait_tracks(self) -> None:
        """
        現フレームの pose キーポイントと track_id から、
        track_id ごとの歩容シーケンス deque を更新する。

        - self._kpts_pose : List[np.ndarray] or np.ndarray, shape (Np, 17, 3)
        - self._track_ids_pose : List[int] or np.ndarray, shape (Np,)
        """
        # 属性がまだ無ければ初期化
        if not hasattr(self, "_gait_tracks"):
            self._gait_tracks = {}

        kpts_list = getattr(self, "_kpts_pose", None)
        track_ids = getattr(self, "_track_ids_pose", None)

        if kpts_list is None or track_ids is None:
            return

        # numpy / list 両対応のため enumerate で回す
        for pose_idx, tid in enumerate(track_ids):
            if tid is None or tid < 0:
                # 無効 track_id は無視
                continue

            # deque がなければ作る
            dq = self._gait_tracks.get(tid)
            if dq is None:
                dq = deque(maxlen=AUTO_GAIT_FRAMES)
                self._gait_tracks[tid] = dq

            # 対応するキーポイントを追加
            try:
                kpts = kpts_list[pose_idx]
            except Exception:
                continue
            if kpts is None:
                continue

            dq.append(kpts)
            
    def _build_gait_queries_from_tracks(self) -> tuple[list[int], list[np.ndarray]]:
        """
        track_id ごとに、AUTO_GAIT_FRAMES 以上たまっているものだけ
        (T,17,3) シーケンスを作り、GaitMixer に入れる準備をする。

        戻り値:
            track_ids : List[int]       # クエリ順の track_id
            seq_list  : List[np.ndarray]  # 各 (T,17,3) シーケンス
        """
        track_ids_pose = getattr(self, "_track_ids_pose", None)
        if track_ids_pose is None:
            return [], []

        if not hasattr(self, "_gait_tracks") or not self._gait_tracks:
            return [], []

        out_track_ids: list[int] = []
        seq_list: list[np.ndarray] = []

        # 「今フレームで生きている track_id だけ」を対象にする
        for tid in track_ids_pose:
            if tid is None or tid < 0:
                continue
            dq = self._gait_tracks.get(tid)
            if dq is None:
                continue
            if len(dq) < AUTO_GAIT_FRAMES:
                # まだ T フレームたまっていない
                continue

            # deque[(17,3)] → (T,17,3)
            seq = np.stack(list(dq), axis=0)
            out_track_ids.append(int(tid))
            seq_list.append(seq)

        return out_track_ids, seq_list
