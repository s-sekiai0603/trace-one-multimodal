import os

DEFAULT_VIDEO_PATH = r"C:\work\s_sekiai\trace-one\videos\学習用_ABCD\学習用_Cさん、Dさん_熊_4k.MOV"

# GPU： ["CUDAExecutionProvider","CPUExecutionProvider"]
# CPU： ["CPUExecutionProvider"]
ORT_PROVIDERS = ["CUDAExecutionProvider","CPUExecutionProvider"]
# GPU： "cuda:0"
# CPU： "cpu"
TORCH_DEVICE = "cuda:0"

VIDEO_ORIENTATION = {
    "AUTO_ROTATE": True,     # メタデータ(CAP_PROP_ROTATION/ORIENTATION_META)があれば自動適用
    "ROTATE_DEG": 0,         # 0/90/180/270 を指定すると手動上書き（0は上書きしない）
    "FORCE_VFLIP": False,    # Trueで上下反転（逆さ対策の手動スイッチ）
}

BBOX_COLOR = (255, 0, 0)
SELECTED_BBOX_COLOR = (0, 0, 255) 
TOP_MATCH_BBOX_COLOR  = (0, 255, 0)

FPS = 30

# 動画前処理
PREPROCESS = {
    "ENABLE": False,  # ★ 全体ON/OFF
    "ORDER": ["BRICON", "GAMMA", "HIST_EQ", "DENOISE", "SHARPEN", "SUPERRES"],

    "BRICON": {      # 明るさ/コントラスト
        "ENABLE": True,
        "ALPHA": 1.15,   # 1.0=等倍, >1でコントラスト↑
        "BETA": 8.0      # -255〜255 で明るさ調整
    },
    "GAMMA": {       # ガンマ補正（<1で明るく、>1で暗く）
        "ENABLE": True,
        "GAMMA": 0.95
    },
    "HIST_EQ": {     # ヒストグラム平坦化（CLAHE）
        "ENABLE": True,
        "CLAHE_CLIP": 2.0,
        "CLAHE_TILE": 8
    },
    "DENOISE": {     # ノイズ除去
        "ENABLE": True,
        "H": 3.0,
        "HCOLOR": 3.0,
        "TEMPLATE_WINDOW": 7,
        "SEARCH_WINDOW": 21
    },
    "SHARPEN": {     # アンシャープマスク
        "ENABLE": True,
        "SIGMA": 1.2,
        "AMOUNT": 1.0,
        "THRESHOLD": 3.0
    },
    "SUPERRES": {    # 超解像（dnn_superresが使えれば使用）
        "ENABLE": False,
        "ENGINE": "edsr",        # edsr / espcn / fsrcnn / lapsrn
        "SCALE": 2,
        "MODEL_PATH": "models/EDSR_x2.pb"
    }
}


# ByteTrack 風トラッカー設定

# 有効/無効のフラグ（とりあえず常に True でOK。様子見したくなったらFalseに）
BYTE_TRACK_ENABLED      = True

# 検出スコアの閾値
BYTE_TRACK_TRACK_THRESH = 0.5   # これ未満の検出は追跡対象から除外
BYTE_TRACK_HIGH_THRESH  = 0.6   # 「高スコア」とみなす閾値（最初のマッチング用）

# IoUマッチング閾値
BYTE_TRACK_MATCH_THRESH = 0.7   # BBOX同士のIoUがこれ以上なら同一トラックとみなす

# 何フレーム検出が途切れたらトラックを消すか
BYTE_TRACK_MAX_AGE      = 30    # 約1秒相当（30fpsの場合）


# TrackID別ランキング設定

# 何フレームごとにランキングログを出すか（例：30なら約1秒ごと @30fps）
AUTO_TRACK_RANK_INTERVAL_FRAMES = 60

# 類似度ランキングの上位何件を出すか（Face/App/Gait 共通）
AUTO_TRACK_RANK_TOPN = 3

# 0.0〜1.0 の範囲、小数第2位まで使用
AUTO_FACE_THRESH_DEFAULT = 0.20000  # 顔閾値
AUTO_APP_THRESH_DEFAULT  = 0.80000  # 外見閾値
AUTO_GAIT_THRESH_DEFAULT = 0.99700  # 歩容閾値

# ランキングモード
# "face" : 顔 > 外見 > 歩容 の優先度で、ランキングに使うモダリティを決定
AUTO_RANK_MODE: str = "face"

# HTMLレポートとキャプチャ画像の出力先
AUTO_REPORT_DIR: str = "data/evidence/ranking"

# HTMLに Face/App/Gait の類似度を表示するか（False なら画像だけ）
AUTO_REPORT_SHOW_SIM: bool = True

# YOLO Face
YOLO_FACE_MODEL_PATH = "models/yolov8s-face-lindevs.pt"
YOLO_FACE_CONF = 0.08
YOLO_FACE_IOU = 0.6
YOLO_FACE_IMGSZ = 640
YOLO_FACE_MAX_DET = 200

# YOLO Pose
# ONNX 版 YOLOv8-Pose のモデルパスと入出力設定
YOLO_POSE_MODEL_PATH = "models/yolov8l-pose.onnx"   # 例: lサイズ
YOLO_POSE_IMGSZ      = 640                           # 入力サイズ（正方）
YOLO_POSE_CONF       = 0.25                          # 最小スコア
YOLO_POSE_IOU        = 0.50                          # NMS IoU
YOLO_POSE_MAX_DET    = 300

# --- 骨格の描画・フィルタ ---
# 骨格点の最小信頼度（この値未満は点・線を描かない/採用しない）
POSE_KPT_CONF_MIN = 0.35
# 欠落点が多すぎる・足首/顔耳が出ていない等を容赦なく除外するハードフィルタ
POSE_HARD_FILTER_ENABLED = True
# 骨格描画ON/OFF（外見だけのときはOFFにすると軽い）
POSE_DRAW_ENABLED = True
POSE_DRAW_LANDMARK_BGR   = (0, 255, 0)    # 点色
POSE_DRAW_CONNECTION_BGR = (255, 255, 255)    # 線色
POSE_DRAW_THICKNESS      = 2
POSE_DRAW_RADIUS         = 2

YOLO_INPUT_SIZE = 640        # 入力サイズ（SxS）。ONNX出力のスケール復元に使う
DET_CONF_MIN    = 0.25       # 検出スコアの閾値（低すぎると誤検出↑、高すぎると見逃し↑）
NMS_IOU_TH      = 0.50       # NMSのIoUしきい値
MAX_DETS        = 300        # 1フレームの最大検出数
GAIT_DRAW_BBOX  = True       # 歩容モード：骨格に加えてBBOXも重畳表示するか

# 体の部位インデックス（COCO-17）
# 0:nose 1:leye 2:reye 3:lear 4:rear 5:lshoulder 6:rshoulder 7:lelbow 8:relbow
# 9:lwrist 10:rwrist 11:lhip 12:rhip 13:lknee 14:rknee 15:lankle 16:rankle
FACE_IDXS  = (0, 1, 2)       # nose, leye, reye
EAR_IDXS   = (3, 4)          # lear, rear
ANKLE_IDXS = (15, 16)        # lankle, rankle

# MediaPipe Pose

# モデルファイルパス（.task）
MEDIAPIPE_POSE_TASK_PATH: str = "models/pose_landmarker_full.task"

# 小さすぎる人物をスキップする最小BBOX高さ（px）
MEDIAPIPE_MIN_BBOX_H =  120

# BBOXの周囲に足す余白倍率（>1.0）
MEDIAPIPE_CROP_PAD: float = 1.25

# 33点 → COCO17点 対応（COCO順: nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_an, r_an）
MEDIAPIPE_TO_COCO17 = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

# COCO17 スケルトン接続（デバッグ描画用）
COCO17_SKELETON_PAIRS = (
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 12),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (0, 1), (0, 2),
        (1, 3), (2, 4),
    )

# デバッグ描画色・太さ・点サイズ（BGR）
MEDIAPIPE_DEBUG_DRAW_COLOR_BGR = (255, 0, 255)
MEDIAPIPE_DEBUG_DRAW_THICKNESS = 2
MEDIAPIPE_DEBUG_DRAW_POINT_RADIUS = 3

# AdaFace
ADAFACE_CKPT_PATH = "models/adaface_ir101_webface12m.ckpt"
ADAFACE_ARCH = "ir_101"
ADAFACE_MARGIN = 0.20
ADAFACE_OUT_SIZE = 112
ADAFACE_LOG_EVERY = 10

# OSNet
OSNET_MODEL_PATH      = "models/osnet_x1_0.onnx"  # ReID用ONNX
OSNET_INPUT_SIZE_HW   = (256, 128)                # (H, W) ReIDの標準解像度
REID_TTA_HFLIP        = True                     # 左右反転TTA（精度↑、負荷↑）
REID_BATCH_SIZE       = 16

# クロップの短辺がこの値を下回る場合、等倍率で底上げしてからレターボックス
OSNET_MIN_CROP_SIDE: int = 48

# レターボックス時のパディング色（BGR）
OSNET_PAD_VALUE: tuple[int, int, int] = (114, 114, 114)

# GaitMixer
GAIT_MIXER_MODEL_PATH = "models/GaitMixer.pt"


# GaitMixer用 TTA の有効/無効
TTA_FLIP_ENABLED: bool  = True   # 左右反転TTA（flip_tta_embed）のON/OFF
TTA_NOISE_ENABLED: bool = True  # ノイズTTA（noise_tta_embed）のON/OFF
TTA_TEMPORAL_ENABLED: bool  = True  # 時間反転TTAのON/OFF

# GaitMixer用 左右反転TTA
# COCO 17の左右対応
COCO_LR_PAIRS: list[tuple[int, int]] = [
    (5, 6),   # shoulder
    (7, 8),   # elbow
    (9, 10),  # wrist
    (11, 12), # hip
    (13, 14), # knee
    (15, 16), # ankle
]

# GaitMixer用 ユーティリティ
# --- 骨格正規化（方法B）---
NORM_CENTER_HIPS: tuple[int, int]        = (11, 12)      # 左右ヒップ
NORM_SCALE_SHOULDERS: tuple[int, int]    = (5, 6)        # 左右ショルダー
NORM_HEIGHT_TOP_IDX: int                 = 0             # nose
NORM_HEIGHT_BOTTOM_IDXS: tuple[int, int] = (15, 16)      # 左右ankle
NORM_MIN_SCALE: float                    = 1e-6          # スケール下限

# 時系列の安定化（EMA）
NORM_EWMA_ENABLED: bool                  = True          # 有効/無効
NORM_EWMA_ALPHA: float                   = 0.30           # 0<α<1（大きいほど最新重視）0.85
NORM_EWMA_APPLY_TO_CENTER: bool          = True          # centerへEMA適用
NORM_EWMA_APPLY_TO_SCALE: bool           = True          # scaleへEMA適用

# キーポイント有効判定（0近傍や無限大の除外）
KPT_VALID_MIN: float                     = 1e-6

FLIP_TTA_L2_NORMALIZE: bool              = False

# --- ノイズTTA ---
NOISE_TTA_DEFAULT_SIGMA: float           = 0.02        # 正規化座標系の標準偏差
NOISE_TTA_DEFAULT_SAMPLES: int           = 4             # ノイズサンプル数
NOISE_TTA_INCLUDE_BASE: bool             = True          # 元系列も平均に含める
NOISE_TTA_L2_NORMALIZE: bool             = False

# --- 時間反転TTA ---
TTA_TEMPORAL_L2_NORMALIZE: bool = False

# --- 骨格品質 ---
POSE_GATE_ENABLED: bool = True

# 有効判定に使う最小信頼度（YOLO-Poseのkpt conf）。描画設定と揃えてOK
POSE_GATE_MIN_CONF: float = POSE_KPT_CONF_MIN

# 下半身点 (11,12,13,14,15,16) のうち有効な点の最小合計
POSE_GATE_LOWER_MIN: int = 4

# 左右の “半身” ごとに必要な最小有効点
# LEFT=[5,7,9,11,13,15], RIGHT=[6,8,10,12,14,16]
POSE_GATE_SIDE_MIN: int = 3

# --- TTAハイブリッド・スコアリング設定 ---
# ハイブリッド有効/無効（α=0.0でも無効相当）
HYBRID_TTA_ENABLED: bool = False

# ハイブリッドの係数（最終スコア = α*concat_cos + (1-α)*TopK平均）
# 0.0 → 従来のTopK平均のみ / 0.5 → 推奨初期値 / 0.6〜0.7 → 攻め
HYBRID_TTA_ALPHA: float = 1

# TTAで使うモード（順序が「対応付け」に使われます）
# "normal", "flip", "rev", "rev_flip" の中から選択
TTA_MODES = ["normal", "flip", "rev", "rev_flip"]  # 3本=384D。4本にしたいときは "rev_flip" を追加

# Top-K 平均のK
TTA_SCORE_TOPK: int = 2

YAW_MIN_SHOULDER_PX: int = 30         # 肩幅が小さい時は未確定扱い
YAW_MIN_VIS: float = 0.60             # 肩のvisibility平均が低い時は未確定扱い

# BBOX→切り出し時の外枠パディング（枠をわずかに広げて上下端欠けを防ぐ）
REID_CROP_PAD_RATIO   = 0.05

# ギャラリー保存先（外見用）
REID_GALLERY_DIR      = "data/gallery/appearance"

# 検索Top-Kとセンロイド検索の有効/無効
REID_TOPK             = 5
REID_USE_CENTROID     = True  # ラベルごとに平均ベクトルで検索（安定・高速）

# REID関連の設定の近くに追記
REID_TTA_HFLIP = False
REID_BATCH_SIZE = 1

MIN_CROP_SIDE = 48  

# 間引きフレーム数と同時処理BBOX上限
REID_EVERY_N = 1          # 3フレームに1回だけ識別を回す
REID_MAX_BOXES = 12       # 1フレームで識別する人数の上限

WINDOW_T: int = 32
GAIT_GALLERY_DIR      = "data/gallery/gait"

AUTO_GAIT_YAW_ENABLED = False

# Auto mode
AUTO_MODE_NAME = "マルチモーダル"
AUTO_FACE_FRAMES = 3      # 顔：最初に積むフレーム数
AUTO_APP_FRAMES  = 5     # 外見：最初に積むフレーム数
AUTO_GAIT_FRAMES = 32     # 歩容：最初に積むフレーム数

# === AUTO (顔+外見+歩容) 共通保存先 / 連番仕様 ===
AUTO_GALLERY_DIR   = os.path.join("data", "gallery", "auto")  # 既定: data/gallery/auto
AUTO_FACE_JSON     = os.path.join(AUTO_GALLERY_DIR, "face_gallery.json")
AUTO_APP_JSON      = os.path.join(AUTO_GALLERY_DIR, "app_gallery.json")
AUTO_GAIT_JSON     = os.path.join(AUTO_GALLERY_DIR, "gait_gallery.json")

# ベース名 + "-" + PREFIX + 連番 の形式で採番（例: "Mizuta-P0001"）
AUTO_SERIAL_PREFIX = "P"
AUTO_SERIAL_WIDTH  = 4