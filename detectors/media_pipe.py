# media_pipe.py
# MediaPipe Pose (pose_landmarker_full.task) を使って、
# 1) 単一人物の骨格推定
# 2) YOLOの全BBOXに対する骨格推定
# 3) 33点 → COCO17点 への変換
# 4) デバッグ描画（ピンク：33⇒17変換後の骨格）※後で削除可
# 5) モデル(.task)のロード
#
# 使い方イメージ:
#   mp = MediaPipePose()
#   kpts17_list = mp.infer_all_and_draw(frame_bgr, bboxes_xyxy)  # 17点を返しつつフレームへ描画
#   # bboxes_xyxy: np.ndarray shape=(N,4) [x1,y1,x2,y2]

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import math

from ..config import YAW_MIN_VIS, YAW_MIN_SHOULDER_PX, MEDIAPIPE_POSE_TASK_PATH, MEDIAPIPE_MIN_BBOX_H, MEDIAPIPE_CROP_PAD, MEDIAPIPE_TO_COCO17, COCO17_SKELETON_PAIRS, MEDIAPIPE_DEBUG_DRAW_COLOR_BGR, MEDIAPIPE_DEBUG_DRAW_THICKNESS, MEDIAPIPE_DEBUG_DRAW_POINT_RADIUS

# ========= config から設定を読みつつ、無ければ安全なデフォルトにフォールバック =========
try:
    import config as _cfg
except Exception:
    _cfg = None

def _cfg_get(name: str, default):
    return getattr(_cfg, name, default) if _cfg is not None else default

# =============== MediaPipe Tasks: PoseLandmarker を model_asset_buffer でロード ===============
# 非ASCIIパスでも安定するよう、.task をバイトで読み込んで渡す方式を採用
BaseOptions        = mp.tasks.BaseOptions
VisionRunningMode  = mp.tasks.vision.RunningMode
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions

# 画像コンテナは Tasks 直下ではなく、mediapipeのトップの mp.Image を使用するのが最も安定
MPImage            = mp.Image
MPImageFormat      = mp.ImageFormat


@dataclass
class MPResult:
    """
    MediaPipe Pose の出力を格納:
      - kpts33_xyv: (33,4) [x,y,z,visibility] 画素座標（zは相対値）
      - kpts17_xyv: (17,4) [x,y,z,visibility] 画素座標（zは相対値）
    """
    kpts33_xyv: np.ndarray
    kpts17_xyv: np.ndarray


class MediaPipePose:
    """
    YOLOの boxes（xyxy, 画像座標）を受けて、
    各BBOXごとに MediaPipe PoseLandmarker.detect() を実行し、
    33点を COCO17 にマップ、必要なら17点スケルトンをピンクで重ね描画する。
    """

    def __init__(self,
                 task_path: Optional[str] = None,
                 min_bbox_h: int = MEDIAPIPE_MIN_BBOX_H,
                 crop_pad: float = MEDIAPIPE_CROP_PAD):
        """
        Args:
            task_path: pose_landmarker_full.task のパス（省略時は config / 既定値）
            min_bbox_h: 小さすぎるBBOXをスキップする閾値（px）
            crop_pad:   BBOXを拡げる倍率（>1.0）
        """
        self.task_path = str(task_path or MEDIAPIPE_POSE_TASK_PATH)
        self.min_bbox_h = int(min_bbox_h)
        self.crop_pad = float(crop_pad)
        self._pose = mp_pose.Pose(
            static_image_mode=True,         # クロップ画像ごとに単発で処理
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker: Any = None
        self._load_landmarker()

    # 5) モデルをロードするメソッド
    def _load_landmarker(self) -> None:
        model_path = Path(self.task_path).resolve()
        if not model_path.exists() or model_path.stat().st_size <= 0:
            raise FileNotFoundError(f"[MediaPipePose] model not found or empty: {model_path}")

        with open(model_path, "rb") as f:
            model_bytes = f.read()
        if not model_bytes:
            raise RuntimeError(f"[MediaPipePose] failed to read model bytes: {model_path}")

        base_options = BaseOptions(model_asset_buffer=model_bytes)
        options = PoseLandmarkerOpts(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,   # 1枚画像（BBOXクロップ）ごとに判定
            num_poses=1,
            output_segmentation_masks=False,        # 軽量化。必要なら True に
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    # 3) 33点 → COCO17点 へ変換するメソッド
    def _mp33_to_coco17_xyv(self, mp_landmarks, bbox_xyxy, img_wh):
        """
        mp_landmarks: MediaPipe 33点の list/iterable（lm.x,lm.y,visibility を持つ）
        bbox_xyxy:    (x1,y1,x2,y2) 画像座標
        img_wh:       (W,H)  ※使わなくても良いが引数は残す
        戻り:         (17,3) [x,y,v] （画像座標）
        """
        import numpy as np
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

        kpts = np.zeros((17, 3), np.float32)
        for i_coco, i_mp in enumerate(MEDIAPIPE_TO_COCO17):
            lm = mp_landmarks[i_mp]
            px = x1 + float(getattr(lm, "x", 0.0)) * bw
            py = y1 + float(getattr(lm, "y", 0.0)) * bh
            v  = float(getattr(lm, "visibility", 1.0))
            kpts[i_coco] = (px, py, v)
        return kpts
    
    def map_33_to_coco17(self, k33, *, fill_visibility: float = 1.0):
        """
        画像座標の 33点配列 -> COCO17 (x,y,v) に並べ替え
        入力:  k33 (33, D) D=2/3/4（例: [x,y,v] or [x,y,z,v]）
        戻り:  (17, 3) [x, y, v]
        """
        import numpy as np
        a = np.asarray(k33, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] < 29:
            raise ValueError(f"k33 shape must be (33, D). got {a.shape}")

        idx = np.asarray(MEDIAPIPE_TO_COCO17, dtype=np.int64)

        if a.shape[1] >= 4:
            xy = a[idx, :2]
            v  = a[idx, 3:4]              # 列3をvisibilityとして採用
        elif a.shape[1] == 3:
            xy = a[idx, :2]
            v  = a[idx, 2:3]              # 列2をvisibilityとして採用
        elif a.shape[1] == 2:
            xy = a[idx, :2]
            v  = np.full((len(idx), 1), float(fill_visibility), dtype=np.float32)
        else:
            raise ValueError(f"unsupported last-dim D={a.shape[1]}")

        return np.concatenate([xy, v], axis=1)

    # 1) 単一人物の骨格推定メソッド（BBOX=xyxy）
    def infer_person(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[MPResult]:
        """
        Args:
            frame_bgr: (H,W,3) BGR uint8
            bbox_xyxy: (4,) [x1,y1,x2,y2]
        Returns:
            MPResult or None（失敗時）
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None
        if bbox_xyxy is None or len(bbox_xyxy) < 4:
            return None

        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        if (y2 - y1) < self.min_bbox_h:
            return None

        # パディング付きでなるべく正方形に近いクロップを作成
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1) * self.crop_pad
        bh = (y2 - y1) * self.crop_pad
        side = max(bw, bh)

        left   = max(0, int(round(cx - side / 2)))
        top    = max(0, int(round(cy - side / 2)))
        right  = min(W, int(round(cx + side / 2)))
        bottom = min(H, int(round(cy + side / 2)))
        if right - left < 2 or bottom - top < 2:
            return None

        crop_bgr = frame_bgr[top:bottom, left:right]
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        # BGR→RGB, mp.Image へ
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_img   = MPImage(image_format=MPImageFormat.SRGB, data=crop_rgb)

        # 推定
        try:
            res = self._landmarker.detect(mp_img) if self._landmarker is not None else None
        except Exception:
            # モデル初期化に失敗 or detectエラー
            return None
        if res is None or not res.pose_landmarks or len(res.pose_landmarks) == 0:
            return None

        # 33点（NormalizedLandmarkList）→ 元画像の画素座標へ展開
        lms = res.pose_landmarks[0]
        k33 = np.zeros((33, 4), np.float32)  # x,y,z,visibility
        cw, ch = float(right - left), float(bottom - top)
        for i, lm in enumerate(lms):
            # lm.x/lm.y は [0,1] のcrop内正規化座標
            x = float(lm.x) * cw + left
            y = float(lm.y) * ch + top
            # z は相対奥行き（MediaPipe規定の相対値: 画素ではない）
            z = float(getattr(lm, "z", 0.0))
            v = float(getattr(lm, "visibility", 0.0))
            # 画面内にクリップ
            x = np.clip(x, 0.0, W - 1.0)
            y = np.clip(y, 0.0, H - 1.0)
            k33[i] = (x, y, z, v)

        k17 = self.map_33_to_coco17(k33)
        return MPResult(k33, k17)

    # 2) 全BBOXに対して骨格推定を行うメソッド
    def infer_all(self, frame_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> List[MPResult]:
        """
        Args:
            frame_bgr: (H,W,3) BGR
            bboxes_xyxy: (N,4) or list of [x1,y1,x2,y2]
        Returns:
            List[MPResult]（失敗BBOXはスキップ）
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        if bboxes_xyxy is None:
            return []
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 4 or boxes.shape[0] == 0:
            return []

        out: List[MPResult] = []
        for i in range(boxes.shape[0]):
            r = self.infer_person(frame_bgr, boxes[i, :4])
            if r is not None:
                out.append(r)
        return out

    # 4) 17点スケルトンをピンクで重ね描画（デバッグ用; 後で削除可）
    def draw_coco17(self,
                    frame_bgr: np.ndarray,
                    kpts17_xyv: np.ndarray,
                    color: Tuple[int, int, int] = MEDIAPIPE_DEBUG_DRAW_COLOR_BGR,
                    thickness: int = MEDIAPIPE_DEBUG_DRAW_THICKNESS,
                    radius: int = MEDIAPIPE_DEBUG_DRAW_POINT_RADIUS) -> None:
        """
        Args:
            frame_bgr: 画像（インプレースで上書き）
            kpts17_xyv: (17,4) [x,y,z,visibility]
        """
        if frame_bgr is None or kpts17_xyv is None or kpts17_xyv.shape[0] != 17:
            return

        H, W = frame_bgr.shape[:2]
        # 線（骨）
        for a, b in COCO17_SKELETON_PAIRS:
            xa, ya = int(kpts17_xyv[a, 0]), int(kpts17_xyv[a, 1])
            xb, yb = int(kpts17_xyv[b, 0]), int(kpts17_xyv[b, 1])
            if 0 <= xa < W and 0 <= ya < H and 0 <= xb < W and 0 <= yb < H:
                cv2.line(frame_bgr, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)

        # 点（関節）
        for i in range(17):
            x, y = int(kpts17_xyv[i, 0]), int(kpts17_xyv[i, 1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame_bgr, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    # 便利：全BBOXに推定→（任意で）描画まで一気に
    def infer_all_and_draw(self, frame_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> List[np.ndarray]:
        """
        33→17変換まで行い、17点を描画して返します（描画はピンク）。
        Returns:
            List[np.ndarray]  各要素は shape=(17,4) の [x,y,z,visibility]
        """
        results = self.infer_all(frame_bgr, bboxes_xyxy)
        out_17: List[np.ndarray] = []
        for r in results:
            self.draw_coco17(frame_bgr, r.kpts17_xyv,
                             color=MEDIAPIPE_DEBUG_DRAW_COLOR_BGR,
                             thickness=MEDIAPIPE_DEBUG_DRAW_THICKNESS,
                             radius=MEDIAPIPE_DEBUG_DRAW_POINT_RADIUS)
            out_17.append(r.kpts17_xyv)
        return out_17
    
    def estimate_yaw_deg_from_world(self, mp_world_landmarks):
        """
        mp_world_landmarks: 33個の world座標 (x,y,z) [メートル系]
        戻り: yaw_deg（右+ / 左- の斜め向き角）
        """
        import math
        L = mp_world_landmarks[11]  # left_shoulder
        R = mp_world_landmarks[12]  # right_shoulder
        dx = float(R.x) - float(L.x)
        dz = float(R.z) - float(L.z)
        # x-z 平面で肩線の傾きからYaw推定
        return math.degrees(math.atan2(dz, dx))
    
    def _calc_yaw_deg(self, mp_world_landmarks):
        """
        互換用のヘルパー。
        3D worldランドマークから yaw[deg] を返す。
        estimate_yaw_for_boxes() から呼ばれることを想定。
        """
        return self.estimate_yaw_deg_from_world(mp_world_landmarks)

    def _calc_yaw_deg_2d(self, lms_2d):
        """
        2Dランドマークしか無い場合の簡易 yaw 推定。

        厳密な角度は取りづらいので、
        肩の左右位置から「側面かどうか」だけをざっくり判定し、
        ±90° 相当の値を返す。
        （_yaw_to_view 側では |yaw| だけ見て front / side / back を決める想定）
        """
        import math

        if lms_2d is None or len(lms_2d) <= 12:
            # 肩が取れていない → 正面扱い
            return 0.0

        L = lms_2d[11]  # left_shoulder
        R = lms_2d[12]  # right_shoulder

        dx = float(R.x) - float(L.x)

        if abs(dx) < 1e-5:
            # 肩がほぼ左右対称 → 正面扱い
            return 0.0

        # とりあえず「横向き」として ±90° に寄せる
        yaw_deg = 90.0
        return math.copysign(yaw_deg, dx)

    def estimate_yaw_for_boxes(self, frame_bgr, boxes):
        """
        各 BBOX について yaw[deg] を返す。
        顔ランドマーク（耳以外）が見えていない場合は「背面」とみなして 180° を返す。
        """
        # 耳以外の「顔パーツ」ランドマークインデックス
        # NOSE, EYE, MOUTH など（耳: LEFT_EAR=7, RIGHT_EAR=8 は除外）
        FACE_LM_IDXS = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_EYE_INNER,
            mp_pose.PoseLandmark.LEFT_EYE,
            mp_pose.PoseLandmark.LEFT_EYE_OUTER,
            mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
            mp_pose.PoseLandmark.MOUTH_LEFT,
            mp_pose.PoseLandmark.MOUTH_RIGHT,
        ]
        VIS_THR = 0.5  # 可視性のしきい値（必要なら調整）

        yaws = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            result = self._pose.process(crop_rgb)
            if not result.pose_landmarks:
                # そもそも Pose が出ていない → yaw なし（必要なら 180.0 にしてもOK）
                yaws.append(None)
                continue

            lms_2d = result.pose_landmarks.landmark
            lms_3d = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None

            # ① 顔ランドマーク（耳以外）の可視性チェック
            face_visible = False
            for plm in FACE_LM_IDXS:
                idx = int(plm.value)
                if idx < len(lms_2d):
                    if lms_2d[idx].visibility >= VIS_THR:
                        face_visible = True
                        break

            if not face_visible:
                # 顔パーツがほぼ見えていない → 背面とみなす
                yaws.append(180.0)   # ★ ここがポイント：_yaw_to_view で必ず "back" になる
                continue

            # ② 通常どおり yaw を計算
            if lms_3d is None:
                # 3D がない場合は 2D からの近似でもOK（今の実装に合わせる）
                yaw_deg = self._calc_yaw_deg_2d(lms_2d)
            else:
                yaw_deg = self._calc_yaw_deg(lms_3d)

            yaws.append(float(yaw_deg))

        return yaws

__all__ = ["MediaPipePose", "MPResult"]