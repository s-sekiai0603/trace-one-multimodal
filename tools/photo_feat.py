# -*- coding: utf-8 -*-
"""
photo_feat.py
- data/new_id_photo/ 配下の静止画から顔特徴を一括抽出し、
  data/gallery/auto/face_gallery.json にオートモード互換形式で追記するツール。
- 特徴抽出には YOLOFaceDetector + AdaFaceEmbedder を使用。
- ラベルは「ファイル名（拡張子なし）-PXXXX」形式で自動採番。
- 正常に保存できた画像は data/old_id_photo/ に移動する。
"""

from __future__ import annotations

from typing import Optional, Iterable
import os
import sys
import json
import re
import shutil
import logging
from pathlib import Path

import numpy as np
import cv2

# プロジェクト内モジュール（相対インポート）
from ..config import (
    AUTO_FACE_JSON,
    AUTO_APP_JSON,
    AUTO_GAIT_JSON,
    AUTO_SERIAL_PREFIX,
    AUTO_SERIAL_WIDTH,
)
from ..detectors.yolo_face import YOLOFaceDetector
from ..embedders.ada_face import AdaFaceEmbedder
from ..utils import ada_face_utils  # 将来ズーム等を使いたくなった時用に読み込み

log = logging.getLogger("app.photo_feat")


# ------------------------------------------------------------
# パス系ユーティリティ
# ------------------------------------------------------------

def _project_root() -> str:
    """
    src/tools/photo_feat.py → プロジェクトルート(../..) を返す。
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _resolve_project_path(rel_path: str) -> str:
    """
    config で相対パス指定されている場合に、プロジェクトルート起点の絶対パスに解決する。
    """
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_project_root(), rel_path)


# 入出力ディレクトリ
NEW_ID_DIR = os.path.join(_project_root(), "data", "new_id_photo")
OLD_ID_DIR = os.path.join(_project_root(), "data", "old_id_photo")

AUTO_FACE_JSON_ABS = _resolve_project_path(AUTO_FACE_JSON)
AUTO_APP_JSON_ABS = _resolve_project_path(AUTO_APP_JSON)
AUTO_GAIT_JSON_ABS = _resolve_project_path(AUTO_GAIT_JSON)


# ------------------------------------------------------------
# ラベル採番（AutoController と同等ロジックを独立実装）
# ------------------------------------------------------------

def _calc_next_serial_for_base(base: str) -> int:
    """
    base: "Mizuta" のようなベース名
    既存の auto/ 各 JSON を横断して、 base-<PREFIX><NNNN> の N の最大値+1 を返す。
    見つからなければ 1。
    AutoController._calc_next_serial_for_base と同等の振る舞い。
    """
    base = (base or "").strip()
    if not base:
        return 1

    max_n = 0
    pat = re.compile(
        rf"^{re.escape(base)}-{re.escape(AUTO_SERIAL_PREFIX)}(\d+)$",
        re.IGNORECASE,
    )

    def _scan_json(path: str) -> None:
        nonlocal max_n
        if not path or not os.path.isfile(path):
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
            it: list = []
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
            val: Optional[str] = None
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
            if not m:
                continue
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except Exception:
                pass

    for p in (AUTO_FACE_JSON_ABS, AUTO_APP_JSON_ABS, AUTO_GAIT_JSON_ABS):
        _scan_json(p)

    return (max_n + 1) if max_n > 0 else 1


def alloc_full_label_from_filename(stem: str) -> Optional[str]:
    """
    ファイル名の stem から Auto 用のフルラベル "Stem-P0001" のような形式を生成する。
    """
    base = (stem or "").strip()
    if not base:
        return None
    n = _calc_next_serial_for_base(base)
    full = f"{base}-{AUTO_SERIAL_PREFIX}{n:0{AUTO_SERIAL_WIDTH}d}"
    return full


# ------------------------------------------------------------
# face_gallery.json への追記（AutoController._append_face_pose_json 相当）
# ------------------------------------------------------------

def append_face_pose_json(
    path: str,
    label: str,
    pose: str,
    vec: Optional[np.ndarray],
) -> None:
    """
    マルチモーダルモードの顔ギャラリー用:
    - label_to_vecs は従来通り更新
    - 追加で records に {label, pose, vec} を追記
    AutoController._append_face_pose_json と同じ振る舞いを単体関数化したもの。
    """
    if vec is None:
        return

    # 既存データ読み込み
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    label = str(label or "").strip()
    pose = str(pose or "").strip()
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

    recs.append(
        {
            "label": label,
            "pose": pose,
            "vec": v,
        }
    )

    data["dim"] = len(v)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(
        "[PHOTO_FEAT] wrote: label=%s pose=%s dim=%d -> %s",
        label,
        pose,
        len(v),
        path,
    )


# ------------------------------------------------------------
# 画像1枚の処理
# ------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def process_one_image(
    img_path: Path,
    face_detector: YOLOFaceDetector,
    embedder: AdaFaceEmbedder,
) -> bool:
    """
    1枚の画像から顔特徴を抽出し、face_gallery.json に追記する。
    成功したら True, 失敗したら False。
    """
    img_path = img_path.resolve()
    log.info("[PHOTO_FEAT] processing: %s", img_path)

    # 画像読み込み
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None or img_bgr.size == 0:
        log.warning("[PHOTO_FEAT] SKIP(read failed): %s", img_path.name)
        return False

    H, W = img_bgr.shape[:2]

    # 顔検出（オートモードと同じ YOLOFaceDetector を利用）
    boxes_xyxy, scores = face_detector.detect(img_bgr)

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        log.warning("[PHOTO_FEAT] SKIP(no face): %s", img_path.name)
        return False

    if len(boxes_xyxy) > 1:
        log.warning(
            "[PHOTO_FEAT] SKIP(multi face %d): %s", len(boxes_xyxy), img_path.name
        )
        return False

    # 1人だけ想定なので先頭1個だけ使用
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)[:1, :]

    # 将来、ada_face_utils.inflate_boxes_xyxy でズームしたくなったらここで使う。
    # 今はオートモードと特徴がズレないように、BBOXはそのまま AdaFace に渡す。
    # 例: boxes = ada_face_utils.inflate_boxes_xyxy(boxes, (H, W), margin=0.0)
    #    （margin=0.0なら無変化）

    # AdaFace で埋め込み抽出（オートモードと同じ embed_many を利用）
    feats = embedder.embed_many(img_bgr, boxes)  # (1, D) 想定

    if feats is None or feats.size == 0:
        log.warning("[PHOTO_FEAT] SKIP(adaface empty feat): %s", img_path.name)
        return False

    feat = np.asarray(feats[0], dtype=np.float32).reshape(-1)
    if not np.any(feat):
        # embed_one が失敗した場合は 0 ベクトルになる仕様
        log.warning("[PHOTO_FEAT] SKIP(adaface zero vec): %s", img_path.name)
        return False

    # ラベル採番（ファイル名 stem から "Stem-P0001" のように作成）
    stem = img_path.stem
    full_label = alloc_full_label_from_filename(stem)
    if not full_label:
        log.warning("[PHOTO_FEAT] SKIP(empty label base): %s", img_path.name)
        return False

    # ID写真は基本的に正面想定なので pose="front" として保存
    pose = "front"

    # face_gallery.json に追記（AutoController と同じフォーマット）
    try:
        append_face_pose_json(AUTO_FACE_JSON_ABS, full_label, pose, feat)
    except Exception as e:
        log.error(
            "[PHOTO_FEAT] ERROR(write json): %s (%s)", img_path.name, str(e)
        )
        return False

    log.info(
        "[PHOTO_FEAT] OK: %s -> label=%s pose=%s dim=%d",
        img_path.name,
        full_label,
        pose,
        feat.shape[0],
    )
    return True


# ------------------------------------------------------------
# バッチ処理メイン
# ------------------------------------------------------------

def iter_new_id_images(input_dir: Optional[str] = None) -> Iterable[Path]:
    """
    data/new_id_photo/ 配下の画像ファイル(Path)を列挙する。
    """
    dir_path = Path(input_dir or NEW_ID_DIR)
    if not dir_path.exists():
        log.warning("[PHOTO_FEAT] input dir not found: %s", dir_path)
        return []
    return sorted([p for p in dir_path.iterdir() if _is_image_file(p)])


def run_batch(input_dir: Optional[str] = None) -> None:
    """
    new_id_photo 配下の全画像に対して顔特徴抽出を行い、
    成功した画像は old_id_photo に移動する。
    """
    in_dir = Path(input_dir or NEW_ID_DIR)
    out_dir = Path(OLD_ID_DIR)

    os.makedirs(out_dir, exist_ok=True)

    files = list(iter_new_id_images(str(in_dir)))
    total = len(files)
    if total == 0:
        log.info("[PHOTO_FEAT] no image files in: %s", in_dir)
        return

    log.info("[PHOTO_FEAT] start: dir=%s total=%d", in_dir, total)

    # モデルはバッチ処理中に1回だけ生成
    face_detector = YOLOFaceDetector()
    embedder = AdaFaceEmbedder()

    n_ok = 0
    n_fail = 0

    for p in files:
        ok = False
        try:
            ok = process_one_image(p, face_detector, embedder)
        except Exception as e:
            log.error("[PHOTO_FEAT] ERROR(process): %s (%s)", p.name, str(e))
            ok = False

        if ok:
            # 成功した画像のみ old_id_photo に移動
            dst = out_dir / p.name
            try:
                shutil.move(str(p), str(dst))
                log.info("[PHOTO_FEAT] moved: %s -> %s", p.name, dst)
            except Exception as e:
                log.error(
                    "[PHOTO_FEAT] ERROR(move): %s -> %s (%s)",
                    p,
                    dst,
                    str(e),
                )
            n_ok += 1
        else:
            n_fail += 1

    log.info(
        "[PHOTO_FEAT] done: success=%d failed=%d total=%d",
        n_ok,
        n_fail,
        total,
    )


# ------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------

def _setup_basic_logging() -> None:
    """
    スクリプト単体実行用の簡易ログ設定。
    アプリ本体から import される場合は、既存の logging 設定が使われる。
    """
    if logging.getLogger().handlers:
        # すでに設定済みなら何もしない
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _setup_basic_logging()

    # 第1引数に input_dir を指定可能（省略時は data/new_id_photo）
    in_dir_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    run_batch(in_dir_arg)
