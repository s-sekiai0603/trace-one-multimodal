# -*- coding: utf-8 -*-
from __future__ import annotations

"""
feat_fuse.py

Auto モード用ギャラリーの「特徴統合」ユーティリティ。

役割:
- data/gallery/auto/{face_gallery.json, app_gallery.json, gait_gallery.json}
  に保存された特徴を読み込み、
  - 顔   : ラベル > 向き(pose) > {rep, mean}
  - 歩容 : ラベル > 向き(dir8) > {rep, mean}
  - 外見 : ラベル > 統合前ラベル > [生ベクトル...]
  という構造に集約した「統合ギャラリー」を作る。

保存先（デフォルト）:
- data/gallery/auto/face_fused_gallery.json
- data/gallery/auto/app_fused_gallery.json
- data/gallery/auto/gait_fused_gallery.json

UI 側からは、例えば:
- path, fused_label = fuse_face_auto(["Sasakura-P0001", "Sasakura-P0004"], "Sasakura-fused")
- path, fused_label = fuse_app_auto([...], "Sasakura-fused")
- path, fused_label = fuse_gait_auto([...], "Sasakura-fused")
- paths = fuse_all_auto([...], "Sasakura-fused")

のように呼び出す想定。
"""

from typing import List, Dict, Iterable, Optional, Tuple
import os
import json
import time
import re  # ★追加: 連番ラベル用

import numpy as np


# ===================== パス系ユーティリティ =====================

def _project_root() -> str:
    """
    src/db/feat_fuse.py → ../../ をプロジェクトルートとみなす。
    feature.py と同じロジック。
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _auto_gallery_dir() -> str:
    return os.path.join(_project_root(), "data", "gallery", "auto")


def _auto_face_json() -> str:
    return os.path.join(_auto_gallery_dir(), "face_gallery.json")


def _auto_app_json() -> str:
    return os.path.join(_auto_gallery_dir(), "app_gallery.json")


def _auto_gait_json() -> str:
    return os.path.join(_auto_gallery_dir(), "gait_gallery.json")


def _face_fused_json() -> str:
    return os.path.join(_auto_gallery_dir(), "face_fused_gallery.json")


def _app_fused_json() -> str:
    return os.path.join(_auto_gallery_dir(), "app_fused_gallery.json")


def _gait_fused_json() -> str:
    return os.path.join(_auto_gallery_dir(), "gait_fused_gallery.json")


# ===================== 共通ユーティリティ =====================

def _load_json(path: str, default: object) -> object:
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_np_vec(vec_like) -> np.ndarray:
    v = np.asarray(vec_like, dtype=np.float32).reshape(-1)
    return v


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a, b は 1次元ベクトル想定
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _select_rep_and_mean(vecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    ベクトル群から
    - mean: L2 正規化した平均ベクトル
    - rep : mean に最も近い（コサイン類似度が最大）ベクトル
    を返す。
    """
    if not vecs:
        raise ValueError("vecs is empty")

    arr = np.stack(vecs, axis=0).astype(np.float32)
    mean = _l2_normalize(arr.mean(axis=0))

    # mean に最も近い 1本を representative とする
    best_idx = 0
    best_sim = -1.0
    for i in range(arr.shape[0]):
        sim = _cosine_sim(arr[i], mean)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    rep = _l2_normalize(arr[best_idx])
    return rep, mean


# ★追加: 連番付きラベル生成ユーティリティ
def _ensure_serial_label(base_label: str, existing_labels: Iterable[str]) -> str:
    """
    base_label が "Sasakura-fused" のようなベース名の場合、
    既存ラベル群から次の "-P0001" 形式の連番を付与して返す。

    例:
      existing = ["Sasakura-fused-P0001", "Sasakura-fused-P0002"]
      base_label="Sasakura-fused" -> "Sasakura-fused-P0003"

    すでに "-Pdddd" が末尾に付いている場合は、そのまま返す。
    """
    base_label = str(base_label or "").strip()
    if not base_label:
        raise ValueError("base_label is empty")

    # すでに "-P0001" 形式ならそのまま
    if re.search(r"-P\d{4}$", base_label):
        return base_label

    # ベース名 + 既存ラベルから、最大の P 番号を探す
    prefix = base_label + "-P"
    max_n = 0
    for lab in existing_labels:
        lab = str(lab or "").strip()
        if not lab.startswith(prefix):
            continue
        m = re.search(r"-P(\d{4})$", lab)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > max_n:
            max_n = n

    new_n = max_n + 1
    return f"{base_label}-P{new_n:04d}"


# ===================== 顔（Face）統合 =====================

def fuse_face_auto(
    src_labels: Iterable[str],
    new_label: str,
    face_json: Optional[str] = None,
    out_json: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Auto 顔ギャラリーから、指定ラベル群の特徴を集約し、
    「ラベル > 向き(pose) > {rep, mean}」構造の統合結果を保存する。

    - src_labels: 統合元となる Auto ラベル（"Sasakura-P0001" 等）
    - new_label : 統合後のベースラベル名（"Sasakura-fused" 等）
                  ※ ここでは連番は付けず、内部で "-P0001" などを付与する。
    - face_json : 元の Auto 顔ギャラリー（省略時は data/gallery/auto/face_gallery.json）
    - out_json  : 統合結果保存先（省略時は data/gallery/auto/face_fused_gallery.json）

    戻り値:
      (保存パス, 実際に使われたラベル名)
    """
    src_labels = [str(l).strip() for l in src_labels if str(l).strip()]
    if not src_labels:
        raise ValueError("src_labels is empty")

    base_label = str(new_label or "").strip()
    if not base_label:
        raise ValueError("new_label is empty")

    face_json = face_json or _auto_face_json()
    out_json = out_json or _face_fused_json()

    data = _load_json(face_json, default={})
    records = data.get("records") or []

    # records: [{"label": "...", "pose": "...", "vec": [...]}, ...]
    by_pose: Dict[str, List[np.ndarray]] = {}
    dim: Optional[int] = None

    for rec in records:
        try:
            lab = str(rec.get("label", "")).strip()
        except Exception:
            continue
        if lab not in src_labels:
            continue

        pose = str(rec.get("pose", "")).strip() or "unknown"
        v = _to_np_vec(rec.get("vec", []))
        if v.size == 0:
            continue
        if dim is None:
            dim = v.size

        by_pose.setdefault(pose, []).append(v)

    if not by_pose:
        raise ValueError(f"no face records found for labels: {src_labels}")

    fused = _load_json(out_json, default={})
    lab2views: Dict[str, Dict[str, Dict[str, List[float]]]] = fused.setdefault(
        "label_to_views", {}
    )

    # ★ここで既存ラベルを見て連番付きラベルを決定
    existing_labels = list(lab2views.keys())
    final_label = _ensure_serial_label(base_label, existing_labels)

    fused_views: Dict[str, Dict[str, List[float]]] = {}
    for pose, vecs in by_pose.items():
        if not vecs:
            continue
        rep, mean = _select_rep_and_mean(vecs)
        fused_views[pose] = {
            "rep": rep.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
        }

    if not fused_views:
        raise ValueError("fused_views is empty (all poses filtered out)")

    lab2views[final_label] = fused_views
    if dim is not None:
        fused["dim"] = int(dim)
    fused["updated_at"] = int(time.time())

    _save_json(out_json, fused)
    return out_json, final_label


# ===================== 歩容（Gait）統合 =====================

def fuse_gait_auto(
    src_labels: Iterable[str],
    new_label: str,
    gait_json: Optional[str] = None,
    out_json: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Auto 歩容ギャラリーから、指定ラベル群の特徴を集約し、
    「ラベル > 向き(dir8) > {rep, mean}」構造の統合結果を保存する。

    - src_labels: 統合元となる Auto ラベル
    - new_label : 統合後のベースラベル名
                  ※ ここでは連番は付けず、内部で "-P0001" などを付与する。
    - gait_json : 元の Auto 歩容ギャラリー
    - out_json  : 統合結果保存先（デフォルト data/gallery/auto/gait_fused_gallery.json）

    戻り値:
      (保存パス, 実際に使われたラベル名)
    """
    src_labels = [str(l).strip() for l in src_labels if str(l).strip()]
    if not src_labels:
        raise ValueError("src_labels is empty")

    base_label = str(new_label or "").strip()
    if not base_label:
        raise ValueError("new_label is empty")

    gait_json = gait_json or _auto_gait_json()
    out_json = out_json or _gait_fused_json()

    data = _load_json(gait_json, default={})
    records = data.get("records") or []

    # records: [{"label": "...", "dir8" or "view": "...", "vec": [...]}, ...]
    by_view: Dict[str, List[np.ndarray]] = {}
    dim: Optional[int] = None

    for rec in records:
        try:
            lab = str(rec.get("label", "")).strip()
        except Exception:
            continue
        if lab not in src_labels:
            continue

        view = (
            str(rec.get("dir8", "")).strip()
            or str(rec.get("view", "")).strip()
            or "unknown"
        )

        v = _to_np_vec(rec.get("vec", []))
        if v.size == 0:
            continue
        if dim is None:
            dim = v.size

        by_view.setdefault(view, []).append(v)

    if not by_view:
        raise ValueError(f"no gait records found for labels: {src_labels}")

    fused = _load_json(out_json, default={})
    lab2views: Dict[str, Dict[str, Dict[str, List[float]]]] = fused.setdefault(
        "label_to_views", {}
    )

    # ★既存ラベルから連番付きラベルを決定
    existing_labels = list(lab2views.keys())
    final_label = _ensure_serial_label(base_label, existing_labels)

    fused_views: Dict[str, Dict[str, List[float]]] = {}
    for view, vecs in by_view.items():
        if not vecs:
            continue
        rep, mean = _select_rep_and_mean(vecs)
        fused_views[view] = {
            "rep": rep.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
        }

    if not fused_views:
        raise ValueError("fused_views is empty (all views filtered out)")

    lab2views[final_label] = fused_views
    if dim is not None:
        fused["dim"] = int(dim)
    fused["updated_at"] = int(time.time())

    _save_json(out_json, fused)
    return out_json, final_label


# ===================== 外見（Appearance）統合 =====================

def fuse_app_auto(
    src_labels: Iterable[str],
    new_label: str,
    app_json: Optional[str] = None,
    out_json: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Auto 外見ギャラリーから、指定ラベル群の「生ベクトル群」を統合し、
    「ラベル > 統合前ラベル > [生ベクトル...]」構造で保存する。

    - src_labels: 統合元となる Auto ラベル
    - new_label : 統合後ベースラベル
                  ※ ここでは連番は付けず、内部で "-P0001" などを付与する。
    - app_json  : 元の Auto 外見ギャラリー
    - out_json  : 統合結果保存先（デフォルト data/gallery/auto/app_fused_gallery.json）

    ※ 外見は「代表値/平均値」ではなく、生ベクトルを統合前ラベルごとに保持する。

    戻り値:
      (保存パス, 実際に使われたラベル名)
    """
    src_labels = [str(l).strip() for l in src_labels if str(l).strip()]
    if not src_labels:
        raise ValueError("src_labels is empty")

    base_label = str(new_label or "").strip()
    if not base_label:
        raise ValueError("new_label is empty")

    app_json = app_json or _auto_app_json()
    out_json = out_json or _app_fused_json()

    data = _load_json(app_json, default={})
    lab2vecs = data.get("label_to_vecs") or {}

    dim: Optional[int] = None
    sources: Dict[str, List[List[float]]] = {}

    for lab in src_labels:
        vec_list = lab2vecs.get(lab)
        if not isinstance(vec_list, list) or not vec_list:
            continue

        # そのラベルの生ベクトル群をそのままコピー
        cleaned: List[List[float]] = []
        for v_like in vec_list:
            v = _to_np_vec(v_like)
            if v.size == 0:
                continue
            if dim is None:
                dim = v.size
            cleaned.append(v.astype(np.float32).tolist())

        if cleaned:
            sources[lab] = cleaned

    if not sources:
        raise ValueError(f"no appearance vectors found for labels: {src_labels}")

    fused = _load_json(out_json, default={})
    lab2src: Dict[str, Dict[str, List[List[float]]]] = fused.setdefault(
        "label_to_sources", {}
    )

    # ★既存ラベルから連番付きラベルを決定
    existing_labels = list(lab2src.keys())
    final_label = _ensure_serial_label(base_label, existing_labels)

    lab2src[final_label] = sources
    if dim is not None:
        fused["dim"] = int(dim)
    fused["updated_at"] = int(time.time())

    _save_json(out_json, fused)
    return out_json, final_label


# ===================== 一括（全モダリティ）統合ヘルパ =====================

def fuse_all_auto(
    src_labels: Iterable[str],
    new_label: str,
    face_json: Optional[str] = None,
    app_json: Optional[str] = None,
    gait_json: Optional[str] = None,
    face_out_json: Optional[str] = None,
    app_out_json: Optional[str] = None,
    gait_out_json: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    顔・外見・歩容をまとめて統合するヘルパ。

    - src_labels: 統合元ラベル群
    - new_label : 統合後の「ベースラベル名」
                  ※ 実際には "-P0001" などの連番が付いたラベルが使われる。

    どれか一つでエラーになった場合は例外を送出する。
    戻り値は、各モダリティの保存パスと、共通ラベル名:

    {
      "face_path": "...",
      "app_path":  "...",
      "gait_path": "...",
      "label":     "Sasakura-fused-P0001"
    }
    """
    paths: Dict[str, Optional[str]] = {
        "face_path": None,
        "app_path": None,
        "gait_path": None,
        "label": None,
    }

    # まず顔で「連番付きラベル」を決める
    face_path, final_label = fuse_face_auto(
        src_labels=src_labels,
        new_label=new_label,
        face_json=face_json,
        out_json=face_out_json,
    )
    paths["face_path"] = face_path
    paths["label"] = final_label

    # 外見・歩容は、決まった final_label をそのまま使う
    app_path, _ = fuse_app_auto(
        src_labels=src_labels,
        new_label=final_label,  # すでに "-P0001" 付きなので _ensure_serial_label は変更しない
        app_json=app_json,
        out_json=app_out_json,
    )
    paths["app_path"] = app_path

    gait_path, _ = fuse_gait_auto(
        src_labels=src_labels,
        new_label=final_label,
        gait_json=gait_json,
        out_json=gait_out_json,
    )
    paths["gait_path"] = gait_path

    return paths
