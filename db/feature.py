# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict
import os, json, time, re
import numpy as np

# ----------------- パス系 -----------------
def _project_root() -> str:
    # src/db/feature.py → ../../
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _face_gallery_dir() -> str:
    return os.path.join(_project_root(), "data", "gallery", "face")

def _default_face_gallery_json() -> str:
    os.makedirs(_face_gallery_dir(), exist_ok=True)
    return os.path.join(_face_gallery_dir(), "face_gallery.json")

def _face_pid_index_json() -> str:
    # 末尾連番の永続化ファイル
    os.makedirs(_face_gallery_dir(), exist_ok=True)
    return os.path.join(_face_gallery_dir(), "face_pid_index.json")

# ----------------- ヘルパ -----------------
def _load_json_list(path: str) -> list:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _load_pid_index(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def _save_pid_index(path: str, d: Dict[str, int]) -> None:
    tmp = dict(sorted(d.items(), key=lambda kv: kv[0]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)

def _max_suffix_in_gallery(entries: list, base: str) -> int:
    """ギャラリー内の base に対する P連番の最大を返す（無ければ0）。"""
    pat = re.compile(rf"^{re.escape(base)}-P(\d{{4}})$")
    mx = 0
    for e in entries:
        pid = str(e.get("pid", "")).strip()
        m = pat.match(pid)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

def _alloc_pid(existing_max_in_gallery: int, idx_last: int, base: str) -> str:
    n = max(existing_max_in_gallery, idx_last) + 1
    return f"{base}-P{n:04d}", n

# ----------------- 本体 -----------------
def save_face_features(
    pid: str,
    feats: List[np.ndarray],
    json_path: Optional[str] = None
) -> str:
    """
    学習セッションで蓄積した複数フレームの埋め込み（feats）を
    平均→L2正規化した 1本だけ保存する（＝一回の学習で1ID）。
    末尾の P連番は face_pid_index.json に永続化し、再起動後も+1で継続。

    レコード形式: {"pid": "<base>-P0001", "feat": [...], "ts": 1731030000}
    """
    if not feats:
        raise ValueError("保存する特徴がありません（feats が空）")

    # パス準備
    gal_path = os.path.abspath(json_path or _default_face_gallery_json())
    os.makedirs(os.path.dirname(gal_path), exist_ok=True)
    idx_path = _face_pid_index_json()

    # 既存読込
    gallery = _load_json_list(gal_path)
    pid_index = _load_pid_index(idx_path)  # { base: last_num }

    base = str(pid).strip()
    # ギャラリー内の実績とインデックスの両方を見る
    max_in_gal = _max_suffix_in_gallery(gallery, base)
    last_idx   = int(pid_index.get(base, 0))

    # --- 平均→L2正規化 ---
    arr = np.asarray([np.asarray(v, dtype=float).reshape(-1) for v in feats], dtype=float)
    gal = arr.mean(axis=0).astype(np.float32)
    nrm = float(np.linalg.norm(gal))
    if nrm < 1e-12:
        raise ValueError("平均ベクトルのノルムが 0 です。入力特徴を確認してください。")
    gal = (gal / nrm).tolist()

    # --- 連番払い出し（永続化対応） ---
    next_pid, used_num = _alloc_pid(max_in_gal, last_idx, base)
    pid_index[base] = used_num  # ここで更新して保存

    # --- 1件だけ追記 ---
    now = int(time.time())
    gallery.append({"pid": next_pid, "feat": gal, "ts": now})

    # 書き戻し
    with open(gal_path, "w", encoding="utf-8") as f:
        json.dump(gallery, f, ensure_ascii=False, indent=2)
    _save_pid_index(idx_path, pid_index)

    return gal_path
