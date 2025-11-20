# utils/feat_fuse.py
import json
import os
import time
import numpy as np
from typing import Any, Dict, List, Set

def load_gallery(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    return []


def save_gallery(json_path: str, items: list[dict]):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)


def fuse_features(
    json_path: str,
    label_a: str,
    label_b: str,
    new_label: str,
    mode: str
):
    """
    外見/歩容ギャラリーを読み取り、
    2つのラベル（特徴）を統合した新しいラベルを追加する。

    - 元のラベルの特徴（feat）は削除しない
    - 新しい特徴はサブラベルとして sub_pids に保持
    - 特徴ベクトル自体はここでは作らず、識別時に動的に「より近いほう」を使う
    """
    items = load_gallery(json_path)

    # --- 2つのラベルを抽出 ---
    sub_feats = []
    for it in items:
        if str(it.get("pid")) in (label_a, label_b):
            fv = np.asarray(it.get("feat"), dtype=np.float32)
            if fv.ndim == 1:
                sub_feats.append((str(it["pid"]), fv))

    if len(sub_feats) < 2:
        raise RuntimeError("指定ラベルの特徴が揃っていません")

    # --- 新しい項目として統合ラベルを追加 ---
    new_item = {
        "pid": new_label,
        "mode": mode,                       # "appearance" or "gait"
        "sub_pids": [label_a, label_b],     # ← ここが重要
        "meta": {"type": "fused"},
        "ts": float(os.path.getmtime(json_path)) if os.path.exists(json_path) else 0
    }

    items.append(new_item)
    save_gallery(json_path, items)

    return new_item

# =========================
# オート外見ギャラリー用の統合
# =========================

def _ff_load_json(path: str):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ff_save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fuse_auto_app_features(json_path: str,
                           label_a: str,
                           label_b: str,
                           new_label: str) -> None:
    """
    オート外見ギャラリー(AUTO_APP_JSON)用の統合処理。

    想定フォーマット:
      1) {"label_to_vecs": {label: [[...], [...]], ...}}
         → new_label に A/B のベクトルを全部突っ込む（既存 new_label があれば append）
         → 元の A/B は一切削除・変更しない

      2) list[{"pid":..., "vec": [...]}] 形式などは、
         現状は何もしない（フォーマットを増やす場合はここに追記する）。
    """
    data = _ff_load_json(json_path)
    if data is None:
        return

    label_a = (label_a or "").strip()
    label_b = (label_b or "").strip()
    new_label = (new_label or "").strip()
    if not (label_a and label_b and new_label):
        return

    # --- パターン1: dict + label_to_vecs 形式 ---
    if isinstance(data, dict) and isinstance(data.get("label_to_vecs"), dict):
        m: Dict[str, Any] = data["label_to_vecs"]

        def _get_vecs(lb: str) -> List[List[float]]:
            arr = m.get(lb)
            if not isinstance(arr, list):
                return []
            out: List[List[float]] = []
            for v in arr:
                a = np.asarray(v, np.float32)
                if a.ndim == 1:
                    out.append(a.tolist())
                elif a.ndim == 2:
                    for r in a:
                        out.append(np.asarray(r, np.float32).tolist())
            return out

        vecs_a = _get_vecs(label_a)
        vecs_b = _get_vecs(label_b)
        if not (vecs_a or vecs_b):
            # 統合対象が何もなければ何もしない
            return

        existed = m.get(new_label, [])
        # 既存 + A + B を結合
        merged = []
        def _append_clean(lst):
            for v in lst:
                a = np.asarray(v, np.float32)
                if a.size > 0:
                    merged.append(a.tolist())

        _append_clean(existed)
        _append_clean(vecs_a)
        _append_clean(vecs_b)

        # 空しか無ければキー自体を消す（≒作らない）
        if len(merged) == 0:
            if new_label in m:
                m.pop(new_label, None)
            _ff_save_json(json_path, data)
            return

        m[new_label] = merged
        _ff_save_json(json_path, data)
        return

    # --- それ以外の形式は、現状は何もしない ---
    return

# =========================
# オート歩容ギャラリー用の統合
# =========================
def fuse_auto_gait_features(json_path: str,
                            label_a: str,
                            label_b: str,
                            new_label: str) -> None:
    """
    オート歩容ギャラリー(AUTO_GAIT_JSON)用の統合処理。

    想定フォーマット:
      - dict{"records": [...]} もしくは list[...] で、
        各要素 rec は:
          rec = {"label": str, "view": str, "vec": [...], ...}

    仕様:
      - label が label_a / label_b のレコードを集め、
        view ごとに平均ベクトルを new_label として追加する。
      - 元の A/B レコードは一切変更・削除しない。
      - 同じ new_label, view に対して何度か追加されても、
        AutoController 側の「view別ギャラリー読み込み＋平均化」で破綻しない。
    """
    data = _ff_load_json(json_path)
    if data is None:
        return

    label_a = (label_a or "").strip()
    label_b = (label_b or "").strip()
    new_label = (new_label or "").strip()
    if not (label_a and label_b and new_label):
        return

    # records を取り出す
    if isinstance(data, dict):
        recs = data.get("records")
        if not isinstance(recs, list):
            recs = []
            data["records"] = recs
    elif isinstance(data, list):
        recs = data
        # list形式の場合は data = recs のまま保存する
    else:
        return

    target_labels = {label_a, label_b}

    # A/B のレコードを view 別に集約
    view_to_vecs: Dict[str, List[np.ndarray]] = {}

    for rec in recs:
        if not isinstance(rec, dict):
            continue
        lb = str(rec.get("label", "")).strip()
        if lb not in target_labels:
            continue

        view = str(rec.get("view", "")).strip().lower()
        if not view:
            continue

        v = rec.get("vec")
        if v is None:
            continue

        a = np.asarray(v, np.float32).reshape(-1)
        n = float(np.linalg.norm(a)) + 1e-12
        a = a / n
        view_to_vecs.setdefault(view, []).append(a)

    if not view_to_vecs:
        # 統合対象の歩容レコードが無ければ何もしない
        return

    now = int(time.time())

    # ★ 既存の new_label で vec が空 or 欠損のレコードを掃除
    recs[:] = [
        r for r in recs
        if not (isinstance(r, dict)
                and str(r.get("label","")).strip() == new_label
                and (r.get("vec") is None or (isinstance(r.get("vec"), list) and len(r["vec"]) == 0)))
    ]

    added = 0
    for view, vecs in view_to_vecs.items():
        if not vecs:
            continue
        m = np.stack(vecs, axis=0).mean(axis=0)
        n = float(np.linalg.norm(m)) + 1e-12
        m = (m / n).astype(np.float32)
        recs.append({
            "label": new_label,
            "view": view,
            "vec": m.tolist(),
            "from_labels": [label_a, label_b],
            "ts": now,
        })
        added += 1

    # ★ 1つも作れなかったら保存だけして終了（空の new_label は作らない）
    _ff_save_json(json_path, data)