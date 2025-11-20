# -*- coding: utf-8 -*-
"""
ui/identify.py — 識別設定ダイアログ
- ギャラリーJSONをドロップダウン選択
- 保存済みのID(pid)をドロップダウン選択
- 左：キャンセル / 右：識別（既定）
"""
from __future__ import annotations
import os, json, logging, re
from typing import List, Tuple, Optional

from ..config import AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON

log = logging.getLogger("app.ui")

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QLabel,
    QPushButton, QHBoxLayout, QWidget
)
from PySide6.QtCore import Qt

def _project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _find_gallery_jsons() -> List[Tuple[str, str]]:
    """
    既定フォルダ配下のギャラリーJSON候補を列挙。
    戻り値: [(表示名, フルパス), ...]
    """
    root = _project_root()
    base = os.path.join(root, "data", "gallery")
    result = []
    for sub in ("face", "appearance", "gait"):
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".json"):
                path = os.path.join(d, fn)
                disp = f"{sub}: {fn}"
                result.append((disp, path))
    # フォールバック：何もないときはデフォルト想定を出す
    if not result:
        # face_gallery.json を暗黙想定（存在していなくても可）
        path = os.path.join(base, "face", "face_gallery.json")
        result.append(("face: face_gallery.json", path))
    return result

def _load_pids_from_json(json_path: str) -> List[str]:
    """
    ギャラリーJSON/JSONLからPID(またはlabel)一覧を抽出するユーティリティ。

    対応フォーマット:
      - face:
          data/gallery/face/face_gallery.json
          → JSON配列: [ {"pid": "...", ...}, ... ]

      - appearance(外見):
          data/gallery/appearance/appearance_gallery.json
          → JSON dict: { "label_to_vecs": { "<pid>": [...], ... } }
          もしくは labels.jsonl:
          → 1行1レコードのJSONL: {"pid": "..."} または {"label": "..."}

      - gait(歩容):
          data/gallery/gait/gait_gallery.json
          → JSON dict: { "items": [ {"pid": "...", ...}, ... ], "updated_at": "..." }
    """
    try:
        pids: List[str] = []

        # --- JSONL形式 (1行1レコード) ---
        if json_path.lower().endswith(".jsonl"):
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        # 壊れた行があっても他行は読む
                        continue
                    if not isinstance(rec, dict):
                        continue

                    # {"pid": "..."} 優先
                    if "pid" in rec:
                        pids.append(str(rec["pid"]).strip())
                    # 次に {"label": "..."} もPID相当として扱う
                    elif "label" in rec:
                        pids.append(str(rec["label"]).strip())

        else:
            # --- JSON形式 (list または dict) ---
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 顔用: [{"pid": "A001", ...}]
            if isinstance(data, list):
                for e in data:
                    if isinstance(e, dict) and "pid" in e:
                        pids.append(str(e.get("pid", "")).strip())

            # dict の場合: 外見 / 歩容 など
            elif isinstance(data, dict):
                # 外見用: {"label_to_vecs": {...}}
                if "label_to_vecs" in data and isinstance(data["label_to_vecs"], dict):
                    for k in data["label_to_vecs"].keys():
                        pids.append(str(k).strip())

                # 歩容用: {"items": [ {"pid": "...", ...}, ... ], ...}
                if "items" in data and isinstance(data["items"], list):
                    for e in data["items"]:
                        if isinstance(e, dict) and "pid" in e:
                            pids.append(str(e.get("pid", "")).strip())

        # --- 重複除去 + ソート ---
        seen = set()
        uniq: List[str] = []
        for p in pids:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)

        return sorted(uniq)

    except Exception:
        # 何かあったら「PIDなし」として扱う（呼び出し側でフォールバック）
        return []

def _collect_auto_labels() -> list[str]:
    labels = set()

    def _scan(path: str):
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if isinstance(data, list):
            it = data
        elif isinstance(data, dict):
            if "label_to_vecs" in data and isinstance(data["label_to_vecs"], dict):
                it = [{"pid": k} for k in data["label_to_vecs"].keys()]
            else:
                it = [{"pid": k} for k in data.keys()]
        else:
            it = []

        for rec in it:
            val = None
            if isinstance(rec, dict):
                for key in ("pid", "label", "name", "id"):
                    if key in rec:
                        val = str(rec[key]); break
                if val is None and len(rec) == 1:
                    val = str(next(iter(rec.keys())))
            elif isinstance(rec, str):
                val = rec
            if not val:
                continue
            s = val.strip()
            # “名前-連番”だけ採用（例: Mizuta-P0001 / Aoki-0003 など）
            if re.search(r"-[A-Za-z]?(\d{2,})$", s):
                labels.add(s)

    for p in (AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON):
        _scan(p)

    # ベース名→ラベルで安定ソート
    return sorted(labels, key=lambda x: (x.split("-")[0].lower(), x))

class IdentifyDialog(QDialog):
    def __init__(self,
                parent=None,
                default_gallery: Optional[str] = None,
                *,
                auto_mode: bool = False,
                auto_labels: Optional[List[str]] = None):
        super().__init__(parent)
        self._auto_mode = bool(auto_mode)
        self.setModal(True)
        self.setWindowTitle("識別設定（オート）" if self._auto_mode else "識別設定")

        v = QVBoxLayout(self)

        # === ★ autoモード：ギャラリーUIは一切作らない（早期return） ===
        if self._auto_mode:
            self.lbl_auto = QLabel("保存した『名前+連番』:", self)
            self.cmb_auto = QComboBox(self)
            labels = auto_labels if auto_labels is not None else _collect_auto_labels()
            if labels:
                self.cmb_auto.addItems(labels)

            v.addWidget(self.lbl_auto)
            v.addWidget(self.cmb_auto)

            row = QHBoxLayout()
            self.btn_identify = QPushButton("識別", self)
            self.btn_cancel   = QPushButton("キャンセル", self)
            self.btn_identify.clicked.connect(self.accept)
            self.btn_cancel.clicked.connect(self.reject)
            self.btn_identify.setEnabled(self.cmb_auto.count() > 0)
            row.addWidget(self.btn_identify)
            row.addWidget(self.btn_cancel)
            v.addLayout(row)

            self.setMinimumWidth(420)
            return  # ★ 通常モード初期化に進まない

        # === 通常モード（従来どおり：ギャラリー + PID） ===
        form = QFormLayout()
        self.cmb_gallery = QComboBox(self)
        self.cmb_pid     = QComboBox(self)
        form.addRow(QLabel("ギャラリー:", self), self.cmb_gallery)
        form.addRow(QLabel("対象ID:", self),     self.cmb_pid)
        v.addLayout(form)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_identify = QPushButton("識別", self)
        self.btn_cancel   = QPushButton("キャンセル", self)
        self.btn_identify.setDefault(True)
        self.btn_identify.setAutoDefault(True)
        self.btn_identify.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        row.addWidget(self.btn_identify)
        row.addWidget(self.btn_cancel)
        v.addLayout(row)

        # シグナル
        self.cmb_gallery.currentIndexChanged.connect(self._on_gallery_changed)

        # ギャラリー列挙
        self._galleries = _find_gallery_jsons()
        for disp, path in self._galleries:
            self.cmb_gallery.addItem(disp, userData=path)

        # 既定ギャラリー選択
        if default_gallery:
            for i in range(self.cmb_gallery.count()):
                if self.cmb_gallery.itemData(i) == default_gallery:
                    self.cmb_gallery.setCurrentIndex(i)
                    break

        # 初回のIDロード
        self._on_gallery_changed(self.cmb_gallery.currentIndex())
        self.setMinimumWidth(420)

    def _on_gallery_changed(self, idx: int):
        path = self.cmb_gallery.itemData(idx)
        pids = _load_pids_from_json(path) if path else []
        self.cmb_pid.clear()
        if not pids:
            self.cmb_pid.addItem("(登録なし)", userData="")
        else:
            for p in pids:
                self.cmb_pid.addItem(p, userData=p)

    # 公開getter
    def selected_gallery_path(self) -> str:
        if getattr(self, "_auto_mode", False):
            return ""  # autoではギャラリー未使用
        return str(self.cmb_gallery.currentData() or self.cmb_gallery.currentText() or "")

    def selected_pid(self) -> str:
        if getattr(self, "_auto_mode", False):
            # “名前+連番”をPID相当で返す（例: Mizuta-P0001）
            return str(self.cmb_auto.currentText() if hasattr(self, "cmb_auto") else "").strip()
        return str(self.cmb_pid.currentText() or "").strip()
