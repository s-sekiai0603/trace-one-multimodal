# -*- coding: utf-8 -*-
"""
ui/feat_fuse_dialog.py — 特徴統合ダイアログ

目的:
- Autoモードで収集した特徴を「統合」するためのUI。
- ユーザーが:
  - モード（顔 / 外見 / 歩容 / ALL）
  - 統合対象ラベルA
  - 統合対象ラベルB
  - 新しいラベル名（ベース名）
  を指定できる。

このダイアログ自体は「UIのみ」を担当し、
実際の統合処理は db/feat_fuse.py 側の
  - fuse_face_auto
  - fuse_app_auto
  - fuse_gait_auto
  - fuse_all_auto
などから呼び出してもらう想定。
"""
from __future__ import annotations

import os
import json
import logging
import re
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
)
from PySide6.QtCore import Qt

from ..config import AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON

log = logging.getLogger("app.ui")


# ===================== Autoラベル収集ヘルパ =====================

def _collect_auto_labels() -> List[str]:
    """
    Auto モードで保存済みの「名前+連番」ラベルを収集してソート。

    対象:
      - AUTO_FACE_JSON / AUTO_APP_JSON / AUTO_GAIT_JSON
      - 同じフォルダ内の face_fused_gallery.json / app_fused_gallery.json / gait_fused_gallery.json
    """
    labels: set[str] = set()

    def _scan(path: str) -> None:
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
            # records を持つ場合はまずそれを優先
            if "records" in data and isinstance(data["records"], list):
                it = data["records"]
            else:
                # マップ系: label_to_vecs / label_to_views / label_to_sources / それ以外
                if "label_to_vecs" in data and isinstance(data["label_to_vecs"], dict):
                    keys = data["label_to_vecs"].keys()
                elif "label_to_views" in data and isinstance(data["label_to_views"], dict):
                    keys = data["label_to_views"].keys()
                elif "label_to_sources" in data and isinstance(data["label_to_sources"], dict):
                    keys = data["label_to_sources"].keys()
                else:
                    keys = data.keys()
                it = [{"pid": k} for k in keys]
        else:
            it = []

        for rec in it:
            val = None
            if isinstance(rec, dict):
                for key in ("pid", "label", "name", "id"):
                    if key in rec:
                        val = str(rec[key])
                        break
                if val is None and rec:
                    val = str(next(iter(rec.keys())))
            elif isinstance(rec, str):
                val = rec

            if not val:
                continue

            s = val.strip()
            # “名前-連番”だけ採用（例: Mizuta-P0001 / Mizuta-back-fused-P0001 など）
            if re.search(r"-[A-Za-z]?(\d{2,})$", s):
                labels.add(s)

    # ベース: 従来3ファイル
    for p in (AUTO_FACE_JSON, AUTO_APP_JSON, AUTO_GAIT_JSON):
        _scan(p)

    # 追加: auto/ 配下の fused ギャラリー
    try:
        base_dir = os.path.dirname(AUTO_FACE_JSON) or os.path.dirname(AUTO_APP_JSON) or os.path.dirname(AUTO_GAIT_JSON)
    except Exception:
        base_dir = ""

    if base_dir:
        fused_files = (
            os.path.join(base_dir, "face_fused_gallery.json"),
            os.path.join(base_dir, "app_fused_gallery.json"),
            os.path.join(base_dir, "gait_fused_gallery.json"),
        )
        for p in fused_files:
            _scan(p)

    # ベース名→ラベルで安定ソート
    return sorted(labels, key=lambda x: (x.split("-")[0].lower(), x))



# ===================== ダイアログ本体 =====================

class FeatFuseDialog(QDialog):
    """
    特徴統合ダイアログ。

    使い方イメージ:
        dlg = FeatFuseDialog(self)
        if dlg.exec() == QDialog.Accepted:
            mode_key = dlg.selected_mode_key()      # "face" / "appearance" / "gait" / "all"
            a, b = dlg.selected_labels()            # ("Sasakura-P0001", "Sasakura-P0002")
            base = dlg.new_label_base()             # 例: "Sasakura-fused"
            # ここで db.feat_fuse の fuse_* を呼び出す
    """

    MODE_ITEMS = [
        ("顔", "face"),
        ("外見", "appearance"),
        ("歩容", "gait"),
        ("ALL", "all"),
    ]

    def __init__(self, parent=None, *, default_mode: str = "all"):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("特徴統合")

        v = QVBoxLayout(self)

        # --- フォーム部 ---
        form = QFormLayout()

        # モード選択
        self.cmb_mode = QComboBox(self)
        for text, key in self.MODE_ITEMS:
            self.cmb_mode.addItem(text, userData=key)
        # 既定モードを選択
        self._set_default_mode(default_mode)
        form.addRow(QLabel("モード:", self), self.cmb_mode)

        # 統合対象A/B
        self.cmb_label_a = QComboBox(self)
        self.cmb_label_b = QComboBox(self)
        form.addRow(QLabel("統合先:", self), self.cmb_label_a)
        form.addRow(QLabel("統合元:", self), self.cmb_label_b)

        # 新ラベル（ベース名）
        self.edt_new_label = QLineEdit(self)
        self.edt_new_label.setPlaceholderText("")
        form.addRow(QLabel("新ラベル名:", self), self.edt_new_label)

        v.addLayout(form)

        # --- ボタン行 ---
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_ok = QPushButton("統合", self)
        self.btn_cancel = QPushButton("キャンセル", self)
        self.btn_ok.setDefault(True)
        self.btn_ok.setAutoDefault(True)
        row.addWidget(self.btn_ok)
        row.addWidget(self.btn_cancel)
        v.addLayout(row)

        # --- シグナル接続 ---
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)

        # ラベル一覧をロードしてコンボに反映
        self._auto_labels: List[str] = _collect_auto_labels()
        self._populate_label_combos()

        # 何もラベルが無い場合は警告
        if not self._auto_labels:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("警告")
            msg.setText("統合可能なラベルが見つかりません。\nAutoモードで特徴を保存してから再度お試しください。")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            # ラベルが無いと統合できないので、OKボタンを無効化
            self.btn_ok.setEnabled(False)

    # ------------------------- 内部ヘルパ -------------------------

    def _set_default_mode(self, mode_key: str) -> None:
        mode_key = str(mode_key or "").strip().lower()
        for i in range(self.cmb_mode.count()):
            k = str(self.cmb_mode.itemData(i) or "")
            if k == mode_key:
                self.cmb_mode.setCurrentIndex(i)
                return
        # 見つからなければ ALL
        for i in range(self.cmb_mode.count()):
            k = str(self.cmb_mode.itemData(i) or "")
            if k == "all":
                self.cmb_mode.setCurrentIndex(i)
                return

    def _populate_label_combos(self) -> None:
        """
        Autoラベル一覧を A/B 両方のコンボボックスに流し込む。
        （モードに依存せず、共通のラベル集合を使う）
        """
        self.cmb_label_a.clear()
        self.cmb_label_b.clear()

        for lab in self._auto_labels:
            self.cmb_label_a.addItem(lab, userData=lab)
            self.cmb_label_b.addItem(lab, userData=lab)

        # デフォルトで A/B が同じ値にならないように少しだけ調整
        if self.cmb_label_a.count() > 0:
            self.cmb_label_a.setCurrentIndex(0)
        if self.cmb_label_b.count() > 1:
            self.cmb_label_b.setCurrentIndex(1)

    def _on_accept(self) -> None:
        """
        OK(統合)ボタンクリック時のバリデーション。
        条件を満たさない場合はメッセージを出してダイアログは閉じない。
        """
        mode_key = self.selected_mode_key()
        lab_a, lab_b = self.selected_labels()
        new_label = self.new_label_base()

        # モード必須
        if not mode_key:
            QMessageBox.warning(self, "エラー", "モードを選択してください。")
            return

        # ラベルA/B 必須
        if not lab_a or not lab_b:
            QMessageBox.warning(self, "エラー", "統合対象A/Bのラベルを選択してください。")
            return

        # 同じラベル同士の統合は避ける（必要であれば許可してもよい）
        if lab_a == lab_b:
            QMessageBox.warning(self, "エラー", "統合対象AとBは異なるラベルを選択してください。")
            return

        # 新ラベル名（ベース）が空なら警告
        if not new_label:
            QMessageBox.warning(self, "エラー", "新しいラベル名を入力してください。")
            return

        # すべてOKなら accept()
        self.accept()

    # ------------------------- 公開getter -------------------------

    def selected_mode_key(self) -> str:
        """
        選択モードの内部キーを返す。
        "face" / "appearance" / "gait" / "all"
        """
        return str(self.cmb_mode.currentData() or "").strip()

    def selected_mode_label(self) -> str:
        """
        選択モードの表示テキスト（日本語）を返す。
        "顔" / "外見" / "歩容" / "ALL"
        """
        return str(self.cmb_mode.currentText() or "").strip()

    def selected_labels(self) -> tuple[str, str]:
        """
        統合対象A/Bのラベル名を返す。
        """
        a = str(self.cmb_label_a.currentData() or self.cmb_label_a.currentText() or "").strip()
        b = str(self.cmb_label_b.currentData() or self.cmb_label_b.currentText() or "").strip()
        return a, b

    def new_label_base(self) -> str:
        """
        新しいラベル名（ベース）を返す。

        例:
        - "Sasakura-fused" など
        ※ 実際の "-P0001" などの連番付与は db/feat_fuse.py 側で行う想定。
        """
        return str(self.edt_new_label.text() or "").strip()
