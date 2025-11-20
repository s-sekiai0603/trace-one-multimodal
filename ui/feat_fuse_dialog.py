# ui/feat_fuse_dialog.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton
)
from PySide6.QtCore import Qt, Signal


class FeatureFuseDialog(QDialog):
    """
    外見/歩容特徴の 2 ラベルを統合し、新しいラベルとして保存するダイアログ。
    """
    fused = Signal(str, str, str)  
    # → (label_a, label_b, new_label) を返す

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("特徴統合")
        self.setMinimumWidth(340)

        self.labels = labels

        main = QVBoxLayout(self)

        # --- 1段目（上のドロップダウン）---
        self.cmb1 = QComboBox()
        self.cmb1.addItem("")  # 未選択
        for lb in labels:
            self.cmb1.addItem(lb)

        # --- 2段目（& と 下のドロップダウン）---
        mid = QHBoxLayout()
        self.cmb2 = QComboBox()
        self.cmb2.addItem("")  # 未選択
        for lb in labels:
            self.cmb2.addItem(lb)

        mid.addWidget(self.cmb1)
        mid.addWidget(QLabel("  &  "))
        mid.addWidget(self.cmb2)

        # --- 新しいラベル名 ---
        self.txt_new = QLineEdit()
        self.txt_new.setPlaceholderText("新しい保存用ラベル名")

        # --- ボタン ---
        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("統合")
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_ok.setEnabled(False)

        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)

        # 配置
        main.addLayout(mid)
        main.addWidget(self.txt_new)
        main.addLayout(btn_row)

        # --- シグナル ---
        self.cmb1.currentTextChanged.connect(self._validate)
        self.cmb2.currentTextChanged.connect(self._validate)
        self.txt_new.textChanged.connect(self._validate)
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)

    def _validate(self):
        ok = (
            self.cmb1.currentText().strip() != "" and
            self.cmb2.currentText().strip() != "" and
            self.txt_new.text().strip() != ""
        )
        self.btn_ok.setEnabled(ok)

    def _on_ok(self):
        a = self.cmb1.currentText().strip()
        b = self.cmb2.currentText().strip()
        new = self.txt_new.text().strip()
        if a and b and new:
            self.fused.emit(a, b, new)
            self.accept()
