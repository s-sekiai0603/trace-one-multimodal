# main.py
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import sys

from .ui.player import PlayerWindow
from .controllers.app_controller import AppController
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def main():
    app = QApplication(sys.argv)

    w = PlayerWindow()
    w.ctrl = AppController(w)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
