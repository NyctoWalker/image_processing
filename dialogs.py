from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialogButtonBox, QFormLayout, QSlider,
    QLabel, QPushButton, QInputDialog
)
from PyQt6.QtCore import Qt
import numpy as np


class KernelDialog(QDialog):
    def __init__(self, size=3, kernel=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка ядра")
        self.size = size

        if kernel is None or kernel.shape[0] != size:
            self.kernel = np.zeros((size, size))
            if size % 2 == 1:
                center = size // 2
                self.kernel[center, center] = 1.0
        else:
            self.kernel = kernel.copy()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Таблица для ввода ядра
        self.table = QTableWidget(self.size, self.size)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Заполняем таблицу значениями
        for i in range(self.size):
            for j in range(self.size):
                item = QTableWidgetItem(str(self.kernel[i, j]))
                self.table.setItem(i, j, item)

        layout.addWidget(self.table)

        # Кнопки
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_kernel(self):
        # Считываем значения из таблицы
        for i in range(self.size):
            for j in range(self.size):
                try:
                    text = self.table.item(i, j).text()
                    text = text.replace(',', '.')
                    value = float(text)
                    self.kernel[i, j] = value
                except (ValueError, AttributeError):
                    self.kernel[i, j] = 0.0
        return self.kernel
