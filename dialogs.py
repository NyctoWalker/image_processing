import time

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialogButtonBox, QFormLayout, QSlider,
    QLabel, QPushButton, QInputDialog, QGroupBox, QHBoxLayout, QComboBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
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
        kernel = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                item = self.table.item(i, j)
                if item is not None:
                    try:
                        text = item.text().replace(',', '.')
                        kernel[i, j] = float(text)
                    except (ValueError, AttributeError):
                        kernel[i, j] = 0.0
        return kernel


class PixelArtDialog(QWidget):
    preview_requested = pyqtSignal(dict)

    def __init__(self, params=None, parent=None):
        super().__init__(parent)
        self.default_params = {
            "pixel_size": 8,
            "method": "quantize",
            "num_colors": 16,
            "edge_threshold": 30,
            "dither_strength": 50
        }
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        self.last_update_time = 0
        self.method_slider_connections = []
        self.init_ui()

        # self.pixel_size_slider["slider"].valueChanged.connect(self.emit_preview)
        # self.method_combo.currentTextChanged.connect(self.handle_method_change)

    def init_ui(self):
        layout = QVBoxLayout(self)

        method_group = QGroupBox("Метод для Pixel Art (тяжёлый фильтр)")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        methods = ["Simple", "Quantize", "Edge Preserving", "Dither"]
        self.method_combo.addItems(methods)

        current_method = self.params.get("method", "quantize")
        method_index = {"simple": 0, "quantize": 1, "edge": 2, "dither": 3}.get(current_method, 1)

        try:
            self.method_combo.setCurrentIndex(method_index)
        except Exception:
            self.method_combo.setCurrentIndex(1)  # Quantize

        method_layout.addWidget(QLabel("Метод обработки:"))
        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        common_group = QGroupBox("Общие настройки")
        common_layout = QVBoxLayout()

        pixel_size = max(1, min(32, self.params.get("pixel_size", 8)))
        self.pixel_size_slider = self.create_slider("Pixel Size:", pixel_size, 1, 32, "px")
        common_layout.addLayout(self.pixel_size_slider["layout"])

        common_group.setLayout(common_layout)
        layout.addWidget(common_group)

        self.method_group = QGroupBox("Настройки метода")
        self.method_layout = QVBoxLayout()
        self.method_group.setLayout(self.method_layout)
        layout.addWidget(self.method_group)

        self.controls = {}
        self.create_method_controls()
        self.update_visible_controls()

        # self.pixel_size_slider["slider"].valueChanged.connect(self.emit_preview)
        self.pixel_size_slider["slider"].sliderReleased.connect(self.emit_preview)
        self.method_combo.currentTextChanged.connect(self.handle_method_change)

    def create_slider(self, label, value, min_val, max_val, suffix=""):
        layout = QHBoxLayout()

        value = max(min_val, min(max_val, value))  # Clamp

        lbl = QLabel(label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(value)

        value_lbl = QLabel(f"{value}{suffix}")
        slider.valueChanged.connect(lambda v: value_lbl.setText(f"{v}{suffix}"))

        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(value_lbl)

        return {
            "layout": layout,
            "slider": slider,
            "label": value_lbl
        }

    def create_method_controls(self):
        while self.method_layout.count():
            item = self.method_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.controls = {
            "quantize": {},
            "edge": {},
            "dither": {}
        }

        # Quantize
        quantize_group = QGroupBox("Квантизация цветов")
        quantize_layout = QVBoxLayout()

        num_colors = max(2, min(32, self.params.get("num_colors", 16)))
        self.controls["quantize"]["num_colors"] = self.create_slider("Количество цветов:", num_colors, 2, 32)
        quantize_layout.addLayout(self.controls["quantize"]["num_colors"]["layout"])

        quantize_group.setLayout(quantize_layout)
        quantize_group.setVisible(False)
        self.method_layout.addWidget(quantize_group)
        self.controls["quantize"]["group"] = quantize_group

        # Edge
        edge_group = QGroupBox("Сохранение граней")
        edge_layout = QVBoxLayout()

        edge_threshold = max(1, min(100, self.params.get("edge_threshold", 30)))
        self.controls["edge"]["edge_threshold"] = self.create_slider("Порог границ:", edge_threshold, 1, 100)
        edge_layout.addLayout(self.controls["edge"]["edge_threshold"]["layout"])

        edge_group.setLayout(edge_layout)
        edge_group.setVisible(False)
        self.method_layout.addWidget(edge_group)
        self.controls["edge"]["group"] = edge_group

        # Dither
        dither_group = QGroupBox("Дизеринг")
        dither_layout = QVBoxLayout()

        dither_strength = max(0, min(100, self.params.get("dither_strength", 50)))
        self.controls["dither"]["dither_strength"] = self.create_slider("Сила дизеринга:", dither_strength, 0, 100, "%")
        dither_layout.addLayout(self.controls["dither"]["dither_strength"]["layout"])

        dither_group.setLayout(dither_layout)
        dither_group.setVisible(False)
        self.method_layout.addWidget(dither_group)
        self.controls["dither"]["group"] = dither_group

        self.disconnect_method_sliders()

        if "num_colors" in self.controls["quantize"]:
            self.method_slider_connections.append(
                # self.controls["quantize"]["num_colors"]["slider"].valueChanged.connect(self.emit_preview))
                self.controls["quantize"]["num_colors"]["slider"].sliderReleased.connect(self.emit_preview))

        if "edge_threshold" in self.controls["edge"]:
            self.method_slider_connections.append(
                # self.controls["edge"]["edge_threshold"]["slider"].valueChanged.connect(self.emit_preview))
                self.controls["edge"]["edge_threshold"]["slider"].sliderReleased.connect(self.emit_preview))

        if "dither_strength" in self.controls["dither"]:
            self.method_slider_connections.append(
                # self.controls["dither"]["dither_strength"]["slider"].valueChanged.connect(self.emit_preview))
                self.controls["dither"]["dither_strength"]["slider"].sliderReleased.connect(self.emit_preview))

    def disconnect_method_sliders(self):
        for connection in self.method_slider_connections:
            try:
                connection.disconnect()
            except:
                pass
        self.method_slider_connections = []

    def handle_method_change(self):
        self.update_visible_controls()
        self.emit_preview()

    def emit_preview(self):
        if hasattr(self, '_update_timer'):
            self._update_timer.stop()

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._process_preview_emit)
        self._update_timer.start(350)

    def _process_preview_emit(self):
        params = self.get_params()
        try:
            self.preview_requested.emit(params)
        except Exception as e:
            print(f"Error emitting preview signal: {e}")

    def update_visible_controls(self):
        try:
            method = self.method_combo.currentText().lower()
            method_key = {
                "simple": None,
                "quantize": "quantize",
                "edge preserving": "edge",
                "dither": "dither"
            }.get(method)

            for control_group in self.controls.values():
                if "group" in control_group:
                    control_group["group"].setVisible(False)

            if method_key and method_key in self.controls:
                self.controls[method_key]["group"].setVisible(True)

            self.method_group.update()
            self.update()
        except Exception as e:
            print(f"Error updating controls: {e}")

    def get_params(self):
        try:
            method_map = {
                "simple": "simple",
                "quantize": "quantize",
                "edge preserving": "edge",
                "dither": "dither"
            }

            current_method = self.method_combo.currentText().lower()
            method = method_map.get(current_method, "quantize")

            params = {
                "pixel_size": self.pixel_size_slider["slider"].value(),
                "method": method
            }

            if method == "quantize" and "num_colors" in self.controls["quantize"]:
                params["num_colors"] = self.controls["quantize"]["num_colors"]["slider"].value()
            elif method == "edge" and "edge_threshold" in self.controls["edge"]:
                params["edge_threshold"] = self.controls["edge"]["edge_threshold"]["slider"].value()
            elif method == "dither" and "dither_strength" in self.controls["dither"]:
                params["dither_strength"] = self.controls["dither"]["dither_strength"]["slider"].value()

            return params
        except Exception as e:
            print(f"Error getting params: {e}")
            return self.default_params.copy()
