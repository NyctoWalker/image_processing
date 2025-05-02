import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QInputDialog, QSlider, QDialog, QFormLayout, QDialogButtonBox,
    QCheckBox, QToolBar, QListWidgetItem, QGroupBox, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction
import cv2
import numpy as np
import time
import os

from dialogs import KernelDialog, PixelArtDialog
from image_viewer import ImageViewer
from filter_statics import apply_sepia, apply_hsb_adjustment, resize_image, pixelize_image, pixelize_kmeans, \
    pixelize_edge_preserving, pixelize_dither, apply_grayscale, apply_posterize, apply_threshold, \
    apply_bleach_bypass, apply_halftone, apply_chromatic_aberration, apply_canny_thresh, apply_ordered_dither, \
    apply_crt_effect, apply_voxel_effect, apply_blur, apply_multitone_gradient, adjust_brightness_contrast

FILTER_DEFINITIONS = {
    "HSB Adjustment": {
        "has_params": True,
        "default_params": {"hue": 0, "saturation": 100, "brightness": 100},
        "display_text": lambda p: f"HSB (H:{p['hue']}°, S:{p['saturation']}%, B:{p['brightness']}%)",
        "dialog_sliders": [
            {"label": "Оттенок:", "key": "hue", "min": -180, "max": 180, "step": 10, "value_label": lambda v: f"{v}°"},
            {"label": "Насыщенность:", "key": "saturation", "min": 0, "max": 300, "step": 20,
             "value_label": lambda v: f"{v}%"},
            {"label": "Яркость:", "key": "brightness", "min": 0, "max": 200, "step": 10,
             "value_label": lambda v: f"{v}%"}
        ],
        "apply": lambda img, params: apply_hsb_adjustment(
            img,
            params.get('hue', 0),
            params.get('saturation', 100),
            params.get('brightness', 100)
        )
    },
    "Brightness/Contrast": {
        "has_params": True,
        "default_params": {"brightness": 0, "contrast": 100},
        "display_text": lambda p: f"Яркость/Контрастность (Я {p['brightness']}, К {p['contrast']}%)",
        "dialog_sliders": [
            {"label": "Яркость:", "key": "brightness", "min": -100, "max": 100, "step": 10,
             "value_label": lambda v: str(v)},
            {"label": "Контраст:", "key": "contrast", "min": 0, "max": 200, "step": 10,
             "value_label": lambda v: f"{v}%"}
        ],
        "apply": lambda img, params: adjust_brightness_contrast(
            img,
            params.get('brightness', 0),
            params.get('contrast', 100)
        )
    },
    "Blur": {
        "has_params": True,
        "default_params": {"size": 1},
        "display_text": lambda p: f"Размытие (size: {p['size']})",
        "dialog_sliders": [
            {"label": "Размер ядра:", "key": "size", "min": 1, "max": 31,
             "value_label": lambda v: str(v), "odd_only": True}
        ],
        "apply": lambda img, params: apply_blur(img, max(1, params.get('size', 5))
        )
    },
    "Edge Detection": {
        "has_params": True,
        "default_params": {"threshold1": 50, "threshold2": 200, "kernel_size": 1, "color": 0},
        "display_text": lambda p: f"Детекция краёв ({p['threshold1']}-{p['threshold2']}, {p['kernel_size']}, цвет {p['color']})",
        "dialog_sliders": [
            {"label": "Порог 1:", "key": "threshold1", "min": 0, "max": 500, "step": 10,
             "value_label": lambda v: str(v)},
            {"label": "Порог 2:", "key": "threshold2", "min": 0, "max": 300, "step": 10,
             "value_label": lambda v: str(v)},
            {"label": "Ядро жирности:", "key": "kernel_size", "min": 1, "max": 5, "step": 1,
             "value_label": lambda v: str(v)},
            {"label": "Цвет:", "key": "color", "min": 0, "max": 1,
             "value_label": lambda v: "Нет" if v == 0 else "Да"}
        ],
        "apply": lambda img, params: apply_canny_thresh(
            img,
            max(1, params.get('threshold1', 50)),
            max(1, params.get('threshold2', 200)),
            params.get('kernel_size', 1),
            params.get('color', 0)
        )
    },
    "Invert": {
        "has_params": False,
        "display_text": lambda p: "Инверсия",
        "apply": lambda img, params: cv2.bitwise_not(img)
    },
    "Sepia": {
        "has_params": False,
        "display_text": lambda p: "Сепия",
        "apply": lambda img, params: apply_sepia(img)
    },
    "Grayscale": {
        "has_params": False,
        "display_text": lambda p: "Ч/Б (градации серого)",
        "apply": lambda img, params: apply_grayscale(img)
    },
    "Bleach": {
        "has_params": False,
        "display_text": lambda p: "Выцветание (Ч/Б с контрастом)",
        "apply": lambda img, params: apply_bleach_bypass(img)
    },
    "Custom Kernel": {
        "has_params": True,
        "default_params": {
            "kernel_size": 3,
            "kernel": np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
        },
        "display_text": lambda p: f"Кастомное ядро ({p['kernel'].shape[0]}x{p['kernel'].shape[0]})",
        "custom_dialog": True
    },
    "Pixel Art": {
        "has_params": True,
        "default_params": {
            "pixel_size": 8,
            "method": "quantize",
            "num_colors": 16,
            "edge_threshold": 30,
            "dither_strength": 50
        },
        "display_text": lambda p: f"Пиксель-арт ({p['method']}, {p['pixel_size']}px)",
        "custom_dialog": True,
        "apply": "_apply_pixel_art"
    },
    "Resize": {
        "has_params": True,
        "default_params": {"scale": 20, "interpolation": 1},
        "display_text": lambda p: f"Изменение размера ({p['scale']}%)",
        "dialog_sliders": [
            {"label": "Размер:", "key": "scale", "min": 2, "max": 100, "value_label": lambda v: f"{v*5}%"},
            {"label": "Метод:", "key": "interpolation", "min": 0, "max": 4, "step": 1,
             "value_label": lambda v: ["Nearest", "Linear", "Cubic", "Area", "Lanczos"][int(v)]},
        ],
        "apply": lambda img, params: resize_image(
            img,
            params.get('scale', 100) / 100.0 * 5.0,
            params.get('interpolation', 1)
        )
    },
    "Posterize": {
        "has_params": True,
        "default_params": {"levels": 4},
        "display_text": lambda p: f"Пастеризация ({p['levels']})",
        "dialog_sliders": [
            {"label": "Уровни:", "key": "levels", "min": 2, "max": 12, "value_label": lambda v: str(v)}
        ],
        "apply": lambda img, params: apply_posterize(img, params.get('levels', 4))
    },
    "Stepped Gradient": {
        "has_params": True,
        "default_params": {"hue": 60, "palette": 1, "color_count": 2, "darken": 5},
        "display_text": lambda p: f"Ступенчатый градиент ({p['hue']}, "
                                  f"{p['color_count']} цветов, темн. {p['darken'] / 10}), палитра {p['palette']}",
        "dialog_sliders": [
            {"label": "Оттенок:", "key": "hue", "min": 0, "max": 360, "step": 10, "value_label": lambda v: str(v)},
            {"label": "Палитра:", "key": "palette", "min": 0, "max": 7, "step": 1,
             "value_label": lambda v: ["Монохромная", "Натуральная", "Аналоговая", "Линейная",
                                       "Экстремальная линейная", "Геометрическая", "Пастель", "Холод"][int(v)]},
            {"label": "Количество цветов:", "key": "color_count", "min": 2, "max": 16, "value_label": lambda v: str(v)},
            {"label": "Фактор затемнения:", "key": "darken", "min": 0, "max": 10, "step": 1,
             "value_label": lambda v: str(v / 10)},
        ],
        "apply": lambda img, params: apply_multitone_gradient(
            img,
            params.get("hue", 60),
            params.get("palette", 1),
            params.get("color_count", 2),
            params.get("darken", 5) / 10
        )
    },
    "Threshold": {
        "has_params": True,
        "default_params": {"thresh": 128, "color": 0},
        "display_text": lambda p: f"Двоичный порог ({p['thresh']}, цвет {p['color']})",
        "dialog_sliders": [
            {"label": "Порог:", "key": "thresh", "min": 1, "max": 254, "step": 5, "value_label": lambda v: str(v)},
            {"label": "Цвет:", "key": "color", "min": 0, "max": 1,
             "value_label": lambda v: "Нет" if v == 0 else "Да"}
        ],
        "apply": lambda img, params: apply_threshold(img, params.get('thresh', 128), params.get('color', 0))
    },
    "Bayer Dithering": {
        "has_params": True,
        "default_params": {"size": 2, "color": 0},
        "display_text": lambda p: f"Дизеринг Байеса ({p['size']}, цвет {p['color']})",
        "dialog_sliders": [
            {"label": "Размер ядра:", "key": "size", "min": 1, "max": 5, "step": 1, "value_label": lambda v: str(v)},
            {"label": "Цвет:", "key": "color", "min": 0, "max": 1,
             "value_label": lambda v: "Нет" if v == 0 else "Да"},
        ],
        "apply": lambda img, params: apply_ordered_dither(img, int(2 ** params.get("size", 2)), params.get("color", 0))
    },
    "Dotted": {
        "has_params": True,
        "default_params": {"size": 10, "color": 0},
        "display_text": lambda p: f"Точки ({p['size']}, цвет {p['color']})",
        "dialog_sliders": [
            {"label": "Уровни:", "key": "size", "min": 4, "max": 20, "value_label": lambda v: str(v)},
            {"label": "Цвет:", "key": "color", "min": 0, "max": 1,
             "value_label": lambda v: "Нет" if v == 0 else "Да"},
        ],
        "apply": lambda img, params: apply_halftone(img, params.get("size", 10), preserve_color=params.get("color", 0))
    },
    "Chromatic Abberation": {
        "has_params": True,
        "default_params": {"value": 3},
        "display_text": lambda p: f"Хроматическая абберация ({p['value']})",
        "dialog_sliders": [
            {"label": "Смещение:", "key": "value", "min": 2, "max": 50, "value_label": lambda v: str(v)}
        ],
        "apply": lambda img, params: apply_chromatic_aberration(img, params.get("value", 10))
    },
    "CRT": {
        "has_params": True,
        "default_params": {"opacity": 3, "scan_space": 2, "scanline_thickness": 1, "pixel_diff": 3},
        "display_text": lambda p: f"CRT ({p['opacity']}, разм. {p['scan_space']}-{p['scanline_thickness']}, цв. {p['pixel_diff']})",
        "dialog_sliders": [
            {"label": "Интенсивность:", "key": "opacity", "min": 0, "max": 11, "value_label": lambda v: str(v)},
            {"label": "Отступы:", "key": "scan_space", "min": 2, "max": 10, "value_label": lambda v: str(v)},
            {"label": "Жирность:", "key": "scanline_thickness", "min": 1, "max": 10, "value_label": lambda v: str(v)},
            {"label": "Смещение цвета:", "key": "pixel_diff", "min": 0, "max": 30, "value_label": lambda v: str(v)}
        ],
        "apply": lambda img, params: apply_crt_effect(
            img,
            scanline_intensity=params.get("opacity", 3)/10,
            scanline_spacing=params.get("scan_space", 2),
            scanline_thickness=params.get("scanline_thickness", 1),
            pixel_glow=params.get("pixel_diff", 3)/10
        )
    },
    "Voxelize Pixels": {
        "has_params": True,
        "default_params": {"block_size": 8, "height_scale": 2},
        "display_text": lambda p: f"Вокселизация пикселей ({p['block_size']}, {p['height_scale']})",
        "dialog_sliders": [
            {"label": "Размер блока:", "key": "block_size", "min": 0, "max": 24, "value_label": lambda v: str(v)},
            {"label": "Сила тени:", "key": "height_scale", "min": 0, "max": 20, "value_label": lambda v: str(v)}
        ],
        "apply": lambda img, params: apply_voxel_effect(
            img,
            params.get("block_size", 3),
            params.get("height_scale", 3)/10
        )
    },
}

FILTER_DISPLAY_NAMES = {
    "HSB Adjustment": "Цветокоррекция HSB",
    "Brightness/Contrast": "Яркость/Контрастность",
    "Blur": "Размытие (Гаусс)",
    "Edge Detection": "Детекция краёв",
    "Invert": "Инверсия",
    "Sepia": "Сепия",
    "Grayscale": "Ч/Б (градации серого)",
    "Bleach": "Выцветание (Ч/Б с контрастом)",
    "Custom Kernel": "Кастомное ядро",
    "Pixel Art": "Пиксель-арт",
    "Resize": "Изменение размера",
    "Posterize": "Пастеризация",
    "Stepped Gradient": "Ступенчатый градиент",
    "Threshold": "Двоичный порог",
    "Bayer Dithering": "Дизеринг Байеса",
    "Dotted": "Точки",
    "CRT": "CRT-фильтр",
    "Chromatic Abberation": "Хроматическая абберация",
    "Voxelize Pixels": "Вокселизация пикселей/Вангеры :)",
    "Test": "Тест",
}


class FilterDialog(QDialog):
    preview_requested = pyqtSignal(str, dict)

    def __init__(self, filter_name, params=None, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.filter_def = FILTER_DEFINITIONS[filter_name]
        self.original_params = params if params is not None else self.filter_def.get("default_params", {}).copy()
        self.current_params = self.original_params.copy()
        self.params = self.current_params

        self.preview_enabled = True
        self.last_update_time = 0
        self.user_accepted = False
        self.initialized = False

        self.setWindowTitle(f"Настройка {filter_name}")
        self.init_ui()
        self.initialized = True

    def init_ui(self):
        layout = QFormLayout(self)

        if self.filter_def.get("custom_dialog", False):
            self.init_custom_dialog(layout)
        elif self.filter_def["has_params"]:
            self.init_sliders(layout)

        self.preview_checkbox = QCheckBox("Показывать предпросмотр")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(self.toggle_preview)
        layout.addRow(self.preview_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_accept(self):
        self.user_accepted = True
        self.current_params = self.get_current_params()
        self.accept()

    def toggle_preview(self, state):
        try:
            self.preview_enabled = state == Qt.CheckState.Checked.value
            if not self.preview_enabled:
                self.preview_checkbox.blockSignals(True)
                self.preview_checkbox.setChecked(False)
                self.preview_checkbox.blockSignals(False)
                self.preview_requested.emit("", {})
            else:
                self.on_slider_changed()
        except Exception as e:
            print(f"Error in toggle_preview: {e}")

    def init_sliders(self, layout):
        self.sliders = {}
        for slider_def in self.filter_def["dialog_sliders"]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(slider_def["min"], slider_def["max"])

            _step = 2
            if "step" in slider_def:
                _step = slider_def["step"]
            slider.setSingleStep(_step)
            slider.setPageStep(_step)

            value = self.current_params.get(slider_def["key"], 0)
            if "scale" in slider_def:
                value = int(value * slider_def["scale"])
            slider.setValue(value)

            value_label = QLabel(slider_def["value_label"](value))

            slider.valueChanged.connect(lambda v, l=value_label, d=slider_def: l.setText(d["value_label"](v)))
            slider.valueChanged.connect(self.on_slider_changed)

            if slider_def.get("odd_only", False):
                slider.valueChanged.connect(lambda v, s=slider, l=value_label, d=slider_def:
                                            self.handle_odd_slider(v, s, l, d))

            layout.addRow(slider_def["label"], slider)
            layout.addRow("Значение:", value_label)
            self.sliders[slider_def["key"]] = slider

        if self.initialized:
            for key, slider in self.sliders.items():
                slider.valueChanged.emit(slider.value())

    def on_slider_changed(self):
        if not self.preview_enabled or not self.initialized:
            return

        if hasattr(self, '_update_timer'):
            self._update_timer.stop()

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._process_slider_change)
        self._update_timer.start(150)  # мс от последних изменений

    def _process_slider_change(self):
        self.current_params = self.get_current_params()
        try:
            self.preview_requested.emit(self.filter_name, self.current_params.copy())
        except Exception as e:
            print(f"Error emitting preview signal: {e}")

    def init_custom_dialog(self, layout):
        if self.filter_name == "Custom Kernel":
            self.kernel_size = self.params.get('kernel_size', 3)
            self.kernel = self.params.get('kernel', np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]))

            self.size_button = QPushButton("Изменить размер ядра")
            self.size_button.clicked.connect(self.change_kernel_size)
            layout.addRow(self.size_button)

            self.kernel_button = QPushButton("Настроить ядро")
            self.kernel_button.clicked.connect(self.edit_kernel)
            layout.addRow(self.kernel_button)
        elif self.filter_name == "Pixel Art":
            try:
                self.pixel_art_widget = PixelArtDialog(self.params, self)
                self.pixel_art_widget.preview_requested.connect(
                    lambda params: self.preview_requested.emit(self.filter_name, params)
                )
                layout.addWidget(self.pixel_art_widget)
            except Exception as e:
                print(f"Error in pixel art dialog: {e}")
                QMessageBox.warning(self, "Error", "Failed to configure pixel art settings")

    def handle_odd_slider(self, value, slider, label, slider_def):
        if value % 2 == 0:
            value = value + 1 if value < slider.maximum() else value - 1
            slider.setValue(value)
        label.setText(slider_def["value_label"](value))

    def change_kernel_size(self):
        new_size, ok = QInputDialog.getInt(
            self, "Размер ядра", "Введите размер ядра (нечетное число 3-15):",
            self.kernel_size, 3, 15, 2
        )

        if ok and new_size % 2 == 1:
            new_kernel = np.zeros((new_size, new_size))
            min_size = min(self.kernel.shape[0], new_size)
            for i in range(min_size):
                for j in range(min_size):
                    new_kernel[i, j] = self.kernel[i, j]

            self.kernel_size = new_size
            self.current_params['kernel'] = new_kernel
            self.current_params['kernel_size'] = new_size
        elif ok:
            QMessageBox.warning(self, "Ошибка", "Размер ядра должен быть нечетным числом")

    def edit_kernel(self):
        self.kernel_size = self.params.get('kernel_size', 3)

        current_kernel = self.params.get('kernel', None)
        if current_kernel is None or current_kernel.shape[0] != self.kernel_size:
            current_kernel = np.zeros((self.kernel_size, self.kernel_size))
            if self.kernel_size % 2 == 1:
                center = self.kernel_size // 2
                current_kernel[center, center] = 1.0
            self.params['kernel'] = current_kernel

        dialog = KernelDialog(self.kernel_size, current_kernel, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params['kernel'] = dialog.get_kernel()
            self.params['kernel_size'] = self.kernel_size

    def get_current_params(self):
        if self.filter_def.get("custom_dialog", False):
            if self.filter_name == "Pixel Art":
                return self.pixel_art_widget.get_params()
            else:
                return self.params.copy()

        params = {}
        for slider_def in self.filter_def["dialog_sliders"]:
            key = slider_def["key"]
            params[key] = self.sliders[key].value()
        return params

    def get_params(self):
        return self.current_params if self.user_accepted else self.original_params


class FilterApp(QMainWindow):
    PRESETS_FILE = "presets.json"
    PRESETS_SORT_ORDER = "alphabetical"  # "alphabetical", "addition"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Обработчик изображений")
        self.setGeometry(100, 100, 1200, 700)

        self.image = None
        self.filtered_image = None
        self.filters = []  # Список словарей: {'name': str, 'params': dict}

        self.cache = []
        self.dirty_flags = []

        self.preview_mode = False
        self.preview_filter_index = -1
        self.preview_filter_params = {}
        self.original_image = None

        self.init_ui()

        self.presets = {}
        self.load_presets()
        self.init_preset_ui()

# region UI
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель
        left_panel = QVBoxLayout()

        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        left_panel.addWidget(self.load_button)

        self.save_button = QPushButton("Сохранить изображение")
        self.save_button.clicked.connect(self.save_image)
        left_panel.addWidget(self.save_button)

        # Список доступных фильтров
        self.available_filters = QListWidget()
        for internal_name, display_name in FILTER_DISPLAY_NAMES.items():
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, internal_name)
            self.available_filters.addItem(item)
        self.available_filters.itemDoubleClicked.connect(self.add_filter)
        left_panel.addWidget(QLabel("Доступные фильтры (двойной клик для добавления):"))
        left_panel.addWidget(self.available_filters)

        # Список активных фильтров
        self.active_filters = QListWidget()
        self.active_filters.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.active_filters.model().rowsMoved.connect(self.filters_reordered)
        self.active_filters.itemDoubleClicked.connect(self.edit_filter)
        left_panel.addWidget(QLabel("Активные фильтры (перетаскивание для изменения порядка):"))
        left_panel.addWidget(self.active_filters)

        self.remove_button = QPushButton("Удалить выбранный фильтр")
        self.remove_button.clicked.connect(self.remove_selected_filter)
        left_panel.addWidget(self.remove_button)

        self.toggle_all_button = QPushButton("Переключить все фильтры")
        self.toggle_all_button.clicked.connect(self.toggle_all_filters)
        left_panel.addWidget(self.toggle_all_button)

        # Правая панель
        right_panel = QVBoxLayout()

        self.image_viewer = ImageViewer()
        self.image_viewer.init_status_bar(self.statusBar())

        right_panel.addWidget(self.image_viewer)

        # Добавление блоков
        main_layout.addLayout(left_panel, 30)
        main_layout.addLayout(right_panel, 70)

        # Панель инструментов
        toolbar = QToolBar("Инструменты")
        self.addToolBar(toolbar)

        # Зум
        self.zoom_in_action = QAction("Приблизить", self)
        self.zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Отдалить", self)
        self.zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        toolbar.addAction(self.zoom_out_action)

        self.fit_to_window_action = QAction("Растянуть", self)
        self.fit_to_window_action.triggered.connect(lambda: self.image_viewer.set_fit_to_window(True))
        self.fit_to_window_action.setCheckable(True)
        self.fit_to_window_action.setChecked(True)
        toolbar.addAction(self.fit_to_window_action)

        self.actual_size_action = QAction("Реальный размер(пикс.)", self)
        self.actual_size_action.triggered.connect(self.image_viewer.set_actual_size)
        toolbar.addAction(self.actual_size_action)

    def init_preset_ui(self):
        left_panel = self.centralWidget().layout().itemAt(0).layout()

        preset_group = QGroupBox("Шаблоны фильтров")
        preset_layout = QVBoxLayout()

        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Сортировка:"))
        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItems(["По алфавиту", "По порядку добавления"])
        self.sort_order_combo.setCurrentText(
            "По алфавиту" if self.PRESETS_SORT_ORDER == "alphabetical" else "По порядку добавления")
        self.sort_order_combo.currentTextChanged.connect(self.update_preset_combo)
        sort_layout.addWidget(self.sort_order_combo)
        preset_layout.addLayout(sort_layout)

        self.preset_combo = QComboBox()
        preset_layout.addWidget(self.preset_combo)

        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Название шаблона")
        preset_layout.addWidget(self.preset_name_edit)

        btn_layout = QHBoxLayout()
        self.save_preset_btn = QPushButton("Сохранить")
        self.save_preset_btn.clicked.connect(self.save_current_preset)
        btn_layout.addWidget(self.save_preset_btn)

        self.load_preset_btn = QPushButton("Загрузить")
        self.load_preset_btn.clicked.connect(self.load_selected_preset)
        btn_layout.addWidget(self.load_preset_btn)

        self.delete_preset_btn = QPushButton("Удалить")
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        btn_layout.addWidget(self.delete_preset_btn)

        preset_layout.addLayout(btn_layout)
        preset_group.setLayout(preset_layout)

        left_panel.insertWidget(4, preset_group)
        self.update_preset_combo()
# endregion

# region presets
    def save_current_preset(self):
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите название шаблона")
            return

        # Проверка существования шаблона с таким названием
        if name in self.presets:
            reply = QMessageBox.question(
                self, "Шаблон существует", f"Шаблон с именем '{name}' уже существует. Хотите перезаписать его?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Сохранение
        self.presets[name] = {
            "filters": [f.copy() for f in self.filters],
            "created_at": time.time()
        }

        for preset in self.presets.values():
            for f in preset["filters"]:
                if 'kernel' in f['params'] and isinstance(f['params']['kernel'], np.ndarray):
                    f['params']['kernel'] = f['params']['kernel'].tolist()

        self.save_presets()
        self.update_preset_combo()
        self.preset_name_edit.clear()

        QMessageBox.information(self, "Сохранено", f"Шаблон '{name}' успешно сохранён!")

    def update_preset_combo(self):
        self.preset_combo.clear()

        sort_order = self.sort_order_combo.currentText()

        if sort_order == "По алфавиту":
            items = sorted(self.presets.keys())
        else:  # "По порядку добавления"
            items = sorted(self.presets.keys(),
                           key=lambda x: self.presets[x].get("created_at", 0))

        self.preset_combo.addItems(items)

    def save_presets(self):
        presets_to_save = {
            "_metadata": {
                "sort_order": self.sort_order_combo.currentText(),
                "version": 1
            },
            "presets": {}
        }

        for name, data in self.presets.items():
            serialized_filters = []
            for f in data["filters"]:
                serialized = f.copy()
                if 'kernel' in serialized['params']:
                    if isinstance(serialized['params']['kernel'], np.ndarray):
                        serialized['params']['kernel'] = serialized['params']['kernel'].tolist()
                serialized_filters.append(serialized)

            presets_to_save["presets"][name] = {
                "filters": serialized_filters,
                "created_at": data.get("created_at", time.time())
            }

        try:
            if Path(self.PRESETS_FILE).exists():
                backup_path = Path(self.PRESETS_FILE).with_suffix('.bak')
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    Path(self.PRESETS_FILE).rename(backup_path)
                except Exception as backup_error:
                    print(f"Could not create backup: {backup_error}")

            with open(self.PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(presets_to_save, f, indent=2, ensure_ascii=False)

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось сохранить шаблоны: {str(e)}\n\nПопробуйте ещё раз."
            )
            if backup_path and backup_path.exists():
                try:
                    if Path(self.PRESETS_FILE).exists():
                        Path(self.PRESETS_FILE).unlink()
                    backup_path.rename(self.PRESETS_FILE)
                except Exception as restore_error:
                    print(f"Could not restore backup: {restore_error}")

    def load_presets(self):
        try:
            if not Path(self.PRESETS_FILE).exists():
                self.presets = {}
                return

            with open(self.PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "_metadata" in data and "sort_order" in data["_metadata"]:
                sort_text = data["_metadata"]["sort_order"]
                if sort_text == "По порядку добавления":
                    self.PRESETS_SORT_ORDER = "addition"
                else:
                    self.PRESETS_SORT_ORDER = "alphabetical"

            self.presets = {}
            for name, preset_data in data.get("presets", {}).items():
                deserialized_filters = []
                for f in preset_data["filters"]:
                    deserialized = f.copy()
                    deserialized['params'] = f['params'].copy()

                    if 'kernel' in deserialized['params'] and isinstance(deserialized['params']['kernel'], list):
                        deserialized['params']['kernel'] = np.array(deserialized['params']['kernel'])

                    deserialized_filters.append(deserialized)

                self.presets[name] = {
                    "filters": deserialized_filters,
                    "created_at": preset_data.get("created_at", time.time())
                }

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить шаблоны: {str(e)}")
            self.presets = {}

    def load_selected_preset(self):
        name = self.preset_combo.currentText()
        if not name or name not in self.presets:
            return

        was_in_preview = self.preview_mode
        preview_index = self.preview_filter_index if was_in_preview else -1

        try:
            self.filters = []
            for f in self.presets[name]["filters"]:
                filter_copy = f.copy()
                filter_copy['params'] = f['params'].copy()

                if 'kernel' in filter_copy['params'] and isinstance(filter_copy['params']['kernel'], list):
                    filter_copy['params']['kernel'] = np.array(filter_copy['params']['kernel'])

                self.filters.append(filter_copy)

            self.loaded_preset_name = name
            self.cache = []
            self.dirty_flags = [True] * len(self.filters)
            self.update_filters_list()
            self.preview_mode = False
            self.update_display()

            if was_in_preview and preview_index < len(self.filters):
                self.preview_mode = True
                self.preview_filter_index = preview_index
                self.update_display()

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка загрузки",
                f"Не удалось загрузить шаблон '{name}':\n{str(e)}"
            )
            print(f"Error loading preset {name}: {e}")
            self.filters = []
            self.update_cache_and_dirty_flags()
            self.update_filters_list()
            self.update_display()

    def delete_selected_preset(self):
        name = self.preset_combo.currentText()
        if not name or name not in self.presets:
            return

        reply = QMessageBox.question(
            self, "Удаление шаблона", f"Вы действительно хотите удалить шаблон '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            current_index = self.preset_combo.currentIndex()
            was_last_item = (current_index == self.preset_combo.count() - 1)

            del self.presets[name]
            self.save_presets()
            self.update_preset_combo()

            if self.preset_combo.count() > 0:
                if was_last_item:
                    self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
                else:
                    new_index = min(current_index, self.preset_combo.count() - 1)
                    self.preset_combo.setCurrentIndex(new_index)
            else:
                self.preset_combo.clear()

            if (hasattr(self, 'loaded_preset_name') and self.loaded_preset_name == name):
                self.filters = []
                self.update_cache_and_dirty_flags()
                self.update_filters_list()
                self.update_display()
                delattr(self, 'loaded_preset_name')
# endregion

# region caching flags
    def mark_dirty_from(self, index):
        for i in range(index, len(self.dirty_flags)):
            self.dirty_flags[i] = True

    def update_cache_and_dirty_flags(self):
        current_count = len(self.filters)
        while len(self.cache) < current_count:
            self.cache.append(None)
        while len(self.cache) > current_count:
            self.cache.pop()

        while len(self.dirty_flags) < current_count:
            self.dirty_flags.append(True)
        while len(self.dirty_flags) > current_count:
            self.dirty_flags.pop()
# endregion

# region save/load
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.original_image = self.image.copy()

                self.cache.clear()
                self.dirty_flags = [True] * len(self.filters)

                self.preview_mode = False
                self.preview_filter_params = None
                self.show_image(self.filtered_image)
                self.update_display()
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение!")

    def save_image(self):
        if self.filtered_image is None:
            QMessageBox.warning(self, "Ошибка", "Нет изображения для сохранения!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if file_path and not cv2.imwrite(file_path, cv2.cvtColor(self.filtered_image, cv2.COLOR_RGB2BGR)):
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить изображение!")
# endregion

# region filters crud
    def add_filter(self, item):
        filter_name = item.data(Qt.ItemDataRole.UserRole)
        filter_def = FILTER_DEFINITIONS[filter_name]

        # Храним текущее превью
        previous_preview_state = {
            'mode': self.preview_mode,
            'index': self.preview_filter_index,
            'name': getattr(self, 'preview_filter_name', None),
            'params': self.preview_filter_params.copy() if self.preview_filter_params else None
        }

        if not filter_def["has_params"]:
            # Если у фильтра нет параметров, пропускаем диалоговое окно
            self.filters.append({'name': filter_name, 'params': {}, 'visible': True})
            self.update_cache_and_dirty_flags()
            self.mark_dirty_from(len(self.filters) - 1)
            self.update_filters_list()
            self.update_display()
            return

        dialog = FilterDialog(filter_name, parent=self)
        dialog.preview_requested.connect(
            lambda name, params, idx=len(self.filters):
            self.handle_preview_request(name, params, idx)
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Принятие - добавляем фильтр
            params = dialog.get_params()
            self.filters.append({'name': filter_name, 'params': params, 'visible': True})
            self.update_cache_and_dirty_flags()
            self.mark_dirty_from(len(self.filters) - 1)
            self.preview_mode = False
            self.update_filters_list()
            self.update_display()
        else:
            # Отмена - восстановление предыдущего состояние
            self.preview_mode = previous_preview_state['mode']
            self.preview_filter_index = previous_preview_state['index']
            self.preview_filter_name = previous_preview_state['name']
            self.preview_filter_params = previous_preview_state['params']
            self.update_display()

    def edit_filter(self, item):
        index = self.active_filters.row(item)
        filter_data = self.filters[index]
        filter_name = filter_data['name']

        if not FILTER_DEFINITIONS[filter_name]["has_params"]:
            return

        # Сохранение превью на случай отмены
        previous_preview_state = {
            'mode': self.preview_mode,
            'index': self.preview_filter_index,
            'name': getattr(self, 'preview_filter_name', None),
            'params': self.preview_filter_params.copy() if self.preview_filter_params else None
        }

        dialog = FilterDialog(filter_name, filter_data['params'], self)
        dialog.preview_requested.connect(
            lambda name, params: self.handle_preview_request(name, params, index)
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Принятие - применение фильтра
            self.filters[index]['params'] = dialog.get_params()
            self.mark_dirty_from(index)
            self.preview_mode = False
            self.update_filters_list()
            self.update_display()
        else:
            # Отмена - восстановление состояния
            self.preview_mode = previous_preview_state['mode']
            self.preview_filter_index = previous_preview_state['index']
            self.preview_filter_name = previous_preview_state['name']
            self.preview_filter_params = previous_preview_state['params']
            self.update_display()

    def remove_selected_filter(self):
        current_row = self.active_filters.currentRow()
        if current_row >= 0:
            if self.preview_mode and self.preview_filter_index == current_row:
                self.preview_mode = False
                self.preview_filter_params = None

            self.filters.pop(current_row)
            self.preview_mode = False
            self.preview_filter_params = None

            self.update_cache_and_dirty_flags()
            self.mark_dirty_from(current_row)

            self.update_filters_list()
            self.update_display()

    def filters_reordered(self, parent, start, end, destination, row):
        item = self.filters.pop(start)
        if row > start:
            row -= 1  # Корректируем индекс, если перемещаем вниз
        self.filters.insert(row, item)
        self.dirty_flags = [True] * len(self.filters)

        self.update_display()

    def update_filters_list(self):
        self.active_filters.clear()
        for i, filter_data in enumerate(self.filters):
            name = filter_data['name']
            params = filter_data['params']
            display_text = FILTER_DEFINITIONS[name]["display_text"](params)

            item = QListWidgetItem(self.active_filters)
            widget = FilterListItem(display_text)
            widget.set_checked(filter_data.get('visible', True))
            widget.checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_filter_visibility(idx, state))

            item.setSizeHint(widget.sizeHint())
            self.active_filters.addItem(item)
            self.active_filters.setItemWidget(item, widget)
# endregion

# region visibility
    def toggle_filter_visibility(self, index, state):
        if 0 <= index < len(self.filters):
            self.filters[index]['visible'] = (state == Qt.CheckState.Checked.value)
            self.mark_dirty_from(index)
            self.update_display()

    def toggle_all_filters(self):
        if not self.filters:
            return

        all_visible = all(f.get('visible', True) for f in self.filters)
        new_state = not all_visible

        for i in range(len(self.filters)):
            self.filters[i]['visible'] = new_state
            item = self.active_filters.item(i)
            if item:
                widget = self.active_filters.itemWidget(item)
                if widget:
                    widget.set_checked(new_state)

        self.update_display()
# endregion

# region display
    def handle_preview_request(self, filter_name, params, filter_index=None):
        try:
            if params is None:
                self.preview_mode = False
                self.preview_filter_params = None
            else:
                self.preview_mode = True
                self.preview_filter_name = filter_name
                self.preview_filter_params = params.copy()
                self.preview_filter_index = filter_index if filter_index is not None else -1

            QApplication.processEvents()
            self.update_display()
        except Exception as e:
            print(f"Error handling preview: {e}")

    def update_display(self):
        if self.image is None:
            return

        self.update_cache_and_dirty_flags()
        view_state = self.image_viewer.get_viewport_state()

        try:
            current_image = self.original_image.copy()

            # Применение всех "чистых" фильтров ДО точки текущего фильтра
            for i in range(len(self.filters)):
                if not self.filters[i]['visible']:
                    continue

                if self.dirty_flags[i] or i >= len(self.cache) or self.cache[i] is None:
                    current_image = self.apply_single_filter(
                        current_image,
                        self.filters[i]['name'],
                        self.filters[i]['params']
                    )
                    self.cache[i] = current_image.copy()
                    self.dirty_flags[i] = False
                else:
                    current_image = self.cache[i].copy()

            # Превью
            if self.preview_mode and self.preview_filter_params is not None:
                if self.preview_filter_index < 0:
                    preview_base = current_image.copy()
                else:
                    # Используем кэшированные картинки если есть
                    preview_base = self.original_image.copy()
                    for i in range(self.preview_filter_index):
                        if not self.filters[i]['visible']:
                            continue

                        if i < len(self.cache) and self.cache[i] is not None and not self.dirty_flags[i]:
                            preview_base = self.cache[i].copy()
                        else:
                            preview_base = self.apply_single_filter(
                                preview_base,
                                self.filters[i]['name'],
                                self.filters[i]['params']
                            )

                # Наложение текущего фильтра если есть превью
                preview_image = self.apply_single_filter(
                    preview_base,
                    self.preview_filter_name,
                    self.preview_filter_params
                )

                # Последующие фильтры
                for i in range(self.preview_filter_index + 1, len(self.filters)):
                    if self.filters[i]['visible']:
                        preview_image = self.apply_single_filter(
                            preview_image,
                            self.filters[i]['name'],
                            self.filters[i]['params']
                        )

                current_image = preview_image

            # Обновление отображения
            self.filtered_image = current_image.copy()
            self.show_image(current_image)
            self.image_viewer.set_viewport_state(view_state)

        except Exception as e:
            print(f"Error in update_display: {e}")
            self.show_image(self.original_image.copy())

    def show_image(self, image):
        view_state = self.image_viewer.get_viewport_state()
        self.image_viewer.set_image(image)

        if (image is not None and self.image_viewer.original_pixmap and
                view_state['fit_to_window'] == False and
                image.shape[1] == self.image_viewer.original_pixmap.width() and
                image.shape[0] == self.image_viewer.original_pixmap.height()):
            self.image_viewer.set_viewport_state(view_state)

    def apply_single_filter(self, img, filter_name, params):
        in_preview = self.preview_mode and (
            filter_name == self.preview_filter_name
            or (
               0 <= self.preview_filter_index < len(self.filters)
               and
               self.filters[self.preview_filter_index]['name'] == filter_name
            )
        )
        original_size = img.shape[:2]

        try:
            if in_preview and filter_name in {"Pixel Art", "otherfilter"} and max(original_size) > 1000:
                scale = 1500 / max(original_size)
                small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                if filter_name == "Pixel Art":
                    result = self._apply_pixel_art(small_img, params)
                elif filter_name == "otherfilter":
                    result = small_img

                return cv2.resize(result, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

            filter_def = FILTER_DEFINITIONS.get(filter_name)
            if filter_def and "apply" in filter_def:
                apply_func = filter_def["apply"]
                if isinstance(apply_func, str):  # Если метод определён строкой
                    method = getattr(self, apply_func)
                    return method(img, params)
                return apply_func(img, params)
            else:
                match filter_name:
                    case "Custom Kernel":
                        kernel = params.get('kernel', np.array([
                            [0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]
                        ]))
                        if kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1:
                            return cv2.filter2D(img, -1, kernel)
                        else:
                            QMessageBox.warning(self, "Ошибка", "Ядро должно быть квадратным с нечетными размерами")
                            return img
                    case _:
                        return img

        except Exception as e:
            print(f"Error applying filter {filter_name}: {str(e)}")
            return img

    def _apply_pixel_art(self, img, params):
        method = params.get("method", "quantize")
        pixel_size = params.get("pixel_size", 8)

        if method == "simple":
            return pixelize_image(img, pixel_size)
        elif method == "quantize":
            num_colors = params.get("num_colors", 16)
            return pixelize_kmeans(img, pixel_size, num_colors)
        elif method == "edge":
            edge_threshold = params.get("edge_threshold", 30)
            return pixelize_edge_preserving(img, pixel_size, edge_threshold)
        elif method == "dither":
            dither_strength = params.get("dither_strength", 50) / 100.0
            return pixelize_dither(img, pixel_size, dither_strength)
        else:
            return pixelize_image(img, pixel_size)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)
    # endregion


class FilterListItem(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.label = QLabel(text)

        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.label)
        self.layout.addStretch()

    def set_text(self, text):
        self.label.setText(text)

    def is_checked(self):
        return self.checkbox.isChecked()

    def set_checked(self, state):
        self.checkbox.setChecked(state)


def get_optimal_workers():
    try:
        physical_cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        if physical_cores is None:
            return 2
        return max(1, physical_cores - 1)
    except:
        return 2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FilterApp()
    window.show()
    sys.exit(app.exec())
