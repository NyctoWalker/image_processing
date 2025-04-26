import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QInputDialog, QSlider, QDialog, QFormLayout, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QToolBar, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QAction
import cv2
import numpy as np
from dialogs import KernelDialog, PixelArtDialog
from image_viewer import ImageViewer
from filter_statics import apply_sepia, apply_hsb_adjustment, adjust_brightness, resize_image, \
    pixelize_image, pixelize_kmeans, pixelize_edge_preserving, pixelize_dither
import time


FILTER_DEFINITIONS = {
    "HSB Adjustment": {
        "has_params": True,
        "default_params": {"hue": 0, "saturation": 100, "brightness": 100},
        "display_text": lambda p: f"HSB (H:{p['hue']}°, S:{p['saturation']}%, B:{p['brightness']}%)",
        "dialog_sliders": [
            {"label": "Оттенок:", "key": "hue", "min": -180, "max": 180, "value_label": lambda v: f"{v}°"},
            {"label": "Насыщенность:", "key": "saturation", "min": 0, "max": 200, "value_label": lambda v: f"{v}%"},
            {"label": "Яркость:", "key": "brightness", "min": 0, "max": 200, "value_label": lambda v: f"{v}%"}
        ]
    },
    "Brightness": {
        "has_params": True,
        "default_params": {"value": 0},
        "display_text": lambda p: f"Brightness ({p['value']})",
        "dialog_sliders": [
            {"label": "Яркость:", "key": "value", "min": -100, "max": 100, "value_label": lambda v: str(v)}
        ]
    },
    "Blur": {
        "has_params": True,
        "default_params": {"size": 1},
        "display_text": lambda p: f"Blur (size: {p['size']})",
        "dialog_sliders": [
            {"label": "Размер ядра:", "key": "size", "min": 1, "max": 31,
             "value_label": lambda v: str(v), "odd_only": True}
        ]
    },
    "Edge Detection": {
        "has_params": True,
        "default_params": {"threshold1": 100, "threshold2": 200},
        "display_text": lambda p: f"Edge Detection ({p['threshold1']}-{p['threshold2']})",
        "dialog_sliders": [
            {"label": "Порог 1:", "key": "threshold1", "min": 0, "max": 500, "value_label": lambda v: str(v)},
            {"label": "Порог 2:", "key": "threshold2", "min": 0, "max": 500, "value_label": lambda v: str(v)}
        ]
    },
    "Invert": {
        "has_params": False,
        "display_text": lambda p: "Invert"
    },
    "Sepia": {
        "has_params": False,
        "display_text": lambda p: "Sepia"
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
        "display_text": lambda p: f"Custom Kernel ({p['kernel'].shape[0]}x{p['kernel'].shape[0]})",
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
        "display_text": lambda p: f"Pixel Art ({p['method']}, {p['pixel_size']}px)",
        "custom_dialog": True
    },
    "Resize": {
        "has_params": True,
        "default_params": {"scale": 100, "interpolation": "linear"},
        "display_text": lambda p: f"Resize ({p['scale']}%)",
        "dialog_sliders": [
            {"label": "Scale:", "key": "scale", "min": 10, "max": 400, "value_label": lambda v: f"{v}%"},
        ],
        "dialog_comboboxes": [
            {"label": "Interpolation:", "key": "interpolation",
             "items": ["Nearest", "Linear", "Cubic", "Area", "Lanczos"]}
        ]
    }
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

        self.setWindowTitle(f"Настройка {filter_name}")
        self.init_ui()

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
        self.accept()

    def toggle_preview(self, state):
        self.preview_enabled = state == Qt.CheckState.Checked.value
        if not self.preview_enabled:
            self.preview_requested.emit(self.filter_name, self.original_params.copy())

    def init_sliders(self, layout):
        self.sliders = {}
        for slider_def in self.filter_def["dialog_sliders"]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(slider_def["min"], slider_def["max"])

            value = self.current_params.get(slider_def["key"], 0)
            if "scale" in slider_def:
                value = int(value * slider_def["scale"])
            slider.setValue(value)
            value_label = QLabel(slider_def["value_label"](slider.value()))

            slider.valueChanged.connect(lambda v, l=value_label, d=slider_def: l.setText(d["value_label"](v)))
            slider.valueChanged.connect(self.on_slider_changed)

            if slider_def.get("odd_only", False):
                slider.valueChanged.connect(lambda v, s=slider, l=value_label, d=slider_def:
                                            self.handle_odd_slider(v, s, l, d))

            layout.addRow(slider_def["label"], slider)
            layout.addRow("Значение:", value_label)
            self.sliders[slider_def["key"]] = slider

    def on_slider_changed(self):
        if not self.preview_enabled:
            return

        current_time = time.time()
        if current_time - self.last_update_time < 0.10:  # до ~10 обновлений в секунду
            return

        self.last_update_time = current_time
        self.current_params = self.get_current_params()
        self.preview_requested.emit(self.filter_name, self.get_current_params())

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработчик изображений")
        self.setGeometry(100, 100, 1200, 700)

        self.image = None
        self.filtered_image = None
        self.filters = []  # Список словарей: {'name': str, 'params': dict}

        self.preview_mode = False
        self.preview_filter_index = -1
        self.preview_filter_params = {}
        self.original_image = None

        self.init_ui()

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
        self.available_filters.addItems([
            "HSB Adjustment", "Brightness", "Blur", "Edge Detection",
            "Invert", "Sepia", "Custom Kernel", "Pixel Art", "Resize"
        ])
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
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        toolbar.addAction(self.zoom_out_action)

        self.fit_to_window_action = QAction("Fit to Window", self)
        self.fit_to_window_action.triggered.connect(lambda: self.image_viewer.set_fit_to_window(True))
        self.fit_to_window_action.setCheckable(True)
        self.fit_to_window_action.setChecked(True)
        toolbar.addAction(self.fit_to_window_action)

        self.actual_size_action = QAction("Actual Size", self)
        self.actual_size_action.triggered.connect(self.image_viewer.set_actual_size)
        toolbar.addAction(self.actual_size_action)

    def filters_reordered(self, parent, start, end, destination, row):
        item = self.filters.pop(start)
        if row > start:
            row -= 1  # Корректируем индекс, если перемещаем вниз
        self.filters.insert(row, item)
        self.update_display()

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

    def add_filter(self, item):
        filter_name = item.text()
        filter_def = FILTER_DEFINITIONS[filter_name]

        if not filter_def["has_params"]:
            self.filters.append({'name': filter_name, 'params': {}, 'visible': True})
            self.update_filters_list()
            self.update_display()
            return

        was_in_preview = self.preview_mode

        dialog = FilterDialog(filter_name, parent=self)
        dialog.preview_requested.connect(self.handle_preview_request)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            self.filters.append({'name': filter_name, 'params': params, 'visible': True})
            self.update_filters_list()
            self.preview_mode = False
            self.update_display()
        else:
            self.preview_mode = was_in_preview
            self.update_display()

    def edit_filter(self, item):
        index = self.active_filters.row(item)
        filter_data = self.filters[index]
        filter_name = filter_data['name']

        if not FILTER_DEFINITIONS[filter_name]["has_params"]:
            return

        dialog = FilterDialog(filter_name, filter_data['params'], self)
        dialog.preview_requested.connect(lambda name, params: self.handle_preview_request(name, params, index))

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.filters[index]['params'] = dialog.get_params()
            self.update_filters_list()
            self.update_display()
        else:
            self.preview_mode = False
            self.update_display()

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

            self.update_display()
        except Exception as e:
            print(f"Error handling preview: {e}")

    def remove_selected_filter(self):
        current_row = self.active_filters.currentRow()
        if current_row >= 0:
            if self.preview_mode and self.preview_filter_index == current_row:
                self.preview_mode = False
                self.preview_filter_params = None

            self.filters.pop(current_row)

            self.update_filters_list()
            self.update_display()

    def toggle_filter_visibility(self, index, state):
        if 0 <= index < len(self.filters):
            self.filters[index]['visible'] = (state == Qt.CheckState.Checked.value)
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

    def update_display(self):
        if self.image is None:
            return

        view_state = self.image_viewer.get_viewport_state()
        self.filtered_image = self.original_image.copy()

        if not self.preview_mode:
            for filter_data in self.filters:
                if filter_data.get('visible', True):
                    self.filtered_image = self.apply_single_filter(
                        self.filtered_image,
                        filter_data['name'],
                        filter_data['params']
                    )
        else:
            if self.preview_filter_index == -1:
                for filter_data in self.filters:
                    if filter_data.get('visible', True):
                        self.filtered_image = self.apply_single_filter(
                            self.filtered_image,
                            filter_data['name'],
                            filter_data['params']
                        )
                if self.preview_filter_params is not None:
                    self.filtered_image = self.apply_single_filter(
                        self.filtered_image,
                        self.preview_filter_name,
                        self.preview_filter_params
                    )
            else:
                for i, filter_data in enumerate(self.filters):
                    if i != self.preview_filter_index and filter_data.get('visible', True):
                        self.filtered_image = self.apply_single_filter(
                            self.filtered_image,
                            filter_data['name'],
                            filter_data['params']
                        )
                if self.preview_filter_params is not None:
                    self.filtered_image = self.apply_single_filter(
                        self.filtered_image,
                        self.preview_filter_name,
                        self.preview_filter_params
                    )

        self.show_image(self.filtered_image)
        self.image_viewer.set_viewport_state(view_state)

    def show_image(self, image):
        view_state = self.image_viewer.get_viewport_state()
        self.image_viewer.set_image(image)

        if (image is not None and self.image_viewer.original_pixmap and
                view_state['fit_to_window'] == False and
                image.shape[1] == self.image_viewer.original_pixmap.width() and
                image.shape[0] == self.image_viewer.original_pixmap.height()):
            self.image_viewer.set_viewport_state(view_state)

    def apply_single_filter(self, img, filter_name, params):
        try:
            if filter_name == "HSB Adjustment":
                return apply_hsb_adjustment(
                    img,
                    params.get('hue', 0),
                    params.get('saturation', 100),
                    params.get('brightness', 100)
                )
            elif filter_name == "Brightness":
                return adjust_brightness(img, params.get('value', 0))
            elif filter_name == "Invert":
                return cv2.bitwise_not(img)
            elif filter_name == "Blur":
                size = max(1, params.get('size', 5))
                if size % 2 == 0:
                    size += 1
                return cv2.GaussianBlur(img, (size, size), 0)
            elif filter_name == "Edge Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(
                    gray,
                    max(1, params.get('threshold1', 100)),
                    max(1, params.get('threshold2', 200))
                )
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            elif filter_name == "Sepia":
                return apply_sepia(img)
            elif filter_name == "Custom Kernel":
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
            elif filter_name == "Pixel Art":
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
            elif filter_name == "Resize":
                scale = params.get('scale', 100) / 100.0
                interp = params.get('interpolation', 'linear').lower()
                return resize_image(img, scale, interp)

            return img
        except Exception as e:
            print(f"Error applying filter {filter_name}: {str(e)}")
            return img

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FilterApp()
    window.show()
    sys.exit(app.exec())
