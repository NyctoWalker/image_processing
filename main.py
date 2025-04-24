import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QInputDialog, QSlider, QDialog, QFormLayout, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QAction
import cv2
import numpy as np
from dialogs import KernelDialog


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
    "Invert": {
        "has_params": False,
        "display_text": lambda p: "Invert"
    },
    "Blur": {
        "has_params": True,
        "default_params": {"size": 15},
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
    }
}


class FilterDialog(QDialog):
    def __init__(self, filter_name, params=None, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.filter_def = FILTER_DEFINITIONS[filter_name]
        self.params = params if params is not None else self.filter_def.get("default_params", {}).copy()

        self.setWindowTitle(f"Настройка {filter_name}")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        if self.filter_def.get("custom_dialog", False):
            self.init_custom_dialog(layout)
        elif self.filter_def["has_params"]:
            self.init_sliders(layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def init_sliders(self, layout):
        self.sliders = {}
        for slider_def in self.filter_def["dialog_sliders"]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(slider_def["min"], slider_def["max"])

            value = self.params.get(slider_def["key"], 0)
            if "scale" in slider_def:
                value = int(value * slider_def["scale"])
            slider.setValue(value)
            value_label = QLabel(slider_def["value_label"](slider.value()))

            if slider_def.get("odd_only", False):
                slider.valueChanged.connect(lambda v, s=slider, l=value_label, d=slider_def:
                                            self.handle_odd_slider(v, s, l, d))
            else:
                slider.valueChanged.connect(lambda v, l=value_label, d=slider_def:
                                            l.setText(d["value_label"](v)))

            layout.addRow(slider_def["label"], slider)
            layout.addRow("Значение:", value_label)
            self.sliders[slider_def["key"]] = slider

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

    def handle_odd_slider(self, value, slider, label, slider_def):
        if value % 2 == 0:
            value = value + 1 if value < slider.maximum() else value - 1
            slider.setValue(value)
        label.setText(slider_def["value_label"](value))

    def change_kernel_size(self):
        current_size = self.kernel_size
        new_size, ok = QInputDialog.getInt(
            self, "Размер ядра", "Введите размер ядра (нечетное число 3-15):",
            current_size, 3, 15, 2
        )

        if ok and new_size % 2 == 1:
            current_kernel = self.params.get('kernel', None)
            new_kernel = np.zeros((new_size, new_size))

            if current_kernel is not None:
                min_size = min(current_kernel.shape[0], new_size)
                for i in range(min_size):
                    for j in range(min_size):
                        new_kernel[i, j] = current_kernel[i, j]
            else:
                center = new_size // 2
                new_kernel[center, center] = 1.0

            self.kernel_size = new_size
            self.params['kernel'] = new_kernel
            self.params['kernel_size'] = new_size

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

    def get_params(self):
        if self.filter_def.get("custom_dialog", False):
            return {
                'kernel_size': self.kernel_size,
                'kernel': self.params['kernel']
            }

        params = {}
        for slider_def in self.filter_def["dialog_sliders"]:
            key = slider_def["key"]
            value = self.sliders[key].value()
            if "scale" in slider_def:
                value = value / slider_def["scale"]
            params[key] = value

        return params


class FilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработчик изображений")
        self.setGeometry(100, 100, 1200, 700)

        self.image = None
        self.filtered_image = None
        self.filters = []  # Список словарей: {'name': str, 'params': dict}

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
            "Invert", "Sepia", "Custom Kernel"
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

        # Правая панель
        right_panel = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.image_label)

        main_layout.addLayout(left_panel, 30)
        main_layout.addLayout(right_panel, 70)

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
        if file_path:
            success = cv2.imwrite(file_path, cv2.cvtColor(self.filtered_image, cv2.COLOR_RGB2BGR))
            if not success:
                QMessageBox.warning(self, "Ошибка", "Не удалось сохранить изображение!")

    def add_filter(self, item):
        filter_name = item.text()
        filter_def = FILTER_DEFINITIONS[filter_name]

        if not filter_def["has_params"]:
            self.filters.append({'name': filter_name, 'params': {}})
            self.update_filters_list()
            self.update_display()
            return

        dialog = FilterDialog(filter_name, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            self.filters.append({'name': filter_name, 'params': params})
            self.update_filters_list()
            self.update_display()

    def edit_filter(self, item):
        index = self.active_filters.row(item)
        filter_data = self.filters[index]
        filter_name = filter_data['name']

        if not FILTER_DEFINITIONS[filter_name]["has_params"]:
            return

        dialog = FilterDialog(filter_name, filter_data['params'], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.filters[index]['params'] = dialog.get_params()
            self.update_filters_list()
            self.update_display()

    def remove_selected_filter(self):
        current_row = self.active_filters.currentRow()
        if current_row >= 0:
            self.filters.pop(current_row)
            self.update_filters_list()
            self.update_display()

    def update_filters_list(self):
        self.active_filters.clear()
        for filter_data in self.filters:
            name = filter_data['name']
            params = filter_data['params']
            display_text = FILTER_DEFINITIONS[name]["display_text"](params)
            self.active_filters.addItem(display_text)

    def update_display(self):
        if self.image is None:
            return

        self.filtered_image = self.image.copy()
        for filter_data in self.filters:
            self.filtered_image = self.apply_single_filter(
                self.filtered_image,
                filter_data['name'],
                filter_data['params']
            )

        self.show_image(self.filtered_image)

    def show_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def apply_single_filter(self, img, filter_name, params):
        if filter_name == "HSB Adjustment":
            return self.apply_hsb_adjustment(
                img,
                params.get('hue', 0),
                params.get('saturation', 100),
                params.get('brightness', 100)
            )
        if filter_name == "Brightness":
            return self.adjust_brightness(img, params.get('value', 0))
        elif filter_name == "Invert":
            return cv2.bitwise_not(img)
        elif filter_name == "Blur":
            size = params.get('size', 5)
            return cv2.GaussianBlur(img, (size, size), 0)
        elif filter_name == "Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, params.get('threshold1', 100), params.get('threshold2', 200))
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif filter_name == "Sepia":
            return self.apply_sepia(img)
        elif filter_name == "Custom Kernel":
            kernel = params.get('kernel', np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]))
            if kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1:
                return cv2.filter2D(img, -1, kernel)
            else:
                QMessageBox.warning(self, "Ошибка",
                                    "Ядро должно быть квадратным с нечетными размерами")
                return img
        return img

    def apply_hsb_adjustment(self, img, hue, saturation, brightness):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')  # HSV

        hsv[..., 0] = (hsv[..., 0] + (hue / 2)) % 180  # H
        hsv[..., 1] = np.clip(hsv[..., 1] * (saturation / 100), 0, 255)  # S
        hsv[..., 2] = np.clip(hsv[..., 2] * (brightness / 100), 0, 255)  # V

        return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)  # RGB

    def adjust_brightness(self, img, value):
        return np.clip(img.astype('int32') + value, 0, 255).astype('uint8')

    def apply_sepia(self, img):
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        return np.clip(img.dot(sepia_filter.T), 0, 255).astype('uint8')

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FilterApp()
    window.show()
    sys.exit(app.exec())
