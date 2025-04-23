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


class KernelDialog(QDialog):
    def __init__(self, size=3, kernel=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка ядра свертки")
        self.size = size
        self.kernel = kernel if kernel is not None else np.zeros((size, size))

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


class FilterDialog(QDialog):
    def __init__(self, filter_name, params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Настройка {filter_name}")
        self.filter_name = filter_name
        self.params = params or {}
        self.value_label = None

        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        if self.filter_name == "Brightness":
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(-100, 100)
            self.slider.setValue(self.params.get('value', 0))
            self.value_label = QLabel(str(self.slider.value()))
            self.slider.valueChanged.connect(self.update_brightness_label)
            layout.addRow("Яркость:", self.slider)
            layout.addRow("Значение:", self.value_label)

        elif self.filter_name == "Contrast":
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 300)
            self.slider.setValue(int(self.params.get('value', 1.0) * 100))
            self.value_label = QLabel(f"{self.slider.value() / 100:.2f}")
            self.slider.valueChanged.connect(self.update_contrast_label)
            layout.addRow("Контраст:", self.slider)
            layout.addRow("Значение:", self.value_label)

        elif self.filter_name == "Blur":
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(1, 31)
            self.slider.setValue(self.params.get('size', 15))
            self.slider.setTickInterval(2)

            self.value_label = QLabel(f"{self.slider.value()}")
            self.slider.valueChanged.connect(self.on_slider_value_changed)
            layout.addRow("Размер ядра:", self.slider)
            layout.addRow("Значение:", self.value_label)

        elif self.filter_name == "Edge Detection":
            self.threshold1 = QSlider(Qt.Orientation.Horizontal)
            self.threshold1.setRange(0, 500)
            self.threshold1.setValue(self.params.get('threshold1', 100))
            layout.addRow("Порог 1:", self.threshold1)

            self.threshold2 = QSlider(Qt.Orientation.Horizontal)
            self.threshold2.setRange(0, 500)
            self.threshold2.setValue(self.params.get('threshold2', 200))
            layout.addRow("Порог 2:", self.threshold2)

        elif self.filter_name == "Custom Kernel":
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

        # Для Sepia и Invert не добавляем никаких элементов управления

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_slider_value_changed(self, value):
        if value % 2 == 0:
            value = value + 1 if value < self.slider.maximum() else value - 1
            self.slider.setValue(value)
        self.value_label.setText(str(value))

    def update_brightness_label(self, value):
        self.value_label.setText(str(value))

    def update_contrast_label(self, value):
        self.value_label.setText(f"{value / 100:.2f}")

    def change_kernel_size(self):
        size, ok = QInputDialog.getInt(
            self, "Размер ядра", "Введите размер ядра (3, 5, 7):",
            self.kernel_size, 3, 7, 2
        )
        if ok:
            self.kernel_size = size

    def edit_kernel(self):
        current_kernel = self.params.get('kernel', np.zeros((self.kernel_size, self.kernel_size)))
        dialog = KernelDialog(self.kernel_size, current_kernel, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params['kernel'] = dialog.get_kernel()

    def get_params(self):
        if self.filter_name == "Brightness":
            return {'value': self.slider.value()}
        elif self.filter_name == "Contrast":
            return {'value': self.slider.value() / 100}
        elif self.filter_name == "Blur":
            return {'size': self.slider.value()}
        elif self.filter_name == "Edge Detection":
            return {
                'threshold1': self.threshold1.value(),
                'threshold2': self.threshold2.value()
            }
        elif self.filter_name == "Custom Kernel":
            return {
                'kernel_size': self.kernel_size,
                'kernel': self.kernel
            }
        return {}


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
            "Brightness", "Contrast", "Invert", "Blur",
            "Edge Detection", "Sepia", "Custom Kernel"
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

        self.apply_button = QPushButton("Применить фильтры")
        self.apply_button.clicked.connect(self.apply_filters)
        left_panel.addWidget(self.apply_button)

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

        # Для фильтров без параметров сразу добавляем
        if filter_name in ["Invert", "Sepia"]:
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

        # Для фильтров без параметров ничего не делаем
        if filter_data['name'] in ["Invert", "Sepia"]:
            return

        dialog = FilterDialog(filter_data['name'], filter_data['params'], self)
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

            if name == "Brightness":
                text = f"{name} ({params.get('value', 0)})"
            elif name == "Contrast":
                text = f"{name} (x{params.get('value', 1.0):.1f})"
            elif name == "Blur":
                text = f"{name} (size: {params.get('size', 5)})"
            elif name == "Edge Detection":
                text = f"{name} ({params.get('threshold1', 100)}-{params.get('threshold2', 200)})"
            elif name == "Custom Kernel":
                size = params.get('kernel', np.zeros((3, 3))).shape[0]
                text = f"{name} ({size}x{size})"
            else:
                text = name

            self.active_filters.addItem(text)

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

        h, w, ch = self.filtered_image.shape
        bytes_per_line = ch * w
        q_img = QImage(self.filtered_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def apply_single_filter(self, img, filter_name, params):
        if filter_name == "Brightness":
            return self.adjust_brightness(img, params.get('value', 0))
        elif filter_name == "Contrast":
            return self.adjust_contrast(img, params.get('value', 1.0))
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
            return cv2.filter2D(img, -1, kernel)
        return img

    def apply_filters(self):
        self.update_display()

    def adjust_brightness(self, img, value):
        return np.clip(img.astype('int32') + value, 0, 255).astype('uint8')

    def adjust_contrast(self, img, value):
        return np.clip(img.astype('float') * value, 0, 255).astype('uint8')

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
