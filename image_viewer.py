from PyQt6.QtWidgets import QLabel, QStatusBar
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QMouseEvent, QWheelEvent


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

        self.zoom_factor = 1.0
        self.max_zoom = 10.0
        self.min_zoom = 0.5
        self.pan_offset = QPointF(0, 0)
        self.pan_start = QPointF(0, 0)
        self.is_panning = False
        self.fit_to_window = True
        self.original_pixmap = None
        self.image_rect = QRect()

        self.status_bar = None
        self.image_size_label = None
        self.mouse_pos_label = None
        self.zoom_label = None

    def __del__(self):
        self.original_pixmap = None

    def init_status_bar(self, status_bar):
        self.status_bar = status_bar
        self.image_size_label = QLabel()
        self.mouse_pos_label = QLabel()
        self.zoom_label = QLabel()
        self.status_bar.addPermanentWidget(self.image_size_label)
        self.status_bar.addPermanentWidget(self.mouse_pos_label)
        self.status_bar.addPermanentWidget(self.zoom_label)

    def set_image(self, image):
        if image is not None:
            old_width = self.original_pixmap.width() if self.original_pixmap else 0
            old_height = self.original_pixmap.height() if self.original_pixmap else 0

            h, w, ch = image.shape
            q_img = QImage(image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            new_pixmap = QPixmap.fromImage(q_img)

            if old_width != w or old_height != h:
                self.reset_view()

            self.original_pixmap = new_pixmap
            self.update_display()
            self.update_status_bar()
        else:
            self.original_pixmap = None
            self.clear()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.fit_to_window = True
        self.update_display()

    def update_status_bar(self):
        if self.original_pixmap is not None:
            size = self.original_pixmap.size()
            self.image_size_label.setText(f"Размер: {size.width()}x{size.height()} пикс.")
            self.zoom_label.setText(f"Масштаб: {self.zoom_factor:.1f}x")
        else:
            self.image_size_label.setText("Размер: N/A")
            self.mouse_pos_label.setText("Позиция: N/A")
            self.zoom_label.setText("Масштаб: N/A")

# region Controls
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.original_pixmap is None:
            return

        try:
            pos = event.position()  # This is QPointF
            img_pos = self.widget_to_image_pos(pos)
            if img_pos:
                self.mouse_pos_label.setText(f"Позиция: {int(img_pos.x()+1)}, {int(img_pos.y()+1)}")
            else:
                self.mouse_pos_label.setText("Позиция: Вне картинки")

            if self.is_panning and not self.fit_to_window:
                current_pos = event.position()  # Keep as QPointF
                delta = current_pos - self.pan_start
                self.pan_start = current_pos
                self.pan_offset = QPointF(self.pan_offset.x() + delta.x(),
                                          self.pan_offset.y() + delta.y())
                self.ensure_visible()
                self.update_display()

        except Exception as e:
            print(f"Error in mouseMoveEvent: {e}")

    def widget_to_image_pos(self, widget_pos):
        if not self.original_pixmap:
            return None

        if self.fit_to_window:
            # Для режима растягивания
            if not self.image_rect.isValid():
                return None
            img_x = (widget_pos.x() - self.image_rect.x()) * (self.original_pixmap.width() / self.image_rect.width())
            img_y = (widget_pos.y() - self.image_rect.y()) * (self.original_pixmap.height() / self.image_rect.height())
        else:
            # Для зума
            img_x = (widget_pos.x() - self.pan_offset.x()) / self.zoom_factor
            img_y = (widget_pos.y() - self.pan_offset.y()) / self.zoom_factor

        # Проверка границ
        if (0 <= img_x < self.original_pixmap.width() and
                0 <= img_y < self.original_pixmap.height()):
            return QPointF(img_x, img_y)
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if self.original_pixmap is None:
            return

        try:
            if event.button() == Qt.MouseButton.LeftButton and not self.fit_to_window:
                self.is_panning = True
                self.pan_start = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        except Exception as e:
            print(f"Error in mousePressEvent: {e}")

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        if self.original_pixmap is None:
            return

        zoom_center = event.position()
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom_in(zoom_center)
        elif delta < 0:
            self.zoom_out(zoom_center)

    def zoom_in(self, zoom_center=None):
        if self.original_pixmap is None or self.zoom_factor >= self.max_zoom:
            return

        self.zoom_factor = min(self.zoom_factor + 0.5, self.max_zoom)
        self.fit_to_window = False

        if zoom_center:
            img_pos = self.widget_to_image_pos(zoom_center)
            if img_pos:
                new_x = zoom_center.x() - img_pos.x() * self.zoom_factor
                new_y = zoom_center.y() - img_pos.y() * self.zoom_factor
                self.pan_offset = QPoint(int(new_x), int(new_y))
        else:
            view_center = QPoint(self.width() // 2, self.height() // 2)
            img_pos = self.widget_to_image_pos(view_center)
            if img_pos:
                new_x = view_center.x() - img_pos.x() * self.zoom_factor
                new_y = view_center.y() - img_pos.y() * self.zoom_factor
                self.pan_offset = QPoint(int(new_x), int(new_y))

        self.ensure_visible()
        self.update_display()
        self.update_status_bar()

    def zoom_out(self, zoom_center=None):
        if self.original_pixmap is None or self.zoom_factor <= self.min_zoom:
            return

        self.zoom_factor = max(self.zoom_factor - 0.5, self.min_zoom)
        self.fit_to_window = False

        if zoom_center:
            img_pos = self.widget_to_image_pos(zoom_center)
            if img_pos:
                new_x = zoom_center.x() - img_pos.x() * self.zoom_factor
                new_y = zoom_center.y() - img_pos.y() * self.zoom_factor
                self.pan_offset = QPoint(int(new_x), int(new_y))
        else:
            view_center = QPoint(self.width() // 2, self.height() // 2)
            img_pos = self.widget_to_image_pos(view_center)
            if img_pos:
                new_x = view_center.x() - img_pos.x() * self.zoom_factor
                new_y = view_center.y() - img_pos.y() * self.zoom_factor
                self.pan_offset = QPoint(int(new_x), int(new_y))

        self.ensure_visible()
        self.update_display()
        self.update_status_bar()
# endregion

    def ensure_visible(self):
        if self.fit_to_window or not self.original_pixmap:
            return

        scaled_width = int(self.original_pixmap.width() * self.zoom_factor)
        scaled_height = int(self.original_pixmap.height() * self.zoom_factor)

        max_x = max(0, scaled_width - self.width())
        max_y = max(0, scaled_height - self.height())

        min_x = min(0, self.width() - scaled_width)
        min_y = min(0, self.height() - scaled_height)

        x = max(min(self.pan_offset.x(), max_x), min_x)
        y = max(min(self.pan_offset.y(), max_y), min_y)
        self.pan_offset = QPointF(x, y)

        if scaled_width < self.width():
            self.pan_offset.setX((self.width() - scaled_width) / 2)
        if scaled_height < self.height():
            self.pan_offset.setY((self.height() - scaled_height) / 2)

    def get_viewport_state(self):
        return {
            'zoom': self.zoom_factor,
            'pan_offset': self.pan_offset,
            'fit_to_window': self.fit_to_window
        }

    def set_viewport_state(self, state):
        if state is None:
            return

        self.zoom_factor = state.get('zoom', 1.0)
        self.pan_offset = state.get('pan_offset', QPointF(0, 0))
        self.fit_to_window = state.get('fit_to_window', True)
        self.update_display()
        self.update_status_bar()

    def get_zoom(self):
        return self.zoom_factor

    def get_fit_to_window(self):
        return self.fit_to_window

    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.fit_to_window = False
        self.update_display()
        self.update_status_bar()

    def set_fit_to_window(self, fit=True):
        self.fit_to_window = fit
        if fit:
            self.pan_offset = QPoint(0, 0)
        self.update_display()
        self.update_status_bar()

    def set_actual_size(self):
        self.fit_to_window = False
        self.zoom_factor = 1.0

        if self.original_pixmap:
            self.pan_offset = QPointF(
                (self.width() - self.original_pixmap.width()) / 2,
                (self.height() - self.original_pixmap.height()) / 2
            )

        self.update_display()
        self.update_status_bar()

    def update_display(self):
        if self.original_pixmap is None:
            self.clear()
            return

        try:
            scaled_width = int(self.original_pixmap.width() * self.zoom_factor)
            scaled_height = int(self.original_pixmap.height() * self.zoom_factor)
            pan_x = int(self.pan_offset.x())
            pan_y = int(self.pan_offset.y())

            result_pixmap = QPixmap(self.size())
            result_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(result_pixmap)

            try:
                self.image_rect = QRect(pan_x, pan_y, scaled_width, scaled_height)

                if self.fit_to_window:
                    scaled_pix = self.original_pixmap.scaled(
                        self.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.image_rect = QRect(
                        int((self.width() - scaled_pix.width()) / 2),
                        int((self.height() - scaled_pix.height()) / 2),
                        scaled_pix.width(),
                        scaled_pix.height()
                    )
                    painter.drawPixmap(self.image_rect, scaled_pix, scaled_pix.rect())
                else:
                    src_x = max(0, int(-pan_x / self.zoom_factor))
                    src_y = max(0, int(-pan_y / self.zoom_factor))
                    src_width = min(
                        self.original_pixmap.width() - src_x,
                        int((self.width() - max(0, pan_x)) / self.zoom_factor)
                    )
                    src_height = min(
                        self.original_pixmap.height() - src_y,
                        int((self.height() - max(0, pan_y)) / self.zoom_factor)
                    )

                    if src_width > 0 and src_height > 0:
                        dst_x = max(0, pan_x)
                        dst_y = max(0, pan_y)

                        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                        painter.drawPixmap(
                            QRect(
                                dst_x,
                                dst_y,
                                int(src_width * self.zoom_factor),
                                int(src_height * self.zoom_factor)
                            ),
                            self.original_pixmap,
                            QRect(src_x, src_y, src_width, src_height)
                        )

            finally:
                painter.end()

            self.setPixmap(result_pixmap)

        except Exception as e:
            print(f"Error in update_display: {e}")
