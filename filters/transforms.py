import cv2
import numpy as np

# region Vector Field Flow

def apply_vector_field(img, style=0, scale=15, color_style=0):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(sobel_x, sobel_y)
    angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    style = int(style)
    color_style = int(color_style)

    if style == 0:
        return _vector_field_hsv(mag, angle, color_style, h, w)
    else:
        return _vector_field_displace(img, sobel_x, sobel_y, mag, scale, h, w)


def _vector_field_hsv(mag, angle, color_style, h, w):
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    if color_style == 0:
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((angle + 180.0) / 360.0 * 179).astype(np.uint8)
        hsv[:, :, 1] = np.clip(mag_norm * 1.5, 50, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(mag_norm, 30, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif color_style == 1:
        m = mag_norm.astype(np.float32) / 255.0
        r = np.clip(m * 255, 0, 255).astype(np.uint8)
        g = np.clip(m * 200.0 * (1.0 - m), 0, 255).astype(np.uint8)
        b = np.clip((1.0 - m) * 255, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    else:
        angle_norm = ((angle + 180.0) / 360.0 * 255).astype(np.uint8)
        val = np.clip(mag_norm, 30, 255).astype(np.uint8)
        gray = cv2.addWeighted(np.stack([angle_norm] * 3, axis=-1), 0.7, np.stack([val] * 3, axis=-1), 0.3, 0)
        return gray


def _vector_field_displace(img, sobel_x, sobel_y, mag, scale, h, w):
    scale_f = np.clip(scale, 1, 100) / 20.0
    mag_mean = max(np.mean(mag), 1e-6)

    dx = sobel_x / mag_mean * scale_f
    dy = sobel_y / mag_mean * scale_f

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# endregion
