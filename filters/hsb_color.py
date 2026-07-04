import cv2
import numpy as np


# region HSB/Pixel modify
def apply_hsb_adjustment(img, hue, saturation, brightness):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:] * np.array([saturation/100, brightness/100]), 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)


def apply_hsb_force_adjustment(img, hue, hue_on, saturation, sat_on, brightness, bright_on):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')

    if hue_on:
        hsv[..., 0] = (hue + 180) % 180
    if sat_on:
        hsv[..., 1] = np.clip(saturation, 0, 255)
    if bright_on:
        hsv[..., 2] = np.clip(brightness, 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)


def adjust_brightness_contrast(img, brightness, contrast):
    contrast = contrast / 100.0

    img = img.astype('float32')
    img = (img - 128) * contrast + 128 + brightness
    return np.clip(img, 0, 255).astype('uint8')


def apply_biological_vision(img, mode=0, intensity=1.0):
    intensity = np.clip(intensity, 0.1, 1.0)
    original = img.copy()
    img_float = img.astype(np.float32) / 255.0

    if mode == 0:  # "Нормальное" зрение (только гамма)
        processed = img_float
    elif mode == 1:  # Протанопия (-красный)
        lms = np.dot(img_float, [[0.2, 0.99, -0.19],
                                 [0.16, 0.79, 0.04],
                                 [0.01, -0.01, 1.0]])
        lms[:, :, 0] *= 0.0
        processed = np.dot(lms, [[1.0, 1.0, 1.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])
    elif mode == 2:  # Дейтеранопия (-зелёный))
        lms = np.dot(img_float, [[1.0, 0.0, 0.0],
                                 [0.5, 0.5, 0.0],
                                 [0.0, 0.0, 1.0]])
        lms[:, :, 1] *= 0.0
        processed = np.dot(lms, [[1.0, 0.0, 0.0],
                                 [0.7, 0.3, 0.0],
                                 [0.0, 0.0, 1.0]])
    elif mode == 3:  # Тританопия (-синий)
        lms = np.dot(img_float, [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.5, 0.5]])
        lms[:, :, 2] *= 0.0
        processed = np.dot(lms, [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.3, 0.7]])
    elif mode == 4:  # Имитация собачьего зрения
        processed = np.dot(img_float, [[0.5, 0.5, 0.0],
                                       [0.3, 0.3, 0.4],
                                       [0.0, 0.0, 0.0]])
        processed = (processed * [1.2, 1.0, 0.8]) ** 0.9
    elif mode == 5:  # Имитация кошачьего зрения
        transformed = np.dot(img_float, [[0.0, 0.5, 0.5],
                                         [0.3, 0.7, 0.0],
                                         [0.1, 0.1, 0.8]])
        luminance = cv2.cvtColor((transformed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        luminance = luminance.astype(np.float32) / 255.0
        processed = transformed * 0.7 + luminance[..., None] * 0.3
    elif mode == 6:  # Птицы, +УФ
        temp_img = (img_float * 255).astype(np.uint8)
        hsv = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32) / [180.0, 255.0, 255.0]
        hsv[:, :, 0] = (hsv[:, :, 0] * 0.7) % 1.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 1)
        temp_img = (hsv * [180, 255, 255]).astype(np.uint8)
        processed = cv2.cvtColor(temp_img, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        uv_mask = (processed.mean(axis=2) < 0.3)
        processed[uv_mask] += [0.4, 0.0, 0.6] * (1 - processed[uv_mask].mean(axis=1, keepdims=True))
    elif mode == 7:  # УФ (Пчела/Мотылёк)
        temp_img = (img_float * 255).astype(np.uint8)
        hsv = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32) / [180.0, 255.0, 255.0]
        uv_pattern = np.sin(hsv[:, :, 1] * 8 + hsv[:, :, 2] * 4) * 0.3
        transformed = np.dot(img_float, [[0.0, 0.0, 0.0],
                                         [0.5, 1.0, 0.0],
                                         [0.5, 0.0, 1.0]])
        processed = transformed * 0.7 + uv_pattern[..., None] * 0.3
    elif mode == 8:  # ИК (Змея)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        thermal = np.clip(1.0 - gray, 0, 1) ** 0.5
        processed = img_float.copy()
        processed[:, :, 0] = np.maximum(processed[:, :, 0], thermal)
        processed[:, :, 1:] *= 0.6
    elif mode == 9:  # Рак-богомол
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32) / [180.0, 255.0, 255.0]
        band1 = np.sin(hsv[:, :, 0] * 0.5) * 0.5 + 0.5
        band2 = np.cos(hsv[:, :, 1] * 2.0) * 0.3 + 0.7
        band3 = (hsv[:, :, 2] * 1.5) % 1.0
        processed = np.stack([
            band1 * band3,
            band2 * band3,
            (1.0 - np.abs(band1 - band2)) * band3
        ], axis=2)
    elif mode == 10:  # Дип си кричур :)
        processed = np.dot(img_float, [[0.8, 0, 0], [0, 0.2, 0], [0, 0, 0.9]])  # Красно-синий спектр
        speckles = (np.random.rand(*img.shape[:2]) < 0.01)[..., None]
        processed = np.where(speckles, [0.7, 0.9, 1.0], processed)  # Точки

    processed = processed * (1/1.5)  # Гамма
    output = np.clip(processed * 255, 0, 255).astype(np.uint8)
    if mode != 0:
        output = cv2.addWeighted(output, intensity, original, 1 - intensity, 0)
    return output
# endregion
