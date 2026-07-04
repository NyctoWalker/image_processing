import cv2
import numpy as np

# region Palette adjustment

def apply_multitone_gradient(img, hue_base=0, palette_type=0, color_count=4, darken_factor=0.5):
    darken_factor = np.clip(darken_factor, 0.0, 1.0)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v_norm = v / 255.0

    palette = []
    base_h = hue_base / 2
    cold_shift = 20

    for i in range(color_count):
        denom = max(color_count - 1, 1)
        progression = i / denom
        sigmoid_darkening = darken_factor * (1 / (1 + np.exp(-10 * (progression - 0.5))))
        sat = 180 + (70 * (1 - abs(progression - 0.5) * 2))
        val = 255 * (1 - sigmoid_darkening)

        if palette_type == 0:  # Монохромная
            hue = base_h
        elif palette_type == 1:  # Натуральная
            hue = (base_h + 30 * (1 - np.exp(-i / (color_count / 2)))) % 180
            sat = 150 + 50 * np.sin(i * np.pi / color_count)
            val = 255 * (0.7 + 0.3 * (i / (color_count - 1)) - sigmoid_darkening)
        elif palette_type == 2:  # Аналоговая
            range_degrees = min(30, 15 + 5 * color_count)
            hue = (base_h + (i - color_count // 2) * (range_degrees / color_count)) % 180
        elif palette_type == 3:  # Линейная
            hue = (base_h + i * (90 // color_count)) % 180
        elif palette_type == 4:  # Экстремальная линейная
            hue = (base_h + i * (180 // color_count)) % 180
        elif palette_type == 5:  # Геометрическая
            hue = (base_h + 90 * np.log1p(i) / np.log1p(color_count - 1)) % 180
        elif palette_type == 6:  # Пастельная
            hue = (base_h + 60 * (i / (color_count - 1))) % 180
            sat = 100 + 30 * np.sin(i * np.pi / color_count)
            val = 255 * (0.85 + 0.1 * (i / (color_count - 1)) - 0.95 * sigmoid_darkening)
        elif palette_type == 7:  # Холод
            hue = (base_h + 90 * (i / (color_count - 1)) ** 0.7) % 180
            sat = 180 - 60 * (i / (color_count - 1))
            val = 255 * (0.6 + 0.4 * (i / (color_count - 1)) - sigmoid_darkening)

        hue = (hue + cold_shift * darken_factor * (i / denom)) % 180
        palette.append((
            np.clip(hue, 0, 179),
            np.clip(sat, 0, 255),
            np.clip(val, 0, 255)
        ))
    palette_indices = np.clip((v_norm * color_count).astype(int), 0, color_count - 1)
    result_hsv = np.zeros_like(hsv)
    for i in range(color_count):
        mask = (palette_indices == i)
        result_hsv[mask, 0] = palette[i][0]
        result_hsv[mask, 1] = palette[i][1]
        result_hsv[mask, 2] = palette[i][2]

    return cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_duotone_gradient(img,
                           hue1=120, hue2=20,
                           saturation1=220, saturation2=150,
                           brightness1=100, brightness2=220,
                           blend_mode=0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_norm = gray / 255.0

    if blend_mode == 1:  # Сигмоида
        gray_norm = 1 / (1 + np.exp(-10 * (gray_norm - 0.5)))
    elif blend_mode == 2:  # Геометрическая прогрессия
        gray_norm = np.power(gray_norm, 2.0)

    color1_rgb = np.array(cv2.cvtColor(
        np.uint8([[[hue1 % 180, np.clip(saturation1, 0, 255), np.clip(brightness1, 0, 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    )
    color2_rgb = np.array(cv2.cvtColor(
        np.uint8([[[hue2 % 180, np.clip(saturation2, 0, 255), np.clip(brightness2, 0, 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    )

    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        result[..., c] = color1_rgb[c] * (1 - gray_norm) + color2_rgb[c] * gray_norm

    return result.astype('uint8')


def apply_neon_diffusion(img, intensity=0.7, glow_size=5, style=0, hue_shift=0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=glow_size)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    if style == 0:
        a = cv2.addWeighted(a, 0.8, np.roll(a, 2, axis=1), 0.2, 0)
        b = cv2.addWeighted(b, 0.8, np.roll(b, -2, axis=1), 0.2, 0)
        glow = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    else:
        a_shifted = cv2.addWeighted(a, 1, np.roll(a, 3, axis=1), -0.3, 0)
        b_shifted = cv2.addWeighted(b, 1, np.roll(b, -3, axis=1), -0.3, 0)

        glow_rgb = cv2.cvtColor(cv2.merge([l, a_shifted, b_shifted]), cv2.COLOR_LAB2RGB)
        hsv = cv2.cvtColor(glow_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        glow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(img, 1 - intensity, glow, intensity, 0)


def apply_ascii_overlay(img, cols=120, brightness=0.7, char_set=1):
    chars = [
        " .,:;+*?%S#@",
        " .'`^,;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        " -~+*x=",
        " 01",
        " -=",
        " -|",
        " .",
        "Oo. ",
        "0123456789",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ][char_set]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    cell_w = w / cols
    cell_h = cell_w * 2
    rows = int(h / cell_h)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    result = np.zeros_like(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cell_w / 20
    thickness = max(1, int(scale * 1.5))
    y_pos = int(cell_h * 0.8)

    for i in range(rows):
        for j in range(cols):
            intensity = small[i, j]
            char_idx = min(int(intensity / 255 * (len(chars) - 1)), len(chars) - 1)
            color = int(255 * brightness * (intensity / 255))
            cv2.putText(result, chars[char_idx], (int(j * cell_w), int((i + 1) * cell_h)), font, scale, (color,) * 3, thickness, cv2.LINE_AA)
    return result

# endregion
