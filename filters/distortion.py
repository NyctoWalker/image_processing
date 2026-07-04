import cv2
import numpy as np
from numba import jit, prange

# region Distortion and special effects

def apply_chromatic_aberration(img, shift=2, mode=0):
    if mode == 0:
        h, w, c = img.shape
        result = np.zeros_like(img)
        result[:-shift, :-shift, 0] = img[shift:, shift:, 0]  # B
        result[shift:, shift:, 2] = img[:-shift, :-shift, 2]  # R
        result[:, :, 1] = img[:, :, 1]  # G
    elif mode == 1:
        intensity = shift / 10
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        y, x = np.indices((h, w))

        shifts = [
            (0.0, 0.0, 1.5 * intensity),  # B
            (0.0, 0.7 * intensity, 0.0),  # G
            (1.0 * intensity, 0.0, 0.0)  # R
        ]

        result = np.zeros_like(img)
        for c in range(3):
            dx = (x - center[0]) / center[0]
            dy = (y - center[1]) / center[1]
            radius = np.sqrt(dx ** 2 + dy ** 2)

            x_shift = shifts[c][0] * radius * dx * 50
            y_shift = shifts[c][1] * radius * dy * 50
            x_map = np.clip(x + x_shift, 0, w - 1).astype(np.float32)
            y_map = np.clip(y + y_shift, 0, h - 1).astype(np.float32)

            remapped = cv2.remap(img[:, :, c], x_map, y_map, cv2.INTER_CUBIC)
            result[:, :, c] = remapped
    elif mode == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150 * 2)
        mask = cv2.dilate(edges, None, iterations=1) / 255.0

        shifted = apply_chromatic_aberration(img, shift, mode=0)
        result = img * (1 - mask[..., np.newaxis]) + shifted * mask[..., np.newaxis]
    elif mode == 3:
        h, w = img.shape[:2]
        result = img.copy()
        intensity = shift / 10

        for y in range(0, h, max(1, int(5 - intensity * 3))):
            if np.random.rand() < intensity * 0.3:
                stripe_width = np.random.randint(1, 4)
                x_start = np.random.randint(0, w - stripe_width)
                if np.random.rand() > 0.5:
                    result[y, x_start:x_start + stripe_width] = result[y, x_start:x_start + stripe_width, ::-1]
                else:
                    result[y, x_start:x_start + stripe_width] = np.random.randint(0, 256, (stripe_width, 3))

        if intensity > 0.3:
            ch_shifts = [(0, shift), (-shift // 2, 0), (shift, shift // 2)]  # B,G,R
            channels = []
            for c, (x_shift, y_shift) in enumerate(ch_shifts):
                M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                channels.append(cv2.warpAffine(img[:, :, c], M, (w, h), borderMode=cv2.BORDER_REPLICATE))
            result = cv2.merge(channels)

        noise_mask = np.random.rand(h, w) < intensity * 0.05
        result[noise_mask] = np.random.randint(0, 256, 3)
        if intensity > 0.5:
            for _ in range(int(intensity * 2)):
                y = np.random.randint(0, h)
                shift_amount = np.random.randint(-shift * 2, shift * 2 + 1)
                result[y] = np.roll(result[y], shift_amount, axis=0)
    return np.clip(result, 0, 255).astype(np.uint8)


@jit(nopython=True, fastmath=True, parallel=True)
def apply_block_corruption(result, intensity, h, w, block_size, h_blocks, w_blocks):
    corrupt_mask = np.random.rand(h_blocks, w_blocks) < intensity * 0.3
    for i in range(h_blocks):
        for j in range(w_blocks):
            if corrupt_mask[i, j]:
                y, x = i * block_size, j * block_size
                block_end_y = min(y + block_size, h)
                block_end_x = min(x + block_size, w)
                result[y:block_end_y, x:block_end_x] = \
                    np.random.randint(0, 256, (block_end_y-y, block_end_x-x, 3))
    return result


@jit(nopython=True, fastmath=True, parallel=True)
def apply_tear_lines(result, intensity, h, w):
    tear_lines = np.random.rand(h) < intensity * 0.03
    tear_shifts = np.random.randint(-int(10 * intensity), int(10 * intensity)+1, size=h)
    for y in range(h):
        if tear_lines[y]:
            shift = tear_shifts[y]
            if shift > 0:
                result[y, :shift] = result[y, -shift:]
                result[y, shift:] = result[y, :-shift]
            elif shift < 0:
                result[y, shift:] = result[y, :-shift]
                result[y, :shift] = result[y, -shift:]
    return result


def apply_distortion(img, intensity=0.5, mode=0):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    if mode == 0:  # LCD-имитация
        x_distort = x + intensity * 50 * np.sin(y / 30)
        y_distort = y + intensity * 30 * np.cos(x / 40)
        return cv2.remap(img, x_distort.astype(np.float32), y_distort.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    elif mode == 1:  # Искажение сетки пикселей
        cell_size = max(4, int(20 - intensity * 15))
        x_distort = x + intensity * 10 * np.sin((x // cell_size) * cell_size / 10) * np.sin((y // cell_size) * cell_size / 10)
        y_distort = y + intensity * 10 * np.cos((x // cell_size) * cell_size / 10) * np.cos((y // cell_size) * cell_size / 10)
        return cv2.remap(img, x_distort.astype(np.float32), y_distort.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    elif mode == 2:  # RGB distortion
        shifted = np.zeros_like(img)
        shift = int(intensity * 10)
        for c in range(3):
            x_shift = x + (c - 1) * shift * np.sin(y / 50)
            shifted[..., c] = cv2.remap(img[..., c], x_shift.astype(np.float32), y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return shifted

    elif mode == 3:  # Mission Control
        scanline_freq = max(2, int(10 - intensity * 8))
        scan_distort = intensity * 15 * np.sin(y / scanline_freq)
        x_distort = np.clip(x + scan_distort * np.sin(x / 30), 0, w - 1)
        result = cv2.remap(img, x_distort.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        scanlines = np.sin((y % scanline_freq) / scanline_freq * np.pi) * 0.3 + 0.7
        return (result * scanlines[..., None]).astype(np.uint8)

    elif mode == 4:  # Шестигранная сетка
        hex_size = max(3, int(15 - intensity * 12))
        hex_grid_x = (x + (y % 2) * hex_size / 2) // hex_size * hex_size
        hex_grid_y = (y // hex_size) * hex_size

        distort_x = hex_grid_x + hex_size / 2 * np.sin(y / 20) * intensity
        distort_y = hex_grid_y + hex_size / 2 * np.cos(x / 20) * intensity

        distorted = cv2.remap(img, distort_x.astype(np.float32), distort_y.astype(np.float32), cv2.INTER_NEAREST)
        return cv2.addWeighted(img, 1 - intensity, distorted, intensity, 0)

    elif mode == 5:  # Выгоревшая киноплёнка
        h, w = img.shape[:2]
        result = img.copy()
        intensity = max(intensity, 0.1)

        line_intensity = int(intensity * 15)
        noise = np.random.randint(-line_intensity, line_intensity, (h, w, 3))
        noise_lines = np.random.rand(h) < intensity * 0.2
        result[noise_lines] = np.clip(result[noise_lines] + noise[noise_lines], 0, 255)

        if intensity > 0.4:
            banding = np.sin(np.arange(w) / 50) * intensity * 40
            result = np.clip(result + banding[None, :, None], 0, 255)

        return result.astype(np.uint8)

    elif mode == 6:  # Магнитная лента
        h, w = img.shape[:2]
        y, x = np.indices((h, w))

        intensity_clip = max(intensity, 0.1)
        dynamic_range = 1.0 + intensity_clip * 0.8
        wave_freq = max(5.0, 30.0 - intensity_clip * 22)
        x_distort = x + intensity_clip * 20 * np.sin(y / wave_freq) * dynamic_range

        distorted = np.zeros_like(img)
        for c in range(3):
            channel_shift = int((c - 1) * intensity_clip * 3 * dynamic_range)
            distorted[..., c] = cv2.remap(img[..., c],
                                          np.clip(x_distort + channel_shift, 0, w - 1).astype(np.float32),
                                          y.astype(np.float32),
                                          cv2.INTER_LINEAR)

        if intensity_clip > 0.5:
            noise_height = min(h, int(h * (intensity_clip - 0.5) * 0.2))
            if noise_height > 0:
                noise_mask = np.linspace(0, 1, noise_height)[:, np.newaxis, np.newaxis]
                noise = np.random.randint(0, 255, (noise_height, w, 3))
                distorted[-noise_height:] = np.clip(distorted[-noise_height:] * (1 - 0.7 * noise_mask) +
                                                    noise * (0.7 * noise_mask), 0, 255)

        return distorted

    elif mode == 7:  # VHS-шумы
        result = img.copy()
        intensity = min(3.0, intensity)

        if intensity > 0.3:
            block_size = max(2, int(5 - intensity * 3))
            h_blocks, w_blocks = h // block_size, w // block_size
            result = apply_block_corruption(result, intensity, h, w, block_size, h_blocks, w_blocks)

        if intensity > 0.5:
            result = apply_tear_lines(result, intensity, h, w)

        return result

        if intensity > 0.5:
            tear_lines = np.random.rand(h) < intensity * 0.03
            tear_shifts = np.random.randint(-int(10 * intensity), int(10 * intensity), size=h)
            for y in np.where(tear_lines)[0]:
                result[y, :] = np.roll(result[y, :], tear_shifts[y], axis=0)
        return result

    elif mode == 8:  # Волновая интерференция
        x_distort = x + intensity * 20 * (np.sin(y / 30) * np.cos(x / 45))
        y_distort = y + intensity * 20 * (np.cos(x / 25) * np.sin(y / 35))
        return cv2.remap(img, x_distort.astype(np.float32), y_distort.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    elif mode == 9:  # Хиральное спиральное искажение
        center_x, center_y = w//2, h//2
        radius = np.sqrt((x-center_x)**2 + (y-center_y)**2)
        angle = np.arctan2(y-center_y, x-center_x)
        x_distort = x + intensity * 15 * np.sin(radius/20 + angle*3)
        y_distort = y + intensity * 15 * np.cos(radius/20 + angle*3)
        return cv2.remap(img, x_distort.astype(np.float32), y_distort.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    elif mode == 10:  # Тесселяционный фрактал
        x_distort = x + intensity * 40 * (np.sin(x / 50) * np.sin(y / 60))
        y_distort = y + intensity * 40 * (np.cos(x / 55) * np.cos(y / 45))
        return cv2.remap(img, x_distort.astype(np.float32), y_distort.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def apply_data_mosh(img, block_size=16, corruption_chance=0.3):
    h, w = img.shape[:2]
    result = img.copy()
    block_size = min(block_size, h // 4, w // 4)
    if block_size < 2:
        return img

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if np.random.rand() < corruption_chance:
                max_offset = min(3 * block_size, h - block_size - y, w - block_size - x)
                src_y = y + np.random.randint(-max_offset, max_offset + 1)
                src_x = x + np.random.randint(-max_offset, max_offset + 1)
                src_y = max(0, min(src_y, h - block_size - 1))
                src_x = max(0, min(src_x, w - block_size - 1))
                if src_y >= 0 and src_x >= 0 and (src_y + block_size) <= h and (src_x + block_size) <= w:
                    result[y:y + block_size, x:x + block_size] = img[src_y:src_y + block_size, src_x:src_x + block_size]
    return result


def apply_kaleidoscope(img, segments=3, mode=0, intensity=1.0, outside=0):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    y, x = np.ogrid[-center[1]:h - center[1], -center[0]:w - center[0]]

    if mode == 0:  # Пузырь
        radius = int(np.hypot(center[0], center[1]))
        circle_mask = x ** 2 + y ** 2 <= radius ** 2
        polar_img = cv2.linearPolar(img, center, radius, cv2.WARP_FILL_OUTLIERS)
        polar_w = polar_img.shape[1]

        segment_width = max(1, polar_w // max(2, segments))
        base_segment = polar_img[:, :segment_width]
        result_polar = np.zeros_like(polar_img)
        for i in range(segments):
            segment = base_segment if i % 2 == 0 else cv2.flip(base_segment, 1)
            start = i * segment_width
            end = min((i + 1) * segment_width, polar_w)
            result_polar[:, start:end] = segment[:, :end - start]

        result = cv2.linearPolar(result_polar, center, radius, cv2.WARP_INVERSE_MAP)
        circle_mask = x ** 2 + y ** 2 <= radius ** 2

    elif mode == 1:  # Срезы
        segments = max(2, min(32, segments))
        angle_step = 2 * np.pi / segments

        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(segments):
            if i % 2 == 0:
                pts = np.array([
                    center,
                    [int(center[0] + w * np.cos(i * angle_step)),
                     int(center[1] + h * np.sin(i * angle_step))],
                    [int(center[0] + w * np.cos((i + 1) * angle_step)),
                     int(center[1] + h * np.sin((i + 1) * angle_step))]
                ], dtype=np.int32)
                cv2.fillConvexPoly(mask, pts, 1)

        result = cv2.flip(img, -1)
        result = cv2.addWeighted(img, 1 - intensity, result, intensity, 0)
        circle_mask = mask == 1

    elif mode == 2:  # Радиальный взрыв
        radius = int(np.hypot(center[0], center[1]))
        segments = max(3, min(16, segments))

        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(segments):
            start_angle = 360 * i / segments
            end_angle = 360 * (i + 1) / segments
            if i % 2 == 0:
                cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 1, -1)
        polar_img = cv2.linearPolar(img, center, radius, cv2.WARP_FILL_OUTLIERS)
        polar_w = polar_img.shape[1]
        segment_width = max(1, polar_w // segments)
        result_polar = np.zeros_like(polar_img)
        for i in range(segments):
            segment = polar_img[:, i * segment_width:(i + 1) * segment_width]
            result_polar[:, i * segment_width:(i + 1) * segment_width] = (
                segment if i % 2 == 0 else cv2.flip(segment, 1))

        result = cv2.linearPolar(result_polar, center, radius, cv2.WARP_INVERSE_MAP)
        circle_mask = mask == 1

    final = img.copy()
    if mode in [0, 2]:
        final = result
        if outside == 1:
            final[~circle_mask] = 0
        elif outside == 2:
            mirrored = cv2.flip(result, 1)
            edge_mask = cv2.dilate(circle_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) - circle_mask
            final[edge_mask == 1] = mirrored[edge_mask == 1]
    else:
        if outside == 0:
            final[circle_mask] = result[circle_mask]
        elif outside == 1:
            final[circle_mask] = result[circle_mask]
            final[~circle_mask] = 0
        elif outside == 2:
            final[circle_mask] = result[circle_mask]
            mirrored = cv2.flip(result, 1)
            edge_mask = cv2.dilate(circle_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) - circle_mask
            final[edge_mask == 1] = mirrored[edge_mask == 1]
    return final


@jit(nopython=True, parallel=True, fastmath=True)
def create_lenticular_pattern(h, w, views, lens_width):
    pattern = np.zeros((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in prange(w):
            pattern[y, x] = (x // lens_width) % views
    return pattern


@jit(nopython=True, parallel=True, fastmath=True)
def apply_distortion_maps(x_map, y_map, h, w, distortion_strength):
    center_x = w // 2
    center_y = h // 2
    max_dist = max(center_x, center_y)

    for y in prange(h):
        for x in prange(w):
            nx = (x - center_x) / center_x
            ny = (y - center_y) / center_y
            r = np.sqrt(nx ** 2 + ny ** 2)
            theta = 1.0 - distortion_strength * r ** 2
            x_map[y, x] = (nx * theta * center_x) + center_x
            y_map[y, x] = (ny * theta * center_y) + center_y
    return x_map, y_map


def apply_lenticular_effect(img, views=3, lens_width=5, distortion_strength=0.3):
    h, w = img.shape[:2]

    shifts = []
    for i in range(views):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + (i * 30)) % 180
        shifts.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    lens_pattern = create_lenticular_pattern(h, w, views, lens_width)
    result = np.zeros_like(img)
    for i in range(views):
        mask = (lens_pattern == i)[..., None]
        result = np.where(mask, shifts[i], result)
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    x_map, y_map = apply_distortion_maps(x_map, y_map, h, w, distortion_strength)

    distorted = np.zeros_like(result)
    for c in range(3):
        distorted[..., c] = cv2.remap(result[..., c], x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return np.clip(distorted, 0, 255).astype('uint8')


def apply_pinch_warp(img,
               strength=0.5,
               center_x=0.5,
               center_y=0.5,
               rotation=0.0):
    h, w = img.shape[:2]
    cx, cy = center_x * w, center_y * h
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x -= cx
    y -= cy

    r = np.sqrt(x * x + y * y)
    r = np.maximum(r, 1e-6)
    factor = 1.0 / ((1.0 + strength * r / max(h, w)) + 10e-9)
    x2 = x * factor
    y2 = y * factor

    if rotation != 0:
        theta = np.deg2rad(rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = x2 * cos_t - y2 * sin_t
        yr = x2 * sin_t + y2 * cos_t
    else:
        xr, yr = x2, y2

    map_x = xr + cx
    map_y = yr + cy

    border = cv2.BORDER_WRAP
    return cv2.remap(img,
                     map_x.astype(np.float32),
                     map_y.astype(np.float32),
                     cv2.INTER_CUBIC,
                     borderMode=border)

# endregion
