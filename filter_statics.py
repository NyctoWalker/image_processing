import cv2
import numpy as np
from numba import jit


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
# endregion


# region Simple filters
def apply_blur(img, size, variation=1):
    size = size + 1 - size % 2
    if variation == 0:
        return cv2.GaussianBlur(img, (size, size), 0)
    elif variation == 1:
        return cv2.medianBlur(img, size)
    else:
        addition = 2 * size
        return cv2.bilateralFilter(img, size, 130 - addition, 130 - addition)


def apply_sepia(img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    return cv2.transform(img, sepia_filter).clip(0, 255).astype('uint8')


def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_posterize(img, levels=4):
    factor = 256 // levels
    return np.clip((img // factor) * factor, 0, 255).astype('uint8')


def apply_bleach_bypass(img):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    desat = (0.7 * gray + 0.3 * img.mean(axis=2))
    high_contrast = np.clip((desat - 128) * 1.5 + 128, 0, 255)
    return np.stack((high_contrast,)*3, axis=-1).astype('uint8')
# endregion


# region Edges and Shading
def apply_canny_thresh(img, threshold1=100, threshold2=250, kernel_size=1, preserve_color=0):
    fine_high = 150  # На практике не сильно влияет на результат
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_fine = cv2.Canny(blurred, threshold1, fine_high)
    edges_coarse = cv2.Canny(blurred, threshold2, threshold1)
    edges = cv2.bitwise_or(edges_fine, edges_coarse)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    if preserve_color == 1:
        result = img.copy()
        result[edges == 255] = 0
        return result
    else:
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_threshold(img, thresh=128, preserve_color=0):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    mask = (gray > thresh)

    if preserve_color == 1:
        result = np.zeros_like(img)
        result[mask] = img[mask]
        return result
    else:
        bw = mask * 255
        return np.stack((bw,) * 3, axis=-1).astype('uint8')


def apply_halftone(img, dot_size=4, max_dot_ratio=0.8, preserve_color=0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height, width = gray.shape

    if preserve_color == 1:
        output = img.copy()
    else:
        output = np.ones((height, width, 3), dtype=np.uint8) * 255

    spacing = max(2, dot_size)
    max_dot_ratio = min(0.9, max(0.1, max_dot_ratio))

    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            center_x = min(x + spacing // 2, width - 1)
            center_y = min(y + spacing // 2, height - 1)
            brightness = gray[center_y, center_x]

            radius = int((1 - brightness) * (spacing // 2 * max_dot_ratio))
            if radius > 0:
                cv2.circle(output, (center_x, center_y), radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)

    return output


def apply_ordered_dither(img, bayer_size=4, preserve_color=0):
    """Bayer matrix dithering"""
    def generate_bayer(n):
        if n == 1:
            return np.array([[0]])
        m = generate_bayer(n // 2)
        return np.block([[4 * m, 4 * m + 2],
                         [4 * m + 3, 4 * m + 1]]) / (n * n)

    bayer_size = max(2, min(32, 1 << (bayer_size - 1).bit_length()))
    bayer_matrix = generate_bayer(bayer_size) * 255

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    repeat_h = (h + bayer_size - 1) // bayer_size
    repeat_w = (w + bayer_size - 1) // bayer_size
    tiled_bayer = np.tile(bayer_matrix, (repeat_h, repeat_w))[:h, :w]

    mask = (gray > tiled_bayer).astype(np.uint8)

    if preserve_color == 1:
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[mask == 1] = img[mask == 1]
        return result
    else:
        return cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)


def apply_pencil_sketch(img, ksize=21, sigma=3, gamma=0.5, preserve_color=0, color_intensity=0.7):
    ksize = ksize + 1 - ksize % 2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma).astype('float32')

    sketch = 255 - cv2.divide(gray, blur + 1e-6, scale=256)
    sketch = np.clip(sketch * gamma, 0, 255).astype('uint8')

    if preserve_color == 1:
        sketch_3ch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB).astype('float32') / 255.0
        blended = img.astype('float32') * (1 - sketch_3ch * color_intensity)
        result = cv2.addWeighted(blended, 0.9, img.astype('float32'), 0.1, 0)
        return np.clip(result, 0, 255).astype('uint8')
    else:
        return cv2.cvtColor(255 - sketch, cv2.COLOR_GRAY2RGB)


def apply_stochastic_diffusion(img, grain_size=0.15, intensity=0.1, preserve_color=0, method=0, threshold_adj=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    density = grain_size

    if method == 0:  # Шахматы
        pattern = np.sin(density * x) * np.cos(density * y)
    elif method == 1:  # Волны
        pattern = np.sin(density * np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2))
    elif method == 2:  # Штрихи
        pattern = np.sin(density * (x + y))  # Если поменять знак x/y, поменяется направление полос
    elif method == 3:  # Спираль
        angle = np.arctan2(y - h / 2, x - w / 2)
        radius = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
        pattern = np.sin(density * radius + angle * 5)
    elif method == 4:  # Ячейки Вороного (~)
        points_x = np.random.randint(0, w, 10)
        points_y = np.random.randint(0, h, 10)
        dist = np.zeros((h, w))
        for px, py in zip(points_x, points_y):
            dist += np.sqrt((x - px) ** 2 + (y - py) ** 2)
        pattern = np.sin(density * dist)
    elif method == 5:  # Повёрнутые шахматы
        hex_x = density * (x * 0.5 + y * np.sqrt(3) / 2)
        hex_y = density * (y * 0.5 - x * np.sqrt(3) / 2)
        pattern = (np.sin(hex_x) + np.cos(hex_y)) * 0.5
    elif method == 6:  # Шум
        phase = np.random.rand(h, w) * 2 * np.pi
        pattern = np.sin(density * (x * np.cos(phase) + y * np.sin(phase)))

    threshold = ((pattern - pattern.min()) * 255 / (pattern.max() - pattern.min())) * threshold_adj
    if preserve_color == 1:
        mask = (gray > threshold).astype(np.uint8)
        result = img.copy()
        result[mask == 0] = result[mask == 0] * (1 - intensity)
        return result
    else:
        dithered = (gray > threshold).astype(np.uint8) * 255
        return cv2.cvtColor(dithered, cv2.COLOR_GRAY2RGB)


@jit(nopython=True, fastmath=True)
def apply_ink_bleed(working, output, kernel, bleed_radius, h, w, threshold):
    kernel_size = kernel.shape[0]
    for y in range(h):
        for x in range(w):
            old_val = working[y, x]
            new_val = 255.0 if old_val > threshold else 0.0
            output[y, x] = new_val
            error = old_val - new_val

            if error != 0:
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        yy = y + (ky - bleed_radius)
                        xx = x + (kx - bleed_radius)
                        if 0 <= yy < h and 0 <= xx < w:
                            working[yy, xx] += error * kernel[ky, kx]
    return working, output


def ink_bleed_dither(img, bleed_radius=2, preserve_color=0, threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    kernel_size = bleed_radius * 2 + 1
    y, x = np.ogrid[-bleed_radius:bleed_radius + 1, -bleed_radius:bleed_radius + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (0.5 * bleed_radius ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)

    working = gray.astype(np.float32)
    output = np.zeros_like(working)
    working, output = apply_ink_bleed(working, output, kernel, bleed_radius, h, w, threshold)
    result = np.clip(output, 0, 255).astype(np.uint8)

    if preserve_color:
        mask = (result < 128)[..., None]
        return np.where(mask, 0, img).astype(np.uint8)
    else:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def cellular_dither(img, cell_density=0.005, preserve_color=0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if preserve_color == 0 else np.mean(img, axis=2)
    h, w = gray.shape

    small_h, small_w = int(h * cell_density ** 0.5), int(w * cell_density ** 0.5)
    points = np.column_stack((
        np.random.randint(0, h, small_h * small_w),
        np.random.randint(0, w, small_h * small_w)
    ))

    blank = np.zeros((h, w), dtype=np.uint8)
    for y, x in points:
        blank[y, x] = 255
    _, dist = cv2.distanceTransformWithLabels(255 - blank, cv2.DIST_L2, 5)

    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dithered = (gray > dist_normalized).astype(np.uint8) * 255

    if preserve_color:
        return np.where(dithered[..., None], img, 0)
    else:
        return cv2.cvtColor(dithered, cv2.COLOR_GRAY2RGB)
# endregion


# region Special
def resize_image(img, scale_factor=1.0, method=1):
    inter = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4
    ]
    inter = inter[method]

    if scale_factor != 1.0:
        return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=inter)
    return img


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


def apply_crt_effect(img, scanline_intensity=0.3, scanline_spacing=2,
                     scanline_thickness=1, pixel_glow=0.2):
    h, w = img.shape[:2]
    scanlines = np.zeros((h, w))
    for i in range(0, h, scanline_spacing):
        for t in range(scanline_thickness):
            if i + t < h:
                scanlines[i + t, :] = scanline_intensity * (1 - t * 0.2)

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8, sigmaY=0.3)
    result = img * (1 - scanlines[..., None]) + blurred * scanlines[..., None] * 0.3
    result = result * (1 + pixel_glow * 0.5)
    return np.clip(result, 0, 255).astype('uint8')


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


def apply_oil(img, stride=15, scale=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    result = img.copy()
    h, w = gray.shape

    for _ in range(5):
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                dx = sobel_x[y, x]
                dy = sobel_y[y, x]
                mag = np.sqrt(dx ** 2 + dy ** 2)

                if mag > 5:
                    nx, ny = dx / mag, dy / mag
                    x2, y2 = int(x + nx * scale), int(y + ny * scale)
                    if 0 <= x < w - scale and 0 <= y < h - scale and 0 <= x2 < w - scale and 0 <= y2 < h - scale:
                        src_region = result[y:y + scale, x:x + scale]
                        dst_region = result[y2:y2 + scale, x2:x2 + scale]
                        if src_region.shape == dst_region.shape:
                            result[y:y + scale, x:x + scale] = cv2.addWeighted(
                                    src_region, 0.7,
                                    dst_region, 0.3, 0
                                )
    return cv2.medianBlur(result, 11)


def vector_field_flow(img, stride=15, scale=3, line_color=(0, 0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    result = img.copy()
    h, w = gray.shape

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            dx = sobel_x[y, x]
            dy = sobel_y[y, x]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            if mag > 10:
                nx, ny = dx / mag, dy / mag
                x2 = int(x + nx * stride * scale)
                y2 = int(y + ny * stride * scale)
                cv2.arrowedLine(result, (x, y), (x2, y2), line_color, 1, tipLength=0.3)

    return result
# endregion


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
        "Oo. "
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


# region Near-pixelizing
def apply_voxel_effect(img, block_size=8, height_scale=0.5, light_angle=45, ambient=0.3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    heightmap = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    heightmap = heightmap ** 0.5
    heightmap = heightmap * block_size * height_scale

    sobel_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)

    light_angle_rad = np.radians(light_angle)
    light_dir = np.array([np.cos(light_angle_rad), np.sin(light_angle_rad), 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)

    normal = np.dstack((-sobel_x, -sobel_y, np.ones_like(heightmap)))
    normal_norm = normal / np.linalg.norm(normal, axis=2, keepdims=True)

    diffuse = np.dot(normal_norm.reshape(-1, 3), light_dir).reshape(h, w)
    diffuse = np.clip(diffuse, 0, 1)
    shading = ambient + (1 - ambient) * diffuse

    result = img * shading[..., None]
    return np.clip(result, 0, 255).astype('uint8')


def glitchy_pixelation(img, base_size=4, channel_shift=3):
    r = pixelize_image(img[..., 0], base_size + channel_shift)
    g = pixelize_image(img[..., 1], base_size)
    b = pixelize_image(img[..., 2], base_size - channel_shift)
    return cv2.merge((r, g, b))
# endregion


# region Pixelizing
def pixelize_image(img, pixel_size=8):
    h, w = img.shape[:2]

    small_h, small_w = h // pixel_size, w // pixel_size
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    pixel_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixel_img


def pixelize_kmeans(img, pixel_size=8, num_colors=16):
    h, w = img.shape[:2]

    if pixel_size > 1:
        small_h, small_w = h // pixel_size, w // pixel_size
        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small_img = img.copy()

    # Color quantization K-means
    pixels = small_img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(small_img.shape)

    return cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelize_edge_preserving(img, pixel_size=8, edge_threshold=30):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 3)

    if pixel_size > 1:
        small_h, small_w = h // pixel_size, w // pixel_size
        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_edges = cv2.resize(edges, (small_w, small_h)) > 0
    else:
        small_img = img.copy()
        small_edges = edges > 0

    filtered = cv2.medianBlur(small_img, 3)
    result = np.where(small_edges[..., None], small_img, filtered)

    return cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelize_dither(img, pixel_size=8, dither_strength=0.5):
    """Floyd-Steinberg dithering"""
    h, w = img.shape[:2]

    small_h, small_w = h // pixel_size, w // pixel_size
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)

    dithered = np.zeros_like(gray)
    for y in range(small_h - 1):
        for x in range(1, small_w - 1):
            old_pixel = gray[y, x]
            new_pixel = 255 * (old_pixel > 127)
            dithered[y, x] = new_pixel
            error = (old_pixel - new_pixel) * dither_strength

            gray[y, x + 1] += error * 7 / 16
            gray[y + 1, x - 1] += error * 3 / 16
            gray[y + 1, x] += error * 5 / 16
            gray[y + 1, x + 1] += error * 1 / 16

    mask = dithered[..., None] > 127
    result = np.where(mask, small_img, small_img * 0.7).astype(np.uint8)

    return cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
# endregion
