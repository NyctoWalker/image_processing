import cv2
import numpy as np
from numba import jit


# region HSB/Pixel modify
def apply_hsb_adjustment(img, hue, saturation, brightness):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
    hsv[..., 0] = (hsv[..., 0] + (hue / 2)) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:] * np.array([saturation/100, brightness/100]), 0, 255)
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


def apply_noise_dither(img, preserve_color=0):
    h, w = img.shape[:2]
    blue_noise = np.random.rand(h, w) * 255

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > blue_noise).astype(np.uint8)
    if preserve_color == 1:
        result = np.zeros_like(img)
        result[mask == 1] = img[mask == 1]
        return result
    else:
        return cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
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


def apply_chromatic_aberration(img, shift=2):
    h, w, c = img.shape
    result = np.zeros_like(img)
    result[:-shift, :-shift, 0] = img[shift:, shift:, 0]
    result[shift:, shift:, 2] = img[:-shift, :-shift, 2]
    result[:, :, 1] = img[:, :, 1]
    return np.clip(result, 0, 255).astype('uint8')


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
# endregion


# region Near-pixelizing
def apply_voxel_effect(img, block_size=8, height_scale=0.5, light_dir=(1.0, 1.0, 1.0)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    heightmap = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    heightmap = heightmap ** 0.5
    heightmap = heightmap * block_size * height_scale

    sobel_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)

    light_dir = np.array(light_dir)
    light_dir = light_dir / np.linalg.norm(light_dir)

    normal = np.dstack((-sobel_x, -sobel_y, np.ones_like(heightmap)))
    normal_norm = normal / np.linalg.norm(normal, axis=2, keepdims=True)
    shading = np.dot(normal_norm.reshape(-1, 3), light_dir).reshape(h, w)

    result = img * (0.5 + 0.5 * shading[..., None])
    return cv2.resize(np.clip(result, 0, 255).astype('uint8'),
                      (img.shape[1], img.shape[0]))
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
