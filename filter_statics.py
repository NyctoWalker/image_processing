import cv2
import numpy as np
from numba import jit


# region HSB/Pixel modify
def apply_hsb_adjustment(img, hue, saturation, brightness):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
    hsv[..., 0] = (hsv[..., 0] + (hue / 2)) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:] * np.array([saturation/100, brightness/100]), 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)


def adjust_brightness(img, value):
    return np.clip(img.astype('int32') + value, 0, 255).astype('uint8')
# endregion


# region Simple filters
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


def apply_bleach_bypass(img):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    desat = (0.7 * gray + 0.3 * img.mean(axis=2))
    high_contrast = np.clip((desat - 128) * 1.5 + 128, 0, 255)
    return np.stack((high_contrast,)*3, axis=-1).astype('uint8')

# endregion


# region Edge detection
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
# endregion


def resize_image(img, scale_factor=1.0, interpolation='linear'):
    inter = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }.get(interpolation.lower(), cv2.INTER_LINEAR)

    if scale_factor != 1.0:
        return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=inter)
    return img


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


def apply_chromatic_aberration(img, shift=2):
    h, w, c = img.shape
    result = np.zeros_like(img)
    result[:-shift, :-shift, 0] = img[shift:, shift:, 0]
    result[shift:, shift:, 2] = img[:-shift, :-shift, 2]
    result[:, :, 1] = img[:, :, 1]
    return np.clip(result, 0, 255).astype('uint8')


# region Near-pixelizing
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

    small_h, small_w = h // pixel_size, w // pixel_size
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

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

    small_h, small_w = h // pixel_size, w // pixel_size
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_edges = cv2.resize(edges, (small_w, small_h)) > 0

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
