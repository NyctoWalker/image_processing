import cv2
import numpy as np
from numba import jit


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
