import cv2
import numpy as np
from numba import jit, prange
import random
import math

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


def topographical_map(img, contour_levels=12, line_thickness=1, elevation_contrast=1.5, line_brightness=0.3, elevation_brightness_boost=0.5):
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    heightmap = cv2.GaussianBlur(gray, (7, 7), 0)
    heightmap = cv2.normalize(heightmap, None, 0, 255, cv2.NORM_MINMAX)
    adjusted = 255 * np.power(heightmap / 255, 1 / max(0.1, elevation_contrast))
    adjusted = cv2.normalize(adjusted, None, 0, 255, cv2.NORM_MINMAX)

    bins = np.linspace(0, 255, contour_levels + 1)
    quantized = np.digitize(adjusted, bins[1:-1]).astype(np.uint8)

    edges = np.zeros_like(quantized, dtype=np.uint8)
    for level in np.unique(quantized):
        mask = (quantized == level).astype(np.uint8) * 255
        level_edges = cv2.Canny(mask, 50, 150)
        edges = cv2.bitwise_or(edges, level_edges)

    if line_thickness > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*line_thickness+1, 2*line_thickness+1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        grad_x = cv2.Sobel(adjusted, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(adjusted, cv2.CV_32F, 0, 1, ksize=5)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        light_dir = np.array([-0.5, -0.7, 0.5])
        normal = np.dstack((-grad_x, -grad_y, np.ones_like(grad_x)))
        normal_norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal /= np.where(normal_norm > 0, normal_norm, 1)
        shading = np.dot(normal, light_dir)

        elevation_scale = 1 + (adjusted/255) * elevation_brightness_boost
        valley_shadows = 1 - (1 - cv2.normalize(adjusted, None, 0, 1, cv2.NORM_MINMAX)) * elevation_brightness_boost
        shading = shading * elevation_scale * valley_shadows
        shading = np.clip(shading, 0.3, 1.8)

        contour_mask = edges.astype(np.float32) / 255.0
        contour_effect = contour_mask * line_brightness * (0.7 + 0.3*(adjusted/255))
        shaded = img.astype(np.float32) * shading[..., None]
        result = shaded + (contour_effect[..., None] * 60)
        result = np.clip(result, 0, 255)
    else:
        result = img.astype(np.float32)

    return result.astype(np.uint8)


def glitchy_pixelation(img, base_size=4, channel_shift=3):
    r = pixelize_image(img[..., 0], base_size + channel_shift)
    g = pixelize_image(img[..., 1], base_size)
    b = pixelize_image(img[..., 2], base_size - channel_shift)
    return cv2.merge((r, g, b))


@jit(nopython=True, fastmath=True)
def _generate_fragments(h, w, fragment_size):
    grid_x = np.arange(0, w + fragment_size, fragment_size)
    grid_y = np.arange(0, h + fragment_size, fragment_size)
    centers = []
    for y in grid_y:
        for x in grid_x:
            centers.append((
                x + random.randint(-fragment_size // 2, fragment_size // 2),
                y + random.randint(-fragment_size // 2, fragment_size // 2)
            ))
    return centers


@jit(nopython=True, parallel=True, fastmath=True)
def _apply_fragments(img, centers, fragment_size, distortion_strength):
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    count = np.zeros((h, w), dtype=np.float32)

    for i in prange(len(centers)):
        cx, cy = centers[i]
        radius = fragment_size * (0.8 + random.random() * 0.4)
        sides = random.randint(3, 5)
        mask = _create_polygon_mask(h, w, cx, cy, radius, sides, distortion_strength)

        total_r = total_g = total_b = 0.0
        pixels = 0
        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    total_r += img[y, x, 0]
                    total_g += img[y, x, 1]
                    total_b += img[y, x, 2]
                    pixels += 1

        if pixels > 0:
            avg_r = total_r / pixels
            avg_g = total_g / pixels
            avg_b = total_b / pixels

            dev_r = random.uniform(-40, 40)
            dev_g = random.uniform(-40, 40)
            dev_b = random.uniform(-40, 40)
            frag_r = min(max(avg_r + dev_r, 0), 255)
            frag_g = min(max(avg_g + dev_g, 0), 255)
            frag_b = min(max(avg_b + dev_b, 0), 255)

            alpha = 0.7 + random.random() * 0.3
            for y in range(h):
                for x in range(w):
                    if mask[y, x]:
                        result[y, x, 0] = result[y, x, 0] * (1 - alpha) + frag_r * alpha
                        result[y, x, 1] = result[y, x, 1] * (1 - alpha) + frag_g * alpha
                        result[y, x, 2] = result[y, x, 2] * (1 - alpha) + frag_b * alpha
                        count[y, x] += alpha

    for y in prange(h):
        for x in prange(w):
            if count[y, x] > 0:
                result[y, x] = result[y, x] / count[y, x]

    return result.astype(np.uint8)


@jit(nopython=True, fastmath=True)
def _create_polygon_mask(h, w, cx, cy, radius, sides, distortion_strength):
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = []

    for i in range(sides):
        angle = 2 * math.pi * i / sides
        offset_x = radius * math.cos(angle) * (1 + random.random() * distortion_strength)
        offset_y = radius * math.sin(angle) * (1 + random.random() * distortion_strength)
        polygon.append((cx + offset_x, cy + offset_y))

    min_y = max(0, int(cy - radius))
    max_y = min(h, int(cy + radius))
    min_x = max(0, int(cx - radius))
    max_x = min(w, int(cx + radius))

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            inside = False
            n = len(polygon)
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                mask[y, x] = 1
    return mask


def apply_cubist_effect(img, fragment_size=50, distortion_strength=0.3):
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    centers = _generate_fragments(h, w, fragment_size)
    return _apply_fragments(img, centers, fragment_size, distortion_strength)
# endregion

# region Oil and vector
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


def apply_molecular_effect(img, atom_scale=0.1, bond_threshold=30):
    h, w = img.shape[:2]
    small_h, small_w = int(h * atom_scale), int(w * atom_scale)

    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Псевдо-3Д
    xx, yy = np.meshgrid(np.arange(small_w), np.arange(small_h))
    zz = l / 255.0 * small_h
    positions = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    colors = small.reshape(-1, 3)

    grad_x = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    result = np.full((h, w, 3), 40, dtype=np.uint8)
    # "Связи"
    for y in range(small_h - 1):
        for x in range(small_w - 1):
            if grad_mag[y, x] > bond_threshold:
                for dy, dx in [(0, 1), (1, 0)]:
                    ny, nx = y + dy, x + dx
                    if ny < small_h and nx < small_w and grad_mag[ny, nx] > bond_threshold:
                        color = (small[y, x] + small[ny, nx]) // 2
                        cv2.line(result,
                                 (int(x / atom_scale), int(y / atom_scale)),
                                 (int(nx / atom_scale), int(ny / atom_scale)),
                                 color.tolist(), 1)
    # "Атомы"
    for i, (x, y, z) in enumerate(positions):
        radius = max(1, int(3 * z / small_h))
        cv2.circle(result, (int(x / atom_scale), int(y / atom_scale)), radius, colors[i].tolist(), -1)
        if radius > 2:
            cv2.circle(result, (int(x / atom_scale), int(y / atom_scale)), max(1, radius // 3), [min(c + 100, 255) for c in colors[i]], -1)
    return result
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
