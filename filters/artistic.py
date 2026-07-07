import cv2
import numpy as np

# region Kuwahara Filter

def apply_kuwahara(img, radius=5, generalized=0):
    r = max(1, min(int(radius), 15))
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)
    result = np.empty_like(img_f)
    win = r + 1

    for c in range(3):
        ch = img_f[:, :, c]

        if generalized:
            sigma = win * 0.6
            mean = cv2.GaussianBlur(ch, (0, 0), sigma)
            sq_mean = cv2.GaussianBlur(ch * ch, (0, 0), sigma)
        else:
            mean = cv2.boxFilter(ch, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
            sq_mean = cv2.boxFilter(ch * ch, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)

        var = np.maximum(0, sq_mean - mean * mean)

        pad = r
        var_pad = cv2.copyMakeBorder(var, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        mean_pad = cv2.copyMakeBorder(mean, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        q_var_stack = np.stack([
            var_pad[pad:pad + h, pad:pad + w],
            var_pad[pad:pad + h, pad + r:pad + r + w],
            var_pad[pad + r:pad + r + h, pad:pad + w],
            var_pad[pad + r:pad + r + h, pad + r:pad + r + w],
        ], axis=-1)

        q_mean_stack = np.stack([
            mean_pad[pad:pad + h, pad:pad + w],
            mean_pad[pad:pad + h, pad + r:pad + r + w],
            mean_pad[pad + r:pad + r + h, pad:pad + w],
            mean_pad[pad + r:pad + r + h, pad + r:pad + r + w],
        ], axis=-1)

        best = np.argmin(q_var_stack, axis=-1)
        result[:, :, c] = np.take_along_axis(
            q_mean_stack, best[..., np.newaxis], axis=-1
        )[..., 0]

    return np.clip(result, 0, 255).astype('uint8')

# endregion

# region Watercolor Effect

def apply_watercolor(img, smooth=10, edge_darken=3):
    sigma_s = max(1, int(smooth))
    sigma_r = max(0.05, smooth / 15.0)

    smoothed = cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

    if edge_darken > 0:
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge = np.abs(lap)
        edge = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)
        edge = np.clip(edge * edge_darken * 0.4, 0, 1)
        result = smoothed.astype(np.float32) * (1.0 - edge[..., np.newaxis] * 0.5)
    else:
        result = smoothed.astype(np.float32)

    return np.clip(result, 0, 255).astype('uint8')

# endregion

# region Cross Processing

_LUT_CACHE = {}


def _build_stock_luts():
    x = np.arange(256, dtype=np.float32) / 255.0
    stocks = []

    def _lut(r_curve, g_curve, b_curve):
        l = np.empty((3, 256), dtype=np.uint8)
        for c, curve in enumerate((r_curve, g_curve, b_curve)):
            l[c] = np.clip(curve(x) * 255, 0, 255).astype(np.uint8)
        return l

    # Kodachrome 64
    stocks.append(_lut(
        lambda x: np.clip(x * 1.12 - 0.06, 0, 1),
        lambda x: np.clip(x * 1.05 - 0.02, 0, 1),
        lambda x: np.clip(x * 0.82 + 0.06, 0, 1),
    ))
    # Fuji Velvia 50
    stocks.append(_lut(
        lambda x: np.clip(x * 0.92 + 0.06, 0, 1),
        lambda x: np.clip(x * 1.18 - 0.08, 0, 1),
        lambda x: np.clip(x * 0.88 + 0.04, 0, 1),
    ))
    # Agfa Ultra 100
    stocks.append(_lut(
        lambda x: np.clip(x * 1.08 + 0.02, 0, 1),
        lambda x: np.clip(x * 1.04 + 0.01, 0, 1),
        lambda x: np.clip(x * 0.72 + 0.08, 0, 1),
    ))
    # Polaroid 600
    stocks.append(_lut(
        lambda x: np.clip((x - 0.5) * 1.35 + 0.5, 0, 1),
        lambda x: np.clip((x - 0.5) * 1.30 + 0.52, 0, 1),
        lambda x: np.clip((x - 0.5) * 1.25 + 0.54, 0, 1),
    ))
    # Fuji Superia
    stocks.append(_lut(
        lambda x: np.clip(x * 1.06 + 0.01, 0, 1),
        lambda x: np.clip(x * 1.03 + 0.01, 0, 1),
        lambda x: np.clip(x * 0.90 + 0.03, 0, 1),
    ))
    # Lomo
    stocks.append(_lut(
        lambda x: np.clip((x - 0.5) * 1.5 + 0.55, 0, 1),
        lambda x: np.clip((x - 0.5) * 1.4 + 0.45, 0, 1),
        lambda x: np.clip((x - 0.5) * 1.3 + 0.40, 0, 1),
    ))

    return stocks


_STOCK_LUTS = _build_stock_luts()
_STOCK_NAMES = ["Kodachrome 64", "Fuji Velvia 50", "Agfa Ultra 100", "Polaroid 600", "Fuji Superia", "Lomo"]


def apply_crossprocess(img, stock=0, intensity=80, brightness=100):
    stock = np.clip(int(stock), 0, len(_STOCK_LUTS) - 1)
    intensity = np.clip(intensity, 0, 100) / 100.0
    brightness = np.clip(brightness, 0, 200) / 100.0

    lut = _STOCK_LUTS[stock]
    processed = np.empty_like(img)
    for c in range(3):
        processed[:, :, c] = cv2.LUT(img[:, :, c], lut[c])

    result = cv2.addWeighted(img.astype(np.float32), 1.0 - intensity, processed.astype(np.float32), intensity, 0)
    result = result * brightness

    return np.clip(result, 0, 255).astype('uint8')

# endregion

# region Fractal Plasma Blend

def _value_noise(shape, scale, rng):
    h, w = shape
    grid_h = max(2, int(h / scale) + 3)
    grid_w = max(2, int(w / scale) + 3)
    grid = rng.rand(grid_h, grid_w)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = yy / scale
    xx = xx / scale

    ix = np.floor(xx).astype(np.int32)
    iy = np.floor(yy).astype(np.int32)
    fx = xx - ix
    fy = yy - iy

    ix = np.clip(ix, 0, grid_w - 2)
    iy = np.clip(iy, 0, grid_h - 2)

    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)

    v00 = grid[iy, ix]
    v10 = grid[iy, ix + 1]
    v01 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]

    v0 = v00 + (v10 - v00) * sx
    v1 = v01 + (v11 - v01) * sx
    return v0 + (v1 - v0) * sy


def _fbm(shape, scale, octaves):
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for o in range(octaves):
        rng = np.random.RandomState(o * 1337 + 42)
        noise += amplitude * _value_noise(shape, scale / frequency, rng)
        max_val += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return noise / max_val


def apply_fractal_plasma(img, scale=30, octaves=4, amount=50, scheme=0, mode=0):
    h, w = img.shape[:2]
    scale = max(5, int(scale))
    octaves = max(1, min(int(octaves), 8))
    amount = np.clip(amount, 0, 100) / 100.0
    scheme = int(scheme)
    mode = int(mode)

    noise = _fbm((h, w), scale, octaves)

    if scheme == 0:
        r = np.clip(noise * 2.0, 0, 1)
        g = np.clip(noise * 1.5 - 0.3, 0, 1)
        b = np.clip(noise - 0.7, 0, 1)
        colors = np.stack([r, g, b], axis=-1)
    elif scheme == 1:
        r = np.clip(noise - 0.5, 0, 1)
        g = np.clip(noise - 0.2, 0, 1)
        b = np.clip(noise * 1.5, 0, 1)
        colors = np.stack([r, g, b], axis=-1)
    elif scheme == 2:
        r = np.clip(noise * 1.2 + 0.3, 0, 1)
        g = np.clip(noise * 0.8 + 0.2, 0, 1)
        b = np.clip(noise * 0.3, 0, 1)
        colors = np.stack([r, g, b], axis=-1)
    elif scheme == 3:
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = (noise * 179).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    elif scheme == 4:
        colors = np.stack([noise] * 3, axis=-1)
    else:
        r = np.clip(noise * 1.5 - 0.2, 0, 1)
        g = np.clip(noise * 0.3, 0, 1)
        b = np.clip(noise * 1.8 - 0.3, 0, 1)
        colors = np.stack([r, g, b], axis=-1)

    colors = (colors * 255).astype(np.uint8)
    img_f = img.astype(np.float32)
    colors_f = colors.astype(np.float32)

    if mode == 0:
        blended = cv2.addWeighted(img_f, 1.0 - amount, colors_f, amount, 0)
    elif mode == 1:
        screen = 255.0 - (255.0 - img_f) * (255.0 - colors_f) / 255.0
        blended = cv2.addWeighted(img_f, 1.0 - amount, screen, amount, 0)
    elif mode == 2:
        mask = img_f < 128.0
        overlay = np.where(mask,
                          2.0 * img_f * colors_f / 255.0,
                          255.0 - 2.0 * (255.0 - img_f) * (255.0 - colors_f) / 255.0)
        blended = cv2.addWeighted(img_f, 1.0 - amount, overlay, amount, 0)
    else:
        t = colors_f / 255.0
        soft = (1.0 - 2.0 * t) * img_f * img_f / 255.0 + 2.0 * t * img_f
        blended = cv2.addWeighted(img_f, 1.0 - amount, soft, amount, 0)

    return np.clip(blended, 0, 255).astype('uint8')

# endregion

# region Orton Effect

def apply_orton_effect(img, blur=8, glow=50, warmth=0, brightness=110):
    blur = max(1, int(blur) * 2 + 1)
    glow = np.clip(glow, 0, 100) / 100.0
    warmth = np.clip(warmth, -50, 50)
    brightness = np.clip(brightness, 50, 200) / 100.0

    blurred = cv2.GaussianBlur(img, (blur, blur), 0)

    img_f = img.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    screen = 255.0 - (255.0 - img_f) * (255.0 - blur_f) / 255.0

    result = cv2.addWeighted(img_f, 1.0 - glow, screen, glow, 0)
    result = result * brightness

    if warmth != 0:
        w = warmth / 100.0
        if w > 0:
            result[:, :, 0] += w * 30
            result[:, :, 2] -= w * 20
        else:
            result[:, :, 0] -= abs(w) * 20
            result[:, :, 2] += abs(w) * 30

    return np.clip(result, 0, 255).astype('uint8')

# endregion
