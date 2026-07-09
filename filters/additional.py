import cv2
import numpy as np

# region Emboss/Relief

def apply_emboss(img, intensity=1.0, direction=0, preserve_color=0):
    kernels = [
        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
        np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    ]
    kernel = kernels[int(direction) % 4] * float(intensity)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    emboss = cv2.filter2D(gray, -1, kernel)
    emboss = np.clip(emboss + 128, 0, 255).astype('uint8')

    if preserve_color:
        factor = emboss.astype(np.float32) / 128.0
        result = img.astype(np.float32) * factor[..., np.newaxis]
        return np.clip(result, 0, 255).astype('uint8')
    else:
        return cv2.cvtColor(emboss, cv2.COLOR_GRAY2RGB)

# endregion

# region CLAHE / Global Histogram Equalization

def apply_clahe(img, clip_limit=20, detail=5, mode=0):
    clip_val = np.clip(float(clip_limit), 0, 100)
    detail_val = np.clip(int(detail), 0, 10)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    if mode == 0:
        clahe_clip = 0.1 + clip_val / 100.0 * 9.9
        tile_size = 2 + detail_val * 2
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile_size, tile_size))
        l_eq = clahe.apply(l)
    else:
        hist = cv2.calcHist([l], [0], None, [256], [0, 256]).ravel()
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]

        linear = np.linspace(0, 1, 256)
        cdf_blend = detail_val / 10.0
        cdf = cdf_blend * cdf + (1.0 - cdf_blend) * linear

        lut = (cdf * 255).astype(np.uint8)
        l_eq = cv2.LUT(l, lut)

        strength = clip_val / 100.0
        if strength < 1.0:
            l_eq = cv2.addWeighted(l.astype(np.float32), 1.0 - strength,
                                    l_eq.astype(np.float32), strength, 0).astype(np.uint8)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

# endregion
