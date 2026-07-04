import cv2
import numpy as np

# region CRT filter

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
