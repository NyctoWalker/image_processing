import cv2
import numpy as np

KERNEL_NAMES = [
    "Sharpen", "Edge", "Blur", "Emboss", "Identity",
    "Sobel X", "Sobel Y", "Laplacian", "Random"
]


def _make_sharpen(size):
    k = np.full((size, size), -1, dtype=np.float32)
    k[size // 2, size // 2] = size * size - 1
    return k


def _make_edge(size):
    k = np.full((size, size), -1, dtype=np.float32)
    k[size // 2, size // 2] = size * size - 1
    return k


def _make_blur(size):
    return np.ones((size, size), dtype=np.float32) / (size * size)


def _make_emboss(size):
    k = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    for i in range(size):
        for j in range(size):
            k[i, j] = (i - cy) - (j - cx)
    return k


def _make_identity(size):
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 1
    return k


def _make_sobel_x(size):
    k = np.zeros((size, size), dtype=np.float32)
    for j in range(size):
        k[:, j] = j - size // 2
    return k


def _make_sobel_y(size):
    k = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        k[i, :] = i - size // 2
    return k


def _make_laplacian(size):
    k = np.ones((size, size), dtype=np.float32)
    k[size // 2, size // 2] = -(size * size - 1)
    return k


def _make_random(size):
    rng = np.random.default_rng(seed=size * 12345)
    k = rng.standard_normal((size, size)).astype(np.float32)
    s = np.sum(np.abs(k))
    return k / s if s != 0 else k


KERNEL_MAKERS = [
    _make_sharpen, _make_edge, _make_blur, _make_emboss, _make_identity,
    _make_sobel_x, _make_sobel_y, _make_laplacian, _make_random
]


def _get_kernel(kernel_type, size):
    if 0 <= int(kernel_type) < len(KERNEL_MAKERS):
        return KERNEL_MAKERS[int(kernel_type)](size)
    return _make_identity(size)


def _conv2d_channel(img, kernel, stride, padding, dilation):
    h, w = img.shape
    k_h, k_w = kernel.shape

    eff_k_h = k_h + (k_h - 1) * (dilation - 1)
    eff_k_w = k_w + (k_w - 1) * (dilation - 1)

    if dilation > 1:
        d_kernel = np.zeros((eff_k_h, eff_k_w), dtype=np.float32)
        d_kernel[::dilation, ::dilation] = kernel
        kernel = d_kernel
        k_h, k_w = eff_k_h, eff_k_w

    if padding > 0:
        img = np.pad(img, padding, mode='constant')
        h, w = img.shape

    h_out = (h - k_h) // stride + 1
    w_out = (w - k_w) // stride + 1

    if h_out <= 0 or w_out <= 0:
        return cv2.resize(img, (max(1, w), max(1, h)))[:h, :w]

    if stride == 1:
        full = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return full[k_h // 2:h - k_h + k_h // 2 + 1, k_w // 2:w - k_w + k_w // 2 + 1]

    windows = np.lib.stride_tricks.sliding_window_view(img, (k_h, k_w))[::stride, ::stride].copy()
    return np.sum(windows * kernel, axis=(-2, -1))


def apply_conv2d(img, kernel_type=0, kernel_size=3, stride=1, padding=0, dilation=1):
    kernel = _get_kernel(kernel_type, kernel_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img[:, :, 0] if len(img.shape) == 3 else img
        result = _conv2d_channel(gray, kernel, stride, padding, dilation)
        return np.stack([result] * 3, axis=-1).clip(0, 255).astype(np.uint8)

    dummy = _conv2d_channel(img[:, :, 0], kernel, stride, padding, dilation)
    h_out, w_out = dummy.shape
    result = np.zeros((h_out, w_out, 3), dtype=np.float32)
    result[:, :, 0] = dummy
    for c in range(1, 3):
        result[:, :, c] = _conv2d_channel(img[:, :, c], kernel, stride, padding, dilation)

    return result.clip(0, 255).astype(np.uint8)


def apply_pooling(img, kernel_size=2, stride=2, pool_type=0, padding=0):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1

    if padding > 0:
        if c > 1:
            img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        else:
            img = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')
        h, w = img.shape[:2]

    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1

    if h_out <= 0 or w_out <= 0:
        return img

    if stride == kernel_size:
        h_eff = h_out * kernel_size
        w_eff = w_out * kernel_size
        if c == 1:
            patches = img[:h_eff, :w_eff].reshape(h_out, kernel_size, w_out, kernel_size)
            reduce_axes = (1, 3)
        else:
            patches = img[:h_eff, :w_eff].reshape(h_out, kernel_size, w_out, kernel_size, c)
            reduce_axes = (1, 3)
    else:
        if c == 1:
            patches = np.lib.stride_tricks.sliding_window_view(img, (kernel_size, kernel_size))[::stride, ::stride]
        else:
            patches = np.lib.stride_tricks.sliding_window_view(img, (kernel_size, kernel_size), axis=(0, 1))[::stride, ::stride]
        reduce_axes = (-2, -1)

    if pool_type == 0:
        result = patches.max(axis=reduce_axes)
    else:
        result = patches.mean(axis=reduce_axes).astype(np.float32)

    if c == 1:
        return np.stack([result] * 3, axis=-1).clip(0, 255).astype(np.uint8)
    return result.clip(0, 255).astype(np.uint8)


def _transposed_filter2d(img_f32, kernel, stride):
    h, w = img_f32.shape
    k_h, k_w = kernel.shape

    h_out = (h - 1) * stride + 2 - k_h
    w_out = (w - 1) * stride + 2 - k_w

    out = np.zeros((h_out, w_out), dtype=np.float32)

    for r in range(stride):
        ri = (stride - r) % stride
        i_start = (r + stride - 1) // stride
        sub_h = h - i_start
        if ri >= k_h:
            continue
        sk_h = (k_h - ri + stride - 1) // stride

        for s in range(stride):
            rj = (stride - s) % stride
            j_start = (s + stride - 1) // stride
            sub_w = w - j_start
            if rj >= k_w:
                continue
            sk_w = (k_w - rj + stride - 1) // stride

            if sk_h <= 0 or sk_w <= 0:
                continue

            sub_k = kernel[ri:ri + sk_h * stride:stride, rj:rj + sk_w * stride:stride].copy()

            sub_img = img_f32[i_start:, j_start:]
            if sub_img.shape[0] < sk_h or sub_img.shape[1] < sk_w:
                continue

            full = cv2.filter2D(sub_img, -1, sub_k, borderType=cv2.BORDER_CONSTANT)
            valid = full[sk_h // 2:sub_h - sk_h + sk_h // 2 + 1, sk_w // 2:sub_w - sk_w + sk_w // 2 + 1]

            out_h_rs = (h_out - r + stride - 1) // stride
            out_w_rs = (w_out - s + stride - 1) // stride

            take_h = min(valid.shape[0], out_h_rs)
            take_w = min(valid.shape[1], out_w_rs)

            out[r:r + take_h * stride:stride, s:s + take_w * stride:stride] = valid[:take_h, :take_w]

    return out


def apply_transposed_conv2d(img, kernel_type=0, kernel_size=3, stride=2):
    kernel = _get_kernel(kernel_type, kernel_size)
    k_h, k_w = kernel.shape

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1

    h_out = (h - 1) * stride + 2 - k_h
    w_out = (w - 1) * stride + 2 - k_w

    if c == 1 or len(img.shape) == 2:
        gray = img[:, :, 0].astype(np.float32) if len(img.shape) == 3 else img.astype(np.float32)
        result = _transposed_filter2d(gray, kernel, stride)
        return np.stack([result] * 3, axis=-1).clip(0, 255).astype(np.uint8)

    result = np.zeros((h_out, w_out, c), dtype=np.float32)
    for ch in range(c):
        result[:, :, ch] = _transposed_filter2d(img[:, :, ch].astype(np.float32), kernel, stride)

    return result.clip(0, 255).astype(np.uint8)
