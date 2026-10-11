"""2D valid convolution via im2col + NumPy matrix multiplication."""
import numpy as np


def im2col_valid(image: np.ndarray, kernel_shape: tuple[int, int]) -> np.ndarray:
    """Flatten each valid image patch into a row of a 2D matrix."""
    image = np.asarray(image, dtype=np.float64)
    kh, kw = kernel_shape
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if kh <= 0 or kw <= 0:
        raise ValueError("kernel dimensions must be positive")
    if kh > image.shape[0] or kw > image.shape[1]:
        raise ValueError("kernel dimensions cannot exceed image dimensions")

    out_h = image.shape[0] - kh + 1
    out_w = image.shape[1] - kw + 1
    patches = np.empty((out_h * out_w, kh * kw), dtype=np.float64)

    idx = 0
    for row in range(out_h):
        for col in range(out_w):
            patches[idx, :] = image[row:row + kh, col:col + kw].ravel()
            idx += 1
    return patches


def matrix_convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute valid 2D cross-correlation with im2col and matrix multiplication."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")

    patches = im2col_valid(image, kernel.shape)
    kernel_vector = kernel.ravel()
    result = patches @ kernel_vector

    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    return result.reshape(out_h, out_w)
