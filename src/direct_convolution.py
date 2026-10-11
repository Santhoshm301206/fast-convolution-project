"""Reference direct 2D valid convolution implemented with explicit loops."""
import numpy as np


def direct_convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute valid 2D cross-correlation using nested loops.

    Note: Most deep-learning libraries call cross-correlation "convolution".
    The kernel is not flipped, matching the im2col implementation in this project.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")
    if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
        raise ValueError("kernel dimensions cannot exceed image dimensions")

    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for row in range(out_h):
        for col in range(out_w):
            total = 0.0
            for kr in range(kernel.shape[0]):
                for kc in range(kernel.shape[1]):
                    total += image[row + kr, col + kc] * kernel[kr, kc]
            output[row, col] = total
    return output
