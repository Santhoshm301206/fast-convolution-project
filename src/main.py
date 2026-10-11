"""Run a small correctness demo followed by performance benchmarks."""
import numpy as np

from src.direct_convolution import direct_convolution2d
from src.matrix_convolution import matrix_convolution2d
from src.benchmark import run_benchmarks


def main():
    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=float)
    kernel = np.array([
        [1, 0],
        [0, 1],
    ], dtype=float)

    direct_result = direct_convolution2d(image, kernel)
    matrix_result = matrix_convolution2d(image, kernel)

    print("=" * 55)
    print("FAST CONVOLUTION USING MATRIX MULTIPLICATION")
    print("=" * 55)
    print("\nSample input:\n", image)
    print("\nKernel:\n", kernel)
    print("\nDirect convolution output:\n", direct_result)
    print("\nMatrix-based output:\n", matrix_result)
    print("\nOutputs match:", np.allclose(direct_result, matrix_result))
    print("\nStarting benchmark experiments...\n")
    run_benchmarks()


if __name__ == "__main__":
    main()
