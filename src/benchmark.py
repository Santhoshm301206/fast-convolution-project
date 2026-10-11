"""Benchmark direct and im2col/matrix convolution and save CSV + plots."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.direct_convolution import direct_convolution2d
from src.matrix_convolution import matrix_convolution2d

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def median_runtime_ms(function, *args, repeats: int = 3) -> float:
    """Return median runtime in milliseconds after one warm-up run."""
    function(*args)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def run_benchmarks(sizes=(16, 32, 48, 64), kernel_size=3, repeats=3):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(2026)

    for size in sizes:
        image = rng.random((size, size), dtype=np.float64)
        kernel = rng.random((kernel_size, kernel_size), dtype=np.float64)

        direct_out = direct_convolution2d(image, kernel)
        matrix_out = matrix_convolution2d(image, kernel)
        max_error = float(np.max(np.abs(direct_out - matrix_out)))
        if not np.allclose(direct_out, matrix_out, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Outputs differ for input size {size}; max error={max_error}")

        direct_ms = median_runtime_ms(direct_convolution2d, image, kernel, repeats=repeats)
        matrix_ms = median_runtime_ms(matrix_convolution2d, image, kernel, repeats=repeats)
        speedup = direct_ms / matrix_ms if matrix_ms > 0 else float("inf")

        rows.append({
            "input_size": size,
            "kernel_size": kernel_size,
            "direct_time_ms": direct_ms,
            "matrix_time_ms": matrix_ms,
            "speedup_direct_over_matrix": speedup,
            "max_absolute_output_error": max_error,
        })
        print(
            f"{size:>4}x{size:<4} | direct={direct_ms:>10.3f} ms | "
            f"im2col+matmul={matrix_ms:>10.3f} ms | speedup={speedup:.3f}x | "
            f"max error={max_error:.3e}"
        )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["input_size"], df["direct_time_ms"], marker="o", label="Direct convolution")
    plt.plot(df["input_size"], df["matrix_time_ms"], marker="o", label="im2col + matrix multiplication")
    plt.xlabel("Input matrix dimension (N × N)")
    plt.ylabel("Median execution time (ms)")
    plt.title(f"Execution time comparison ({kernel_size} × {kernel_size} kernel)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "execution_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(df["input_size"].astype(str), df["speedup_direct_over_matrix"])
    plt.axhline(1.0, linestyle="--", linewidth=1, label="Equal execution time (1×)")
    plt.xlabel("Input matrix dimension (N × N)")
    plt.ylabel("Speedup (direct time / matrix method time)")
    plt.title("Relative performance of matrix-based convolution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "performance_comparison.png", dpi=160)
    plt.close()

    print(f"\nOutput check: PASSED (maximum absolute error <= {df['max_absolute_output_error'].max():.3e})")
    print(f"CSV saved to: {csv_path}")
    print(f"Graphs saved to: {RESULTS_DIR}")
    print("\nInterpretation: speedup > 1 means the matrix method was faster for that run;")
    print("speedup < 1 means direct convolution was faster. Results depend on hardware and sizes.")
    return df


if __name__ == "__main__":
    print("FAST CONVOLUTION BENCHMARK")
    print("Times include the im2col patch-construction overhead for the matrix method.")
    print("-" * 88)
    run_benchmarks()
