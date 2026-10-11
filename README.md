# Fast Convolution Using Matrix Multiplication

## Abstract
This project implements valid 2D convolution in two ways: direct nested-loop computation and a matrix-based method. The matrix method uses **im2col** to flatten every input patch into a row, then calculates the output using NumPy matrix multiplication. The implementations are checked for numerical agreement and benchmarked using median execution times.

## Problem statement
Direct convolution repeatedly multiplies each kernel with overlapping input regions. This project studies whether transforming the input into a matrix and using matrix multiplication can improve execution time for different input dimensions.

## Objectives
- Implement direct 2D convolution.
- Implement im2col-based matrix convolution.
- Verify that both methods produce equivalent output.
- Compare execution times over multiple input sizes.
- Save benchmark results to CSV and performance graphs to PNG files.
- Discuss the trade-off between matrix multiplication efficiency and im2col overhead.

## Concept and methodology
For an input of shape `H × W` and a kernel of shape `Kh × Kw`, valid convolution produces an output of shape `(H-Kh+1) × (W-Kw+1)`.

1. Direct method: loops over every valid output location and kernel element.
2. Matrix method: extracts each valid input patch and flattens it into one row of a matrix (`im2col`).
3. Flatten the kernel into a vector.
4. Multiply the patch matrix by the kernel vector.
5. Reshape the result to the output dimensions.
6. Compare outputs and benchmark both approaches.

The implementation uses the common deep-learning convention in which the kernel is not flipped (technically cross-correlation). Both methods use the same convention so the outputs can be compared fairly.

## Repository structure
```text
fast-convolution-project/
├── README.md
├── requirements.txt
├── src/
│   ├── direct_convolution.py
│   ├── matrix_convolution.py
│   ├── benchmark.py
│   └── main.py
├── results/
│   ├── benchmark_results.csv   # generated when the program runs
│   ├── execution_time.png      # generated when the program runs
│   └── performance_comparison.png
└── screenshots/
```

## Requirements
- Python 3.10 or later
- NumPy
- Matplotlib
- Pandas

## Setup
Create and activate a virtual environment (recommended).

Windows:
```bash
py -m venv .venv
.venv\\Scripts\\activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run
Run from the repository root:
```bash
python -m src.main
```

To run only the benchmark:
```bash
python -m src.benchmark
```

The default benchmark tests square input dimensions 16, 32, 48, and 64 with a 3 × 3 kernel. Edit `sizes`, `kernel_size`, or `repeats` in `src/benchmark.py` to run additional experiments. Increase sizes gradually because the direct method uses Python loops and may become slow.

## Output files
- `results/benchmark_results.csv`: measured times, speedup, and numerical error.
- `results/execution_time.png`: execution time curves.
- `results/performance_comparison.png`: speedup chart.

No benchmark values are hardcoded. Results will vary by CPU, Python/NumPy version, and system load. The matrix-method time includes patch construction, so the comparison reflects the complete approach rather than multiplication alone.

## How to interpret results
`speedup = direct_time_ms / matrix_time_ms`
- Greater than 1: matrix method was faster in that run.
- Equal to 1: approximately equal execution time.
- Less than 1: direct method was faster.

Matrix multiplication may benefit from optimized numerical libraries, but im2col requires allocating and filling a larger intermediate matrix. For small inputs, this overhead may outweigh the multiplication benefit. This project measures rather than assumes which method is faster.

## Conclusion
The project demonstrates how convolution can be expressed as matrix multiplication and compares the performance of both implementations. It validates correctness before benchmarking and reports measured timing results, allowing the performance trade-offs to be analyzed.

## References
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
- Dumoulin, V. and Visin, F., *A guide to convolution arithmetic for deep learning*: https://arxiv.org/abs/1603.07285

## Academic integrity note
Understand the code and adapt it according to your course requirements. Keep meaningful commits as you implement, test, document, and improve the project. Acknowledge external references used in your submission.
