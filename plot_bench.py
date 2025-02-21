from pathlib import Path
import matplotlib.pyplot as plt

from benchopt import run_benchmark
from benchopt.benchmark import Benchmark
from benchopt.plotting import plot_benchmark, PLOT_KINDS
from benchopt.plotting.plot_objective_curve import reset_solver_styles_idx

BENCHMARK_PATH = (
    Path() / '.'
)

save_file = run_benchmark(
    BENCHMARK_PATH,
    solver_names=['sklearn[liblinear]', 'sklearn[newton-cg]', 'lightning'],
    dataset_names=["Simulated[n_features=500,n_samples=200]"],
    objective_filters=['L2 Logistic Regression[lmbd=1.0]'],
    max_runs=100, timeout=20, n_repetitions=15,
    plot_result=False, show_progress=True
)
