import benchopt
from pathlib import Path
from benchopt.plotting import plot_benchmark, PLOT_KINDS
from benchopt.benchmark import Benchmark
import matplotlib.pyplot as plt


fname = Path(
    # "./outputs/benchopt_run_2025-02-27_16h49m04.parquet")
    "./outputs/benchopt_run_2025-03-04_17h43m40.parquet")

kinds = list(PLOT_KINDS.keys())
figs = plot_benchmark(fname, Benchmark(
    "./"),
    kinds=['suboptimality_curve'],
    html=False)


fig_id = 1
ax = figs[fig_id].axes[0]
ax.set_xlim([0, 0.75])
ax.set_xscale("linear")
ax.set_ylabel("F(x) - F(x*)")
ax.set_title(f"p=100, n=1000", fontsize=16)
ax.grid(which='both', alpha=0.9)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# figs[fig_id].savefig("./p200.pdf", bbox_inches='tight')
figs[fig_id].savefig("./quic_p200.pdf", bbox_inches='tight')
