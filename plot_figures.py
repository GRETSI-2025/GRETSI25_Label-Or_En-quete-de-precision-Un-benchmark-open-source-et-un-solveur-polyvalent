import benchopt
from pathlib import Path
from benchopt.plotting import plot_benchmark, PLOT_KINDS
from benchopt.benchmark import Benchmark
import matplotlib.pyplot as plt

from benchopt.utils.parquet import get_metadata
from benchopt.utils.parquet import update_metadata
import pandas as pd
from benchopt.plotting import get_plot_id
import itertools
from benchopt.plotting import plot_benchmark_html
# from benchopt.plotting.plot_objective_curve import plot_suboptimality_curve  # noqa: F401

CMAP = plt.get_cmap('tab20')
COLORS = [CMAP(i) for i in range(CMAP.N)]
COLORS = COLORS[::2] + COLORS[1::2]
MARKERS = {i: v for i, v in enumerate(plt.Line2D.markers)}
solvers_idx = {}
FONTSIZE = 14


def get_solver_style(solver, plotly=True):
    idx = solvers_idx.get(solver, len(solvers_idx))
    solvers_idx[solver] = idx

    color = COLORS[idx % len(COLORS)]
    marker = MARKERS[idx % len(MARKERS)]

    if plotly:
        color = tuple(255*x if i != 3 else x for i, x in enumerate(color))
        color = f'rgba{color}'
        marker = idx

    return color, marker


def custom_plot(df, obj_col,
                ax):
    df = df.copy()
    solver_names = df['solver_name'].unique()
    title = df['data_name'].unique()[0]  # [10:39]
    df.query(f"`{obj_col}` not in [inf, -inf]", inplace=True)

    eps = 1e-10
    y_label = "F(x) - F(x*)"
    c_star = df[obj_col].min() - eps
    df.loc[:, obj_col] -= c_star

    for i, solver_name in enumerate(solver_names):
        # breakpoint()
        if solver_name == 'skglm[algo=dual,inner_anderson=True,outer_anderson=False]':
            solver_name_label = 'Sk-GLM[dual, Anderson] ($Ours$)'
            color = "tab:orange"

        elif solver_name == 'skglm[algo=dual,inner_anderson=False,outer_anderson=False]':
            solver_name_label = 'Sk-GLM[dual] ($Ours$)'
            color = "tab:green"

        elif solver_name == 'gista':
            solver_name_label = f'G-ISTA ($Ours$)'
            color = "tab:blue"

        elif solver_name == 'sklearn':
            solver_name_label = "Scikit-learn ($GLasso$)"
            color = "tab:red"

        elif solver_name == 'gglasso':
            solver_name_label = "GGLasso ($ADMM$)"
            color = "tab:purple"

        elif solver_name == 'skggm':
            solver_name_label = "SkGGM ($QUIC$)"
            color = "tab:brown"

        else:
            breakpoint()

        df_ = df[df['solver_name'] == solver_name]
        curve = df_.groupby('stop_val').median(numeric_only=True)

        q1 = df_.groupby('stop_val')['time'].quantile(.1).to_numpy()
        q9 = df_.groupby('stop_val')['time'].quantile(.9).to_numpy()

        color, marker = get_solver_style(solver_name, plotly=False)
        ax.semilogy(curve['time'],
                    curve[obj_col],
                    color=color,
                    # marker=marker,
                    label=solver_name_label,
                    linewidth=3)
        ax.fill_betweenx(
            curve[obj_col].to_numpy(), q1, q9, color=color, alpha=.3
        )

    ax.hlines(eps, df['time'].min(), df['time'].max(), color='k',
              linestyle='--')
    ax.set_xlim(df['time'].min(), df['time'].max())
    ax.grid(which='both', alpha=0.9)
    return


def plot_bench(fname,
               benchmark,
               kinds=None,
               display=True):
    config = get_metadata(fname)
    params = ["plots", "plot_configs"]
    for param in params:
        options = benchmark.get_setting(param, default_config=config)
        if options is not None:
            config[param] = options
    update_metadata(fname, config)
    if kinds is not None and len(kinds) > 0:
        config["plots"] = kinds

    df = pd.read_parquet(fname)
    obj_cols = [
        k for k in df.columns
        if k.startswith('objective_') and k != 'objective_name'
    ]
    datasets = df['data_name'].unique()

    plt.close('all')
    fig, ax = plt.subplots(3, 2, figsize=(
        [4.96, 6.82]), constrained_layout=True)
    plt.tight_layout()

    for j, data in enumerate(datasets[:3]):  # [1:4]):
        df_data = df[df['data_name'] == data]
        objective_names = df['objective_name'].unique()

        for k, objective_name in enumerate(objective_names):
            df_obj = df_data[df_data['objective_name'] == objective_name]
            for kind, obj_col in itertools.product(
                    config["plots"], obj_cols
            ):
                if obj_col != "objective_value" and (
                        kind == "bar_chart" or "subopt" in kind):
                    continue
                custom_plot(df_obj, obj_col=obj_col, ax=ax[j, k])

                ax[j, 0].set_ylabel("F(x) - F(x*)", fontsize=FONTSIZE)

                if j == 2:
                    ax[j, k].set_xlabel("Time [sec]", fontsize=FONTSIZE)

                ax[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

                if j == 0:
                    ax[j, k].set_xlim([0, 0.05])
                    ax[j, 0].set_title(
                        f"$\lambda = 0.1\lambda_\mathrm{{max}}$\np=50", fontsize=FONTSIZE)
                    ax[j, 1].set_title(
                        f"$\lambda = 0.01\lambda_\mathrm{{max}}$\np=50", fontsize=FONTSIZE)
                elif j == 1:
                    ax[j, k].set_xlim([0, 0.5])
                    ax[j, k].set_title(
                        f"p=100", fontsize=FONTSIZE)
                elif j == 2:
                    ax[j, 0].set_xlim([0, 0.75])
                    ax[j, 1].set_xlim([0, 2.5])
                    ax[j, k].set_title(
                        f"p=200", fontsize=FONTSIZE)

                # plt.legend(
                #     loc='lower center',
                #     bbox_to_anchor=(-1.1, -0.65),
                #     fancybox=True,
                #     shadow=True,
                #     ncol=3)
                # ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig('./test_better.pdf', bbox_inches='tight')
    return fig


fname = Path(
    # "./outputs/benchopt_run_2025-03-05_11h11m39.parquet")
    # "./outputs/benchopt_run_2025-03-07_15h28m13.parquet")
    "./outputs/benchopt_run_2025-03-10_15h03m40.parquet")
kinds = list(PLOT_KINDS.keys())
fig = plot_bench(fname, Benchmark(
    "./"),
    kinds=['suboptimality_curve'])
plt.show()
