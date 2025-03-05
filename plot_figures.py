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
FONTSIZE = 18


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
    title = df['data_name'].unique()[0][10:39]
    df.query(f"`{obj_col}` not in [inf, -inf]", inplace=True)

    eps = 1e-10
    y_label = "F(x) - F(x*)"
    c_star = df[obj_col].min() - eps
    df.loc[:, obj_col] -= c_star

    # fig = plt.figure()

    # if df[obj_col].count() == 0:  # missing values
    #     ax.text(0.5, 0.5, "Not Available")
    #     return fig

    for i, solver_name in enumerate(solver_names):
        if solver_name == 'skglm[algo=dual,inner_anderson=True]':
            solver_name_label = 'skglm[dual, Anderson]'
        elif solver_name == 'skglm[algo=dual,inner_anderson=False]':
            solver_name_label = 'skglm[dual]'
        else:
            solver_name_label = solver_name
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
    ax.set_xlabel("Time [sec]", fontsize=FONTSIZE)

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
    fig, ax = plt.subplots(1, 3, figsize=([7.65, 2.99]))
    for j, data in enumerate(datasets[1:4]):
        df_data = df[df['data_name'] == data]
        objective_names = df['objective_name'].unique()
        for objective_name in objective_names:
            df_obj = df_data[df_data['objective_name'] == objective_name]
            for kind, obj_col in itertools.product(
                    config["plots"], obj_cols
            ):
                if obj_col != "objective_value" and (
                        kind == "bar_chart" or "subopt" in kind):
                    continue
                # fig = custom_plot(df_obj, obj_col=obj_col, ax=ax[j])
                custom_plot(df_obj, obj_col=obj_col, ax=ax[j])
                if j == 0:
                    ax[j].set_ylabel("F(x) - F(x*)", fontsize=FONTSIZE)
                if j == 0:
                    ax[j].set_xlim([0, 0.3])
                    ax[j].set_title("p=100", fontsize=FONTSIZE)
                elif j == 1:
                    ax[j].set_xlim([0, 0.75])
                    ax[j].set_title("p=200", fontsize=FONTSIZE)
                elif j == 2:
                    ax[j].set_xlim([0, 2])
                    ax[j].set_title("p=500", fontsize=FONTSIZE)
                # figs.append(fig)
                plt.legend(
                    loc='lower center',
                    bbox_to_anchor=(-1.1, -0.5),
                    fancybox=True,
                    shadow=True,
                    ncol=6)
                plt.tight_layout()

    fig.savefig('./test.pdf', bbox_inches='tight')
    return fig


fname = Path(
    "./outputs/benchopt_run_2025-03-05_11h11m39.parquet")
kinds = list(PLOT_KINDS.keys())
fig = plot_bench(fname, Benchmark(
    "./"),
    kinds=['suboptimality_curve'])
plt.show()
