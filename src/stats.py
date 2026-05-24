import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "run_results"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots_faceted"
PAIR_PARAMETERS = ["alpha", "beta", "evaporation", "tau_gap", "coverage", "ants", "iterations"]


def load_results(results_dir: Path, pattern: str) -> pd.DataFrame:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files = sorted(results_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in {results_dir}")

    print(f"Files for analysis ({pattern}): {len(csv_files)}")
    frames = []
    for file_path in csv_files:
        frame = pd.read_csv(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def require_columns(df: pd.DataFrame, columns: list[str], mode_name: str) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        print(f"[skip] {mode_name}: missing columns: {', '.join(missing)}")
        return False
    return True


def plot_faceted_parameter_impact(df: pd.DataFrame, param_name: str, output_dir: Path) -> None:
    required = ["combo_name", param_name, "best_length", "update_strategy", "dataset", "coverage"]
    if not require_columns(df, required, f"OFAT {param_name}"):
        return

    mask = (
        df["combo_name"].str.contains("baseline", na=False)
        | df["combo_name"].str.contains(f"{param_name}=", na=False, regex=False)
    )
    filtered_df = df[mask].copy()

    if filtered_df[param_name].nunique() <= 1:
        print(f"[skip] {param_name}: not enough unique values after OFAT filtering.")
        return

    print(f"Generating OFAT plot: {param_name}")
    grid = sns.catplot(
        data=filtered_df,
        x=param_name,
        y="best_length",
        hue="update_strategy",
        col="dataset",
        row="coverage",
        kind="box",
        palette="Set2",
        height=4,
        aspect=1.2,
        sharey=False,
        margin_titles=True,
    )

    grid.fig.suptitle(
        f"Impact of {param_name} on route quality",
        y=1.02,
        fontweight="bold",
    )
    grid.set_axis_labels(param_name, "Best length")
    grid.set_titles(col_template="{col_name}", row_template="Coverage: {row_name}")
    grid.savefig(output_dir / f"impact_{param_name}.png", dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def run_ofat_experiments(results_dir: Path, plots_dir: Path) -> None:
    output_dir = plots_dir / "ofat_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir, "experiment_results_*.csv")
    print(f"Loaded experiment OFAT rows: {len(df)}")

    for param_name in ["alpha", "beta", "evaporation", "ants", "tau_gap"]:
        plot_faceted_parameter_impact(df, param_name, output_dir)

    print(f"[done] OFAT experiment plots saved to: {output_dir.resolve()}")


def detect_pair_parameters(df: pd.DataFrame) -> tuple[str, str] | None:
    source_files = sorted(df["source_file"].dropna().unique()) if "source_file" in df.columns else []
    for file_name in source_files:
        if not file_name.startswith("pairs_"):
            continue
        pair_name = file_name.removeprefix("pairs_")
        for first in PAIR_PARAMETERS:
            prefix = f"{first}_"
            if not pair_name.startswith(prefix):
                continue
            suffix = pair_name[len(prefix):]
            for second in PAIR_PARAMETERS:
                if suffix.startswith(f"{second}_"):
                    return first, second

    available = [param for param in PAIR_PARAMETERS if param in df.columns and df[param].nunique() > 1]
    if len(available) >= 2:
        return available[0], available[1]
    return None


def plot_pair_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    pair = detect_pair_parameters(df)
    if pair is None:
        print("[skip] pairs: could not detect two varied parameters.")
        return

    first_param, second_param = pair
    required = ["dataset", first_param, second_param, "best_length"]
    if not require_columns(df, required, "pairs"):
        return

    if "update_strategy" not in df.columns:
        df = df.copy()
        df["update_strategy"] = "all"
    if "coverage" not in df.columns:
        df = df.copy()
        df["coverage"] = "all"

    group_columns = ["dataset", "coverage", "update_strategy"]
    for (dataset, coverage, strategy), group in df.groupby(group_columns, dropna=False):
        pivot = group.pivot_table(
            index=first_param,
            columns=second_param,
            values="best_length",
            aggfunc="median",
        ).sort_index().sort_index(axis=1)

        if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
            print(
                f"[skip] pairs heatmap: dataset={dataset}, coverage={coverage}, "
                f"strategy={strategy} has insufficient {first_param}/{second_param} grid."
            )
            continue

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="viridis_r",
            linewidths=0.5,
            cbar_kws={"label": "Median best length"},
        )
        plt.title(
            f"{first_param}/{second_param} pair impact\n"
            f"{dataset} | coverage={coverage} | strategy={strategy}"
        )
        plt.xlabel(second_param)
        plt.ylabel(first_param)
        plt.tight_layout()

        safe_dataset = str(dataset).replace(".bin", "").replace("gray_", "")
        safe_coverage = str(coverage).replace(".", "p")
        output_file = output_dir / (
            f"heatmap_{first_param}_{second_param}_{safe_dataset}_cov{safe_coverage}_{strategy}.png"
        )
        plt.savefig(output_file, dpi=300)
        plt.close()


def run_pairs(results_dir: Path, plots_dir: Path, first_param: str = "alpha", second_param: str = "beta") -> None:
    output_dir = plots_dir / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir, f"pairs_{first_param}_{second_param}_*.csv")
    print(f"Loaded pair rows: {len(df)}")
    plot_pair_heatmaps(df, output_dir)
    print(f"[done] Pair heatmaps saved to: {output_dir.resolve()}")


def add_optimization_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def detect_optimization(row: pd.Series) -> str:
        if bool(row["use_node_exchange"]) and bool(row["use_2opt"]):
            return "node_exchange+2-opt"
        if bool(row["use_2opt"]):
            return "2-opt"
        if bool(row["use_or_opt"]):
            return "or-opt"
        if bool(row["use_node_exchange"]):
            return "node_exchange"
        return "baseline"

    df["optimization"] = df.apply(detect_optimization, axis=1)
    df["optimization"] = pd.Categorical(
        df["optimization"],
        categories=["baseline", "2-opt", "or-opt", "node_exchange", "node_exchange+2-opt"],
        ordered=True,
    )
    df["optimization_setting"] = df.apply(
        lambda row: "baseline"
        if row["optimization"] == "baseline"
        else f"{row['optimization']}\n{int(row['ls_top_ants'])} ants",
        axis=1,
    )
    return df


def plot_optimization_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    required = [
        "dataset",
        "coverage",
        "best_length",
        "use_2opt",
        "use_or_opt",
        "use_node_exchange",
        "ls_top_ants",
    ]
    if not require_columns(df, required, "optimization OFAT"):
        return

    plot_df = add_optimization_column(df)
    if plot_df["optimization_setting"].nunique() <= 1:
        print("[skip] optimization OFAT: not enough unique optimization settings.")
        return

    setting_order = ["baseline"]
    for optimization in ["2-opt", "or-opt", "node_exchange", "node_exchange+2-opt"]:
        for top_ants in sorted(plot_df["ls_top_ants"].dropna().unique()):
            setting = f"{optimization}\n{int(top_ants)} ants"
            if setting in set(plot_df["optimization_setting"]):
                setting_order.append(setting)

    print("Generating optimization OFAT boxplot")
    grid = sns.catplot(
        data=plot_df,
        x="optimization_setting",
        y="best_length",
        hue="optimization",
        hue_order=["baseline", "2-opt", "or-opt", "node_exchange", "node_exchange+2-opt"],
        order=setting_order,
        col="dataset",
        row="coverage",
        kind="box",
        palette="Set2",
        height=5,
        aspect=1.2,
        sharey=False,
        margin_titles=True,
    )

    grid.fig.suptitle(
        "Impact of local optimization on route quality",
        y=1.02,
        fontweight="bold",
    )
    grid.set_axis_labels("Optimization setting", "Best length")
    grid.set_titles(col_template="{col_name}", row_template="Coverage: {row_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", labelrotation=60)
        ax.set_xlabel("")
    grid.fig.subplots_adjust(bottom=0.18, wspace=0.25)
    grid.savefig(output_dir / "impact_optimizations.png", dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def run_ofat_optimizations(results_dir: Path, plots_dir: Path) -> None:
    output_dir = plots_dir / "ofat_optimizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir, "opt_results_*.csv")
    print(f"Loaded optimization OFAT rows: {len(df)}")
    plot_optimization_boxplots(df, output_dir)
    print(f"[done] Optimization OFAT plots saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MMAS experiment statistics plots.",
        epilog="pair parameters: alpha beta evaporation tau_gap coverage ants iterations",
    )
    parser.add_argument(
        "mode",
        choices=["ofat", "pair", "opt", "all"],
        nargs="?",
        default="ofat",
        help="Analysis mode to run.",
    )
    parser.add_argument("pair_param_1", nargs="?")
    parser.add_argument("pair_param_2", nargs="?")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    pair_arg_count = sum(value is not None for value in [args.pair_param_1, args.pair_param_2])
    if args.mode != "pair" and pair_arg_count > 0:
        parser.error("pair parameters are only accepted with pair mode")
    if pair_arg_count == 1:
        parser.error("provide both pair parameters or neither")
    if pair_arg_count == 0:
        args.pair_param_1 = "alpha"
        args.pair_param_2 = "beta"
    if (args.pair_param_1 not in PAIR_PARAMETERS) or (args.pair_param_2 not in PAIR_PARAMETERS):
        parser.error(f"pair parameters must be one of: {', '.join(PAIR_PARAMETERS)}")

    return args


def main() -> None:
    args = parse_args()

    if args.mode in {"ofat", "all"}:
        run_ofat_experiments(args.results_dir, args.plots_dir)
    if args.mode in {"pair", "all"}:
        run_pairs(args.results_dir, args.plots_dir, args.pair_param_1, args.pair_param_2)
    if args.mode in {"opt", "all"}:
        run_ofat_optimizations(args.results_dir, args.plots_dir)


if __name__ == "__main__":
    main()
