from pathlib import Path
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "run_results"
OUTPUT_PATH = SCRIPT_DIR / "stats_results"
ALPHA = 0.05
PARAMETERS = ["alpha", "beta", "evaporation", "ants", "tau_gap"]

def format_p_value(p_value: float) -> str:
    return f"{p_value:.6f}"

def test_result(p_value: float) -> str:
    return "Significant" if p_value < ALPHA else "Not Significant"

def load_results(results_dir: Path, prefix: str) -> pd.DataFrame:
    csv_files = sorted(results_dir.glob(f"{prefix}*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        frame["source_file"] = csv_file.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)

def mann_whitney_strategy_test(df: pd.DataFrame, dataset: str, coverage: float) -> dict | None:
    mask = (
            (df["dataset"] == dataset)
            & (df["coverage"] == coverage)
            & df["combo_name"].str.contains("baseline", na=False)
    )
    data = df[mask]

    iteration_best = data[data["update_strategy"] == "iteration_best"]["best_length"]
    best_so_far = data[data["update_strategy"] == "best_so_far"]["best_length"]

    if iteration_best.empty or best_so_far.empty:
        return None

    stat, p_value = stats.mannwhitneyu(iteration_best, best_so_far, alternative="two-sided")

    med_ib = iteration_best.median()
    med_bsf = best_so_far.median()

    if med_ib < med_bsf:
        best_strategy = "iteration_best"
    elif med_bsf < med_ib:
        best_strategy = "best_so_far"
    else:
        best_strategy = "equal"

    return {
        "test": "Mann-Whitney",
        "target": "update_strategy",
        "dataset": dataset,
        "coverage": coverage,
        "p_value": p_value,
        "result": test_result(p_value),
        "med_ib": med_ib,
        "med_bsf": med_bsf,
        "best_strategy": best_strategy
    }

def kruskal_wallis_param_test(df: pd.DataFrame, dataset: str, coverage: float, param: str) -> dict | None:
    mask = (
            (df["dataset"] == dataset)
            & (df["coverage"] == coverage)
            & (df["update_strategy"] == "iteration_best")
            & (df["combo_name"].str.contains("baseline", na=False) | df["combo_name"].str.contains(f"{param}=", na=False))
    )
    data = df[mask]

    groups = [group["best_length"].values for name, group in data.groupby(param)]

    if len(groups) < 2:
        return None

    stat, p_value = stats.kruskal(*groups)

    medians = data.groupby(param)["best_length"].median()
    best_val = medians.idxmin()
    best_median = medians.min()

    return {
        "test": "Kruskal-Wallis",
        "target": param,
        "dataset": dataset,
        "coverage": coverage,
        "p_value": p_value,
        "result": test_result(p_value),
        "best_value": best_val,
        "best_median": best_median
    }

def two_way_anova_test(df: pd.DataFrame, dataset: str, coverage: float, p1: str, p2: str) -> dict | None:
    mask = (df["dataset"] == dataset) & (df["coverage"] == coverage)
    data = df[mask]

    if data.empty or data[p1].nunique() < 2 or data[p2].nunique() < 2:
        return None

    formula = f'best_length ~ C({p1}) + C({p2}) + C({p1}):C({p2})'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    interaction_term = f'C({p1}):C({p2})'
    if interaction_term not in anova_table.index:
        return None

    p_value = anova_table.loc[interaction_term, 'PR(>F)']

    return {
        "test": "Two-Way ANOVA",
        "target": f"{p1}_x_{p2}",
        "dataset": dataset,
        "coverage": coverage,
        "p_value": p_value,
        "result": test_result(p_value)
    }

def mann_whitney_opt_test(df: pd.DataFrame, dataset: str, coverage: float, opt_combo: str) -> dict | None:
    mask_base = (df["dataset"] == dataset) & (df["coverage"] == coverage) & (df["combo_name"] == "baseline")
    mask_opt = (df["dataset"] == dataset) & (df["coverage"] == coverage) & (df["combo_name"] == opt_combo)

    baseline_data = df[mask_base]["best_length"]
    opt_data = df[mask_opt]["best_length"]

    if baseline_data.empty or opt_data.empty:
        return None

    stat, p_value = stats.mannwhitneyu(opt_data, baseline_data, alternative="two-sided")

    med_base = baseline_data.median()
    med_opt = opt_data.median()

    if med_opt < med_base:
        best_strategy = opt_combo
    elif med_base < med_opt:
        best_strategy = "baseline"
    else:
        best_strategy = "equal"

    return {
        "test": "Mann-Whitney (Opt vs Base)",
        "target": opt_combo,
        "dataset": dataset,
        "coverage": coverage,
        "p_value": p_value,
        "result": test_result(p_value),
        "med_opt": med_opt,
        "med_base": med_base,
        "best_opt": best_strategy
    }

def result_line(result: dict) -> str:
    base = (
        f"{result['test']} | Target: {result['target']} | Dataset: {result['dataset']} | "
        f"Coverage: {result['coverage']} | p-value: {format_p_value(result['p_value'])} | "
        f"Result: {result['result']}"
    )

    if result["test"] == "Mann-Whitney":
        extra = f" | MedIB: {result['med_ib']:.2f} | MedBSF: {result['med_bsf']:.2f} | Best: {result['best_strategy']}"
        return base + extra
    elif result["test"] == "Kruskal-Wallis":
        extra = f" | Best Val: {result['best_value']} (Med: {result['best_median']:.2f})"
        return base + extra
    elif result["test"] == "Mann-Whitney (Opt vs Base)":
        extra = f" | MedOpt: {result['med_opt']:.2f} | MedBase: {result['med_base']:.2f} | Best: {result['best_opt']}"
        return base + extra

    return base

def summarize_target(results: list[dict], target: str, datasets: list[str], coverages: list[float]) -> list[str]:
    target_results = [r for r in results if r["target"] == target]
    if not target_results:
        return []

    lines = []
    for coverage in coverages:
        cov_results = [r for r in target_results if r["coverage"] == coverage]
        if not cov_results:
            continue

        significant = {
            result["dataset"]
            for result in cov_results
            if result["result"] == "Significant"
        }

        if len(significant) == len(datasets):
            scope = "Significant for all datasets"
        elif significant:
            scope = "Significant for part of datasets"
        else:
            scope = "Not significant for any dataset"

        dataset_list = ", ".join(sorted(significant)) if significant else "none"
        lines.append(
            f"Target: {target} | Coverage: {coverage} | Result: {scope} | "
            f"Significant datasets: {dataset_list}"
        )

    significant_any = {
        result["dataset"]
        for result in target_results
        if result["result"] == "Significant"
    }

    if len(significant_any) == len(datasets):
        overall_scope = "Significant at least once for all datasets"
    elif significant_any:
        overall_scope = "Significant only for part of datasets"
    else:
        overall_scope = "Not significant for any dataset"

    dataset_list = ", ".join(sorted(significant_any)) if significant_any else "none"
    lines.append(
        f"Target: {target} | Overall: {overall_scope} | Significant datasets: {dataset_list}"
    )
    return lines

def write_report(df_ofat: pd.DataFrame, results_ofat: list[dict],
                 df_pairs: pd.DataFrame, results_pairs: list[dict],
                 df_opt: pd.DataFrame, results_opt: list[dict],
                 output_path: Path) -> None:
    lines = ["Detailed results (OFAT)"]
    lines.extend(result_line(result) for result in results_ofat)

    lines.append("")
    lines.append("Dataset-wide significance summary (OFAT)")
    if not df_ofat.empty:
        datasets_ofat = sorted(df_ofat["dataset"].dropna().unique())
        coverages_ofat = sorted(df_ofat["coverage"].dropna().unique())
        targets_ofat = ["update_strategy", *PARAMETERS]
        for target in targets_ofat:
            lines.extend(summarize_target(results_ofat, target, datasets_ofat, coverages_ofat))

    if results_pairs:
        lines.append("")
        lines.append("============================================================")
        lines.append("Pairwise Interaction Analysis (Two-Way ANOVA)")
        lines.append("")
        lines.append("Detailed results (PAIRS)")
        lines.extend(result_line(result) for result in results_pairs)

        lines.append("")
        lines.append("Dataset-wide interaction significance summary (PAIRS)")
        datasets_pairs = sorted(df_pairs["dataset"].dropna().unique())
        coverages_pairs = sorted(df_pairs["coverage"].dropna().unique())
        targets_pairs = sorted(list(set(r["target"] for r in results_pairs)))
        for target in targets_pairs:
            lines.extend(summarize_target(results_pairs, target, datasets_pairs, coverages_pairs))

    if results_opt:
        lines.append("")
        lines.append("============================================================")
        lines.append("Optimization Significance Analysis (Opt vs Baseline)")
        lines.append("")
        lines.append("Detailed results (OPT)")
        lines.extend(result_line(result) for result in results_opt)

        lines.append("")
        lines.append("Dataset-wide optimization significance summary (OPT)")
        datasets_opt = sorted(df_opt["dataset"].dropna().unique())
        coverages_opt = sorted(df_opt["coverage"].dropna().unique())
        targets_opt = sorted(list(set(r["target"] for r in results_opt)))
        for target in targets_opt:
            lines.extend(summarize_target(results_opt, target, datasets_opt, coverages_opt))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    df_ofat = load_results(RESULTS_DIR, "experiment_results_")
    results_ofat = []

    if not df_ofat.empty:
        datasets = sorted(df_ofat["dataset"].dropna().unique())
        coverages = sorted(df_ofat["coverage"].dropna().unique())

        for dataset in datasets:
            for coverage in coverages:
                mw_result = mann_whitney_strategy_test(df_ofat, dataset, coverage)
                if mw_result:
                    results_ofat.append(mw_result)

                for param in PARAMETERS:
                    kw_result = kruskal_wallis_param_test(df_ofat, dataset, coverage, param)
                    if kw_result:
                        results_ofat.append(kw_result)

    df_pairs = load_results(RESULTS_DIR, "pairs_")
    results_pairs = []

    if not df_pairs.empty:
        datasets = sorted(df_pairs["dataset"].dropna().unique())
        coverages = sorted(df_pairs["coverage"].dropna().unique())

        for dataset in datasets:
            for coverage in coverages:
                mask = (df_pairs["dataset"] == dataset) & (df_pairs["coverage"] == coverage)
                subset = df_pairs[mask]

                varying_params = [p for p in PARAMETERS if subset[p].nunique() > 1]

                if len(varying_params) == 2:
                    p1, p2 = varying_params[0], varying_params[1]
                    anova_result = two_way_anova_test(subset, dataset, coverage, p1, p2)
                    if anova_result:
                        results_pairs.append(anova_result)

    df_opt = load_results(RESULTS_DIR, "opt_results_")
    results_opt = []

    if not df_opt.empty:
        datasets = sorted(df_opt["dataset"].dropna().unique())
        coverages = sorted(df_opt["coverage"].dropna().unique())

        for dataset in datasets:
            for coverage in coverages:
                mask = (df_opt["dataset"] == dataset) & (df_opt["coverage"] == coverage)
                subset = df_opt[mask]

                opt_combos = subset[subset["combo_name"] != "baseline"]["combo_name"].unique()
                for opt_combo in opt_combos:
                    mw_opt_result = mann_whitney_opt_test(df_opt, dataset, coverage, opt_combo)
                    if mw_opt_result:
                        results_opt.append(mw_opt_result)

    write_report(df_ofat, results_ofat, df_pairs, results_pairs, df_opt, results_opt, OUTPUT_PATH)
    print(f"Report successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()