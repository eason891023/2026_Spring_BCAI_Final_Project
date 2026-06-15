import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "data" / "results" / "metrics" / "summary_metrics.json"
OUT_DIR = ROOT / "data" / "results" / "permuted_cms_report"
PLOT_DIR = OUT_DIR / "plots"
SEEDS = {42, 123, 7}


def load_runs():
    with METRICS_JSON.open("r") as f:
        data = json.load(f)

    rows = []
    for run in data:
        if run.get("dataset") != "permuted":
            continue
        if run.get("model") == "sgcms":
            continue
        rows.append(
            {
                "model": run["model"],
                "optimizer": run["optimizer"],
                "f": int(run["f"]),
                "seed": int(run["seed"]),
                "avg_acc": float(run["CIL"]["Average_ACC"]),
                "forgetting": float(run["CIL"]["Forgetting"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No permuted runs found in {METRICS_JSON}")

    return df.drop_duplicates(
        subset=["model", "optimizer", "f", "seed"], keep="last"
    )


def required_conditions():
    conditions = []

    for opt in ["SGD", "Adam", "M3S"]:
        conditions.append(("plot1_baseline_mlp_optimizers", "baseline", opt, 20))

    for model, freq in [
        ("baseline", 20),
        ("scms", 20),
        ("icms", 1),
        ("icms", 20),
    ]:
        conditions.append(("plot2_cms_architecture_sgd", model, "SGD", freq))

    for model in ["baseline", "scms", "icms"]:
        for opt in ["SGD", "Adam", "M3S"]:
            conditions.append((f"plot3_{model}_optimizer_comparison", model, opt, 20))

    for opt in ["SGD", "Adam", "M3S"]:
        for freq in [20, 1000, 5000, 10000]:
            conditions.append((f"plot4_scms_slow_frequency_{opt.lower()}", "scms", opt, freq))

    return conditions


def summarize(df, conditions):
    rows = []
    missing = []

    for group, model, optimizer, freq in conditions:
        sub = df[
            (df["model"] == model)
            & (df["optimizer"] == optimizer)
            & (df["f"] == freq)
            & (df["seed"].isin(SEEDS))
        ].copy()
        seeds = set(sub["seed"].tolist())
        if seeds != SEEDS:
            missing.append(
                {
                    "group": group,
                    "model": model,
                    "optimizer": optimizer,
                    "f": freq,
                    "found_seeds": sorted(seeds),
                    "missing_seeds": sorted(SEEDS - seeds),
                }
            )
            continue

        rows.append(
            {
                "group": group,
                "model": model,
                "optimizer": optimizer,
                "f": freq,
                "n": len(sub),
                "seeds": ",".join(str(s) for s in sorted(seeds)),
                "avg_acc_mean": sub["avg_acc"].mean(),
                "avg_acc_std": sub["avg_acc"].std(ddof=1),
                "forgetting_mean": sub["forgetting"].mean(),
                "forgetting_std": sub["forgetting"].std(ddof=1),
            }
        )

    if missing:
        details = "\n".join(str(item) for item in missing)
        raise RuntimeError(f"Missing required seed coverage:\n{details}")

    return pd.DataFrame(rows)


def write_tables(summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "permuted_cms_summary.csv"
    md_path = OUT_DIR / "permuted_cms_summary.md"

    summary.to_csv(csv_path, index=False)

    ordered_groups = [
        "plot1_baseline_mlp_optimizers",
        "plot2_cms_architecture_sgd",
        "plot3_baseline_optimizer_comparison",
        "plot3_scms_optimizer_comparison",
        "plot3_icms_optimizer_comparison",
        "plot4_scms_slow_frequency_sgd",
        "plot4_scms_slow_frequency_adam",
        "plot4_scms_slow_frequency_m3s",
    ]

    def to_markdown_table(frame):
        headers = list(frame.columns)
        rows = frame.astype(str).values.tolist()
        widths = [
            max(len(str(header)), *(len(str(row[i])) for row in rows))
            for i, header in enumerate(headers)
        ]

        def fmt_row(values):
            return "| " + " | ".join(
                str(value).ljust(widths[i]) for i, value in enumerate(values)
            ) + " |"

        divider = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([fmt_row(headers), divider, *(fmt_row(row) for row in rows)])

    lines = [
        "# Permuted MNIST CMS Report",
        "",
        "All rows use three seeds: 7, 42, 123. SG-CMS is intentionally excluded.",
        "",
    ]
    for group in ordered_groups:
        sub = summary[summary["group"] == group].copy()
        if sub.empty:
            continue
        lines.append(f"## {group}")
        display = sub[
            [
                "model",
                "optimizer",
                "f",
                "n",
                "avg_acc_mean",
                "avg_acc_std",
                "forgetting_mean",
                "forgetting_std",
            ]
        ].copy()
        for col in ["avg_acc_mean", "avg_acc_std", "forgetting_mean", "forgetting_std"]:
            display[col] = display[col].map(lambda x: f"{x:.4f}")
        lines.append(to_markdown_table(display))
        lines.append("")

    md_path.write_text("\n".join(lines))
    return csv_path, md_path


def barplot(summary, group, filename, title, label_func):
    sub = summary[summary["group"] == group].copy()
    sub["label"] = sub.apply(label_func, axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    palette = sns.color_palette("Set2", n_colors=len(sub))
    ax.bar(
        sub["label"],
        sub["avg_acc_mean"],
        yerr=sub["avg_acc_std"],
        capsize=5,
        color=palette,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_title(title)
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    for i, row in enumerate(sub.itertuples()):
        ax.text(
            i,
            min(row.avg_acc_mean + row.avg_acc_std + 0.035, 0.98),
            f"{row.avg_acc_mean:.3f}\n±{row.avg_acc_std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300)
    plt.close(fig)


def lineplot_frequency(summary, optimizer):
    group = f"plot4_scms_slow_frequency_{optimizer.lower()}"
    sub = summary[summary["group"] == group].sort_values("f")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(
        sub["f"].astype(str),
        sub["avg_acc_mean"],
        yerr=sub["avg_acc_std"],
        marker="o",
        linewidth=2.2,
        capsize=5,
        color="#356f8f",
    )
    ax.set_title(f"SCMS slow-frequency sweep + {optimizer}")
    ax.set_xlabel("Slow update frequency f")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(sub.itertuples()):
        ax.text(
            i,
            min(row.avg_acc_mean + row.avg_acc_std + 0.035, 0.98),
            f"{row.avg_acc_mean:.3f}\n±{row.avg_acc_std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"plot4_scms_slow_frequency_{optimizer.lower()}.png", dpi=300)
    plt.close(fig)


def make_plots(summary):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)

    barplot(
        summary,
        "plot1_baseline_mlp_optimizers",
        "plot1_baseline_mlp_optimizers.png",
        "Baseline MLP optimizer comparison",
        lambda r: r["optimizer"],
    )

    def plot2_label(row):
        if row["model"] == "icms" and row["f"] == 1:
            return "Independent arch\nbaseline"
        if row["model"] == "icms":
            return "ICMS"
        return row["model"].upper() if row["model"] != "baseline" else "Baseline"

    barplot(
        summary,
        "plot2_cms_architecture_sgd",
        "plot2_cms_architecture_sgd.png",
        "CMS architecture comparison + SGD",
        plot2_label,
    )

    for model in ["baseline", "scms", "icms"]:
        title_model = model.upper() if model != "baseline" else "Baseline"
        barplot(
            summary,
            f"plot3_{model}_optimizer_comparison",
            f"plot3_{model}_optimizer_comparison.png",
            f"{title_model} optimizer comparison",
            lambda r: r["optimizer"],
        )

    for optimizer in ["SGD", "Adam", "M3S"]:
        lineplot_frequency(summary, optimizer)


def main():
    df = load_runs()
    conditions = required_conditions()
    summary = summarize(df, conditions)
    csv_path, md_path = write_tables(summary)
    make_plots(summary)

    print(f"Checked {len(conditions)} required conditions.")
    print("All required conditions have seeds: 7, 42, 123.")
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary Markdown: {md_path}")
    print(f"Wrote plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
