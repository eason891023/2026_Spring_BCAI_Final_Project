import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
METRICS_DIR = os.path.join("data", "results", "metrics")
PLOTS_DIR = os.path.join("data", "results", "plots")

# Ensure output directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set Seaborn style for publication-ready plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generate_summary_table():
    """Reads the JSON log and prints a formatted comparison table."""
    json_path = os.path.join(METRICS_DIR, "summary_metrics.json")
    if not os.path.exists(json_path):
        print(f"[WARNING] No summary JSON found at {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Flatten the nested JSON into a format Pandas loves
    records = []
    for run in data:
        base_info = {
            # .get keeps backwards compatibility with pre--dataset-flag entries
            "Dataset": run.get("dataset", "split"),
            "Seed": run.get("seed", "n/a"),
            "Model": run["model"],
            "Optimizer": run["optimizer"],
            "f": run["f"],
            "Stabilized": run["stabilized"],
            "Alpha": run["alpha"],
            "Beta3": run["beta3"]
        }
        # Add CIL Metrics
        for metric, val in run["CIL"].items():
            records.append({**base_info, "Evaluation": "Class-IL", "Metric": metric, "Value": val})
        # Add TIL Metrics (absent for Domain-IL runs)
        if run.get("TIL") is not None:
            for metric, val in run["TIL"].items():
                records.append({**base_info, "Evaluation": "Task-IL", "Metric": metric, "Value": val})

    df = pd.DataFrame(records)

    # Pivot for a beautiful terminal view (mean over seeds)
    pivot_df = df.pivot_table(
        index=["Dataset", "Model", "Optimizer", "f"],
        columns=["Evaluation", "Metric"],
        values="Value"
    ).round(4)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(pivot_df.to_string())
    print("="*80 + "\n")
    
    return df

def plot_bar_charts(df):
    """Generates bar charts comparing Average Accuracy and Forgetting across models/optimizers."""
    if df is None or df.empty:
        return

    # One figure per dataset x evaluation so CIL and DIL results are never mixed
    for (dataset, evaluation) in df[["Dataset", "Evaluation"]].drop_duplicates().itertuples(index=False):
        for metric in ["Average_ACC", "Forgetting"]:
            subset = df[(df["Metric"] == metric) & (df["Dataset"] == dataset) & (df["Evaluation"] == evaluation)]
            if subset.empty:
                continue

            plt.figure(figsize=(10, 6))
            # errorbar='sd' shows the spread across seeds
            ax = sns.barplot(
                data=subset,
                x="Optimizer",
                y="Value",
                hue="Model",
                errorbar="sd",
                palette="viridis"
            )

            plt.title(f"{dataset} ({evaluation}) — {metric}", pad=20, fontweight='bold')
            plt.ylabel(metric)
            plt.xlabel("Optimizer")
            plt.legend(title="Architecture")
            plt.tight_layout()

            eval_tag = evaluation.replace("-", "")
            plt.savefig(os.path.join(PLOTS_DIR, f"bar_chart_{dataset}_{eval_tag}_{metric}.png"), dpi=300)
            plt.close()
    print("[INFO] Bar charts generated.")

def plot_learning_dynamics():
    """Generates Line Graphs from history and Heatmaps from raw CSV matrices."""
    
    # --- 1. GENERATE HEATMAPS (From standard Task x Task matrices) ---
    # We use a strict match to avoid grabbing the new history files
    matrix_files = [f for f in glob.glob(os.path.join(METRICS_DIR, "*.csv")) if "history" not in f]
    
    for file in matrix_files:
        filename = os.path.basename(file)
        name_parts = filename.replace(".csv", "").split("_")
        eval_type = name_parts[-1] # CIL or TIL
        exp_name = "_".join(name_parts[:-1])
        
        R = np.loadtxt(file, delimiter=",")
        num_tasks = R.shape[0]
        tasks = [f"Task {i+1}" for i in range(num_tasks)]
        
        plt.figure(figsize=(7, 6))
        mask = np.triu(np.ones_like(R, dtype=bool), k=1)
        sns.heatmap(R, mask=mask, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=tasks, yticklabels=tasks, vmin=0, vmax=1)
        
        plt.title(f"{exp_name} ({eval_type})\nFinal Accuracy Matrix (R)", pad=15, fontweight='bold')
        plt.xlabel("Evaluated On")
        plt.ylabel("Trained Up To")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"heatmap_{exp_name}_{eval_type}.png"), dpi=300)
        plt.close()

    # --- 2. GENERATE HIGH-RES LINE GRAPHS (From Epoch x Task history matrices) ---
    history_files = glob.glob(os.path.join(METRICS_DIR, "*_history.csv"))
    
    for file in history_files:
        filename = os.path.basename(file)
        name_parts = filename.replace("_history.csv", "").split("_")
        eval_type = name_parts[-1] # CIL or TIL
        exp_name = "_".join(name_parts[:-1])
        
        # Read step-based history matrix where column 0 is global_step
        H = np.loadtxt(file, delimiter=",")
        if H.ndim == 1:
            H = H.reshape(1, -1) # Handle edge case of single row
            
        global_steps = H[:, 0]
        num_tasks = H.shape[1] - 1
        
        def plot_history(smoothed=False):
            plt.figure(figsize=(10, 6))
            window_size = 5 # Sliding window size for smoothing
        
            # Plot each task's accuracy curve
            for task_idx in range(num_tasks):
                y_vals = H[:, task_idx + 1].copy()
                
                # Find where this task actually starts being evaluated
                valid_mask = ~np.isnan(y_vals)
                if not np.any(valid_mask):
                    continue
                first_valid_idx = np.argmax(valid_mask)
                
                if smoothed:
                    active_vals = y_vals[first_valid_idx:]
                    smoothed_active = pd.Series(active_vals).rolling(window=window_size, min_periods=1).mean().values
                    y_vals[first_valid_idx:] = smoothed_active
                
                plt.plot(global_steps, y_vals, linewidth=2.5, label=f'Task {task_idx + 1}')

            # Draw vertical lines to mark Task Boundaries based on absolute step count
            for i in range(1, num_tasks):
                y_vals = H[:, i + 1]
                valid_mask = ~np.isnan(y_vals)
                if np.any(valid_mask):
                    first_valid_idx = np.argmax(valid_mask)
                    boundary = global_steps[first_valid_idx]
                    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)
                    plt.text(boundary, 1.02, f'Start T{i+1}', rotation=0, 
                             ha='center', va='bottom', color='gray', fontsize=9, fontweight='bold')

            title_suffix = " (Smoothed)" if smoothed else ""
            plt.title(f"{exp_name} ({eval_type})\nStep-Level Learning Dynamics{title_suffix}", pad=25, fontweight='bold')
            plt.xlabel("Global Step")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1.1)
            plt.xlim(0, np.max(global_steps) if len(global_steps) > 0 else 1)
        
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            suffix = "_smoothed" if smoothed else ""
            plt.savefig(os.path.join(PLOTS_DIR, f"linegraph_highres{suffix}_{exp_name}_{eval_type}.png"), dpi=300)
            plt.close()
            
        plot_history(smoothed=False)
        plot_history(smoothed=True)
        
    print(f"[INFO] Learning dynamics generated: {len(matrix_files)} Heatmaps, {len(history_files)} High-Res Line Graphs.")

def main():
    print("Starting Analysis Pipeline...")
    df = generate_summary_table()
    plot_bar_charts(df)
    plot_learning_dynamics()
    print(f"\nAnalysis complete! Check the '{PLOTS_DIR}' directory for your visualizations.")

if __name__ == "__main__":
    main()