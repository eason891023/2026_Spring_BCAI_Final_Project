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
        # Add TIL Metrics
        for metric, val in run["TIL"].items():
            records.append({**base_info, "Evaluation": "Task-IL", "Metric": metric, "Value": val})

    df = pd.DataFrame(records)
    
    # Pivot for a beautiful terminal view
    pivot_df = df.pivot_table(
        index=["Model", "Optimizer", "f"], 
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

    # Filter for the key metrics
    for metric in ["Average_ACC", "Forgetting"]:
        plt.figure(figsize=(10, 6))
        subset = df[df["Metric"] == metric]
        
        # Create a grouped bar chart
        ax = sns.barplot(
            data=subset, 
            x="Optimizer", 
            y="Value", 
            hue="Model", 
            errorbar=None,
            palette="viridis"
        )
        
        plt.title(f"{metric} Comparison Across Architectures", pad=20, fontweight='bold')
        plt.ylabel(metric)
        plt.xlabel("Optimizer")
        plt.legend(title="Architecture")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOTS_DIR, f"bar_chart_{metric}.png"), dpi=300)
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
        
        H = np.loadtxt(file, delimiter=",")
        total_epochs, num_tasks = H.shape
        epochs_per_task = total_epochs // num_tasks
        
        plt.figure(figsize=(10, 6))
        
        # Plot each task's accuracy curve
        for task_idx in range(num_tasks):
            y_vals = H[:, task_idx]
            # Plot against absolute global epoch
            plt.plot(range(1, total_epochs + 1), y_vals, linewidth=2.5, label=f'Task {task_idx + 1}')

        # Draw vertical lines to mark Task Boundaries
        for i in range(1, num_tasks):
            boundary = i * epochs_per_task
            plt.axvline(x=boundary + 0.5, color='gray', linestyle='--', alpha=0.7)
            plt.text(boundary + 0.5, 1.02, f'Start T{i+1}', rotation=0, 
                     ha='center', va='bottom', color='gray', fontsize=9, fontweight='bold')

        plt.title(f"{exp_name} ({eval_type})\nEpoch-Level Learning Dynamics", pad=25, fontweight='bold')
        plt.xlabel("Global Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.xlim(1, total_epochs)
        
        # Format X-axis to show every epoch, but prioritize task boundaries
        plt.xticks(range(1, total_epochs + 1))
        
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"linegraph_highres_{exp_name}_{eval_type}.png"), dpi=300)
        plt.close()
        
    print(f"[INFO] Learning dynamics generated: {len(matrix_files)} Heatmaps, {len(history_files)} High-Res Line Graphs.")

def main():
    print("Starting Analysis Pipeline...")
    df = generate_summary_table()
    plot_bar_charts(df)
    plot_learning_dynamics()
    print(f"\nAnalysis complete! Check the '{PLOTS_DIR}' directory for your visualizations.")

if __name__ == "__main__":
    main()