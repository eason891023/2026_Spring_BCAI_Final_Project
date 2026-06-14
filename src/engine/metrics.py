import numpy as np
import os

class CLEvaluator:
    def __init__(self, num_tasks=5):
        self.num_tasks = num_tasks
        
        # Initialize the T x T Accuracy Matrix (R) for standard task boundaries
        self.R = np.zeros((num_tasks, num_tasks))
        
        # Initialize dynamic history list for high-resolution tracking
        # Records will be tuples: (global_step, eval_task_id, accuracy)
        self.history_records = []
        
    def update_matrix(self, train_task_id, eval_task_id, accuracy):
        """Records accuracy at strict task boundaries."""
        self.R[train_task_id, eval_task_id] = accuracy
        
    def update_history(self, global_step, eval_task_id, accuracy):
        """Records accuracy at the specified global step."""
        self.history_records.append((global_step, eval_task_id, accuracy))
        
    def compute_metrics(self):
        """Calculates final Average ACC, BWT, and Average Forgetting."""
        T = self.num_tasks
        
        acc = np.mean(self.R[T-1, :])
        
        bwt = 0.0
        forgetting = 0.0
        
        if T > 1:
            bwt_sum = sum([self.R[T-1, i] - self.R[i, i] for i in range(T-1)])
            bwt = bwt_sum / (T - 1)
            f_j = [np.max(self.R[:T-1, j]) - self.R[T-1, j] for j in range(T-1)]
            forgetting = np.mean(f_j)
            
        return {
            "Accuracy_Matrix": self.R,
            "Average_ACC": acc,
            "BWT": bwt,
            "Forgetting": forgetting
        }

    def export_matrix_to_csv(self, filepath):
        """Exports the Accuracy Matrix (R)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, self.R, delimiter=",", fmt="%.4f")

    # History Export Method
    def export_history_to_csv(self, filepath):
        """Exports the high-resolution history matrix with global_step as the first column."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not self.history_records:
            return
            
        import pandas as pd
        df = pd.DataFrame(self.history_records, columns=['global_step', 'task_id', 'accuracy'])
        
        # Pivot so each row is a unique global_step and columns are task IDs
        pivot_df = df.pivot_table(index='global_step', columns='task_id', values='accuracy')
        
        # Ensure all columns from 0 to num_tasks-1 exist
        for t in range(self.num_tasks):
            if t not in pivot_df.columns:
                pivot_df[t] = np.nan
        
        # Sort columns to maintain order
        pivot_df = pivot_df[sorted(pivot_df.columns)]
        
        # Save to CSV (first column will be global_step, subsequent columns are accuracies)
        pivot_df.to_csv(filepath, header=False, na_rep='nan')

    def export_summary_dict(self):
        """Returns scalar metrics ready for JSON serialization."""
        metrics = self.compute_metrics()
        return {
            "Average_ACC": float(metrics["Average_ACC"]),
            "BWT": float(metrics["BWT"]),
            "Forgetting": float(metrics["Forgetting"])
        }