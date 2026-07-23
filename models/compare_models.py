import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

xgb_df = pd.read_csv(os.path.join(RESULTS_DIR, "xgboost_iterations_summary.csv"))
lr_df = pd.read_csv(os.path.join(RESULTS_DIR, "logistic_iterations_summary.csv"))

comparison = pd.DataFrame({
    "Modele": ["XGBoost", "Logistic Regression"],
    "AUC-PR median": [xgb_df["auc_pr"].median(), lr_df["auc_pr"].median()],
    "AUC-PR [min,max]": [
        f"[{xgb_df['auc_pr'].min():.3f}, {xgb_df['auc_pr'].max():.3f}]",
        f"[{lr_df['auc_pr'].min():.3f}, {lr_df['auc_pr'].max():.3f}]",
    ],
    "ROC-AUC median": [xgb_df["roc_auc"].median(), lr_df["roc_auc"].median()],
    "Precision(1) median": [xgb_df["precision_class1"].median(), lr_df["precision_class1"].median()],
    "Recall(1) median": [xgb_df["recall_class1"].median(), lr_df["recall_class1"].median()],
})
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

data_auc_pr = [xgb_df["auc_pr"], lr_df["auc_pr"]]
axes[0].boxplot(data_auc_pr)
axes[0].set_xticklabels(["XGBoost", "Logistic Regression"])
axes[0].set_title("AUC-PR sur 10 iterations")
axes[0].set_ylabel("AUC-PR")
axes[0].grid(alpha=0.3)

data_roc = [xgb_df["roc_auc"], lr_df["roc_auc"]]
axes[1].boxplot(data_roc)
axes[1].set_xticklabels(["XGBoost", "Logistic Regression"])
axes[1].set_title("ROC-AUC sur 10 iterations")
axes[1].set_ylabel("ROC-AUC")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_boxplot.png"), dpi=150)
print(f"\nGraphe sauvegarde dans {FIGURES_DIR}/model_comparison_boxplot.png")
