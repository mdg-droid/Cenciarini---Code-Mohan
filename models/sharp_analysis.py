import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "training_set_italy.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from preprocessing import load_training_data

X, y, names = load_training_data(DATA_PATH)

pipeline = joblib.load(os.path.join(RESULTS_DIR, "xgboost_final_model.pkl"))
model = pipeline["model"]
feature_columns = pipeline["feature_columns"]

X = X[feature_columns]

print(f"Calcul des valeurs SHAP sur {len(X)} entreprises...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_final_model.png"), dpi=150, bbox_inches="tight")
print(f"Graphe SHAP sauvegarde : {FIGURES_DIR}/shap_summary_final_model.png")

mean_abs_shap = pd.Series(
    abs(shap_values).mean(axis=0), index=X.columns
).sort_values(ascending=False)

print("\nTop 20 features par importance SHAP moyenne (valeur absolue) :")
print(mean_abs_shap.head(20))

mean_abs_shap.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"))
print(f"\nClassement complet sauvegarde : {RESULTS_DIR}/shap_feature_importance.csv")
