import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/european_aerospace_ma_xgboost_v2.csv')


df = df.drop(
    columns=[
        'company_name',
        'status',
        'founding_year',

        # Fuites d'information
        'acquirer',
        'ma_date'
    ],
    errors='ignore'
)


X = df.drop(columns=['is_ma_target'])
y = df['is_ma_target']


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=42
)


numeric_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include='object').columns

median_values = X_train[numeric_cols].median()

for dataset in [X_train, X_val, X_test]:
    dataset[numeric_cols] = dataset[numeric_cols].fillna(median_values)

X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    drop_first=True
)

X_val = pd.get_dummies(
    X_val,
    columns=categorical_cols,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    drop_first=True
)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

training_feature_columns = X_train.columns

scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    early_stopping_rounds=30,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

import shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    max_display=20
)

y_pred_proba = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_pred_proba
)

f1 = 2 * precision * recall / (precision + recall + 1e-12)
best_threshold = thresholds[np.argmax(f1[:-1])]

y_pred = (y_pred_proba >= best_threshold).astype(int)

print("Best threshold:", round(best_threshold, 4))
print("AUC-PR:", round(average_precision_score(y_test, y_pred_proba), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 20 features:")
print(importance.head(20))

joblib.dump(
    {
        "model": model,
        "median_values": median_values,
        "training_columns": training_feature_columns,
    },
    "ma_xgboost_pipeline.pkl"
)



def predict_new_entrant_probability(new_data):
    new_df = pd.DataFrame([new_data])

    new_df = new_df.drop(
        columns=['company_name', 'status', 'founding_year'],
        errors='ignore'
    )

    for col in numeric_cols:
        if col in new_df.columns:
            new_df[col] = new_df[col].fillna(median_values[col])

    new_df = pd.get_dummies(
        new_df,
        columns=categorical_cols,
        drop_first=True
    )

    new_df = new_df.reindex(
        columns=training_feature_columns,
        fill_value=0
    )

    return model.predict_proba(new_df)[0, 1]
