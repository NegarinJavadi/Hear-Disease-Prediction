import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay)

# 1. Load Dataset 
print("Heart Disease Prediction")
print("\nLoading dataset from UCI repository...")

heart = fetch_ucirepo(id=45)
X = heart.data.features
y = heart.data.targets

print(f"\nDataset info:")
print(f" Samples: {X.shape[0]}")
print(f" Features: {X.shape[1]}")
print(f"\nFeature names:\n  {list(X.columns)}")
print(f"\nTarget distribution:\n{y.value_counts().to_string()}")

# 2. Preprocessing
y = y.values.ravel()
y = (y > 0).astype(int)

print(f"\nBinary target: 0=No disease ({(y==0).sum()})  1=Disease ({(y==1).sum()})")

# 3. Feature Engineering
X = X.copy()

if "age" in X.columns and "chol" in X.columns:
    X["age_chol_ratio"] = X["age"] / (X["chol"] + 1)

if "thalach" in X.columns and "age" in X.columns:
    X["thalach_age"] = X["thalach"] / (X["age"] + 1)

print(f"\nAfter feature engineering: {X.shape[1]} features")

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Define Models
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

models = {
    "Logistic Regression":  make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    "KNN":                  make_pipeline(KNeighborsClassifier(n_neighbors=7)),
    "SVM":                  make_pipeline(SVC(kernel="rbf", C=10, probability=True, random_state=42)),
    "Random Forest":        make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Gradient Boosting":    make_pipeline(GradientBoostingClassifier(n_estimators=100, random_state=42))}

# 6. Train and Evaluate All Models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "test_acc": pipeline.score(X_test, y_test),
        "auc":      auc,
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "y_proba":  y_proba}

    print(f"\n{name}")
    print(f"  CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Test Acc    : {pipeline.score(X_test, y_test):.3f}")
    print(f"  AUC-ROC     : {auc:.3f}")
    print(classification_report(y_test, y_pred,target_names=["No Disease", "Disease"],zero_division=0))

# 7. Find Best Model
best_name = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest model by AUC: {best_name} ({results[best_name]['auc']:.3f})")

# 8. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project — Heart Disease Prediction", fontsize=14, fontweight="bold")

# Plot 1: CV Accuracy comparison bar chart
names = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds  = [results[n]["cv_std"]  for n in names]
colors   = ["#2ecc71" if n == best_name else "#3498db" for n in names]

axes[0].barh(names, cv_means, xerr=cv_stds, color=colors, edgecolor="white", height=0.6)
axes[0].set_xlabel("CV Accuracy")
axes[0].set_title("Cross-Validation Accuracy (5-fold)")
axes[0].set_xlim(0.5, 1.0)
axes[0].axvline(0.8, color="gray", linestyle="--", linewidth=0.8)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    axes[0].text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)

# Plot 2: Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title(f"Confusion Matrix — {best_name}")

# Plot 3: ROC curves for all models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})", linewidth=1.5)
axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curves — All Models")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("project_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as project_results.png")

# 9. Feature Importance (Random Forest)
rf_pipeline = models["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]
feat_names = X.columns.tolist()

importances = pd.Series(rf_model.feature_importances_, index=feat_names)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#e74c3c" if v >= importances.median() else "#fadbd8" for v in importances]
ax.barh(importances.index, importances.values, color=colors)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importances — Random Forest")
ax.axvline(importances.median(), color="gray", linestyle="--", linewidth=0.8, label="median")
ax.legend()
plt.tight_layout()
plt.savefig("project_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved as project1_features.png")