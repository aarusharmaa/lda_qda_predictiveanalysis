

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
)

RANDOM_STATE = 42

# 1. Load dataset
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target  # 0 = malignant, 1 = benign
class_names = ["malignant", "benign"]

print("=" * 70)
print("LDA AND QDA: PREDICTIVE ANALYSIS")
print("=" * 70)
print(f"Dataset shape: {X.shape}")
print("\nTarget distribution:")
print(y.map({0: "malignant", 1: "benign"}).value_counts())
print("\nFirst five rows:")
print(X.head())

# 2. Split data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# LDA and QDA are fitted independently. Scaling is placed inside each pipeline
# so preprocessing is learned only from the training data.
models = {
    "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)),
}

# 3. Train, evaluate, and cross-validate
rows = []
predictions = {}
probabilities = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    predictions[name] = y_pred
    probabilities[name] = y_prob

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    rows.append({
        "Model": name,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Precision (benign)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (benign)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 (benign)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "5-fold CV Accuracy (mean)": cv_scores.mean(),
        "5-fold CV Accuracy (std)": cv_scores.std(),
    })

results = pd.DataFrame(rows).set_index("Model")
print("\n\nMODEL COMPARISON")
print(results.round(4).to_string())

for name in models:
    print(f"\n\n{name} CLASSIFICATION REPORT")
    print(classification_report(y_test, predictions[name], target_names=class_names, zero_division=0))
    print(f"{name} confusion matrix (rows=actual, columns=predicted; malignant, benign):")
    print(confusion_matrix(y_test, predictions[name]))

# 4. Save the numeric results
results.to_csv("model_comparison.csv")
print("\nSaved model_comparison.csv")

# 5. Visualize model metrics
ax = results[["Test Accuracy", "F1 (benign)", "ROC-AUC"]].plot(
    kind="bar", figsize=(9, 5), ylim=(0, 1.05)
)
ax.set_title("LDA vs QDA: Test-set performance")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=160)
plt.show()

# 6. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, predictions[name])
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
plt.savefig("confusion_matrices.png", dpi=160)
plt.show()

# 7. ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, model in models.items():
    RocCurveDisplay.from_predictions(y_test, probabilities[name], name=name, ax=ax)
ax.set_title("ROC Curves: LDA vs QDA")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=160)
plt.show()

# 8. Demonstrate a single prediction using the first held-out record.
sample = X_test.iloc[[0]]
actual = class_names[int(y_test.iloc[0])]
print("\nSINGLE-RECORD DEMONSTRATION")
print(f"Actual class: {actual}")
for name, model in models.items():
    predicted_idx = int(model.predict(sample)[0])
    benign_probability = float(model.predict_proba(sample)[0, 1])
    print(f"{name}: predicted={class_names[predicted_idx]}, "
          f"P(benign)={benign_probability:.3f}, P(malignant)={1-benign_probability:.3f}")

print("\nInterpretation reminder:")
print("- LDA estimates one shared covariance matrix, leading to linear decision boundaries.")
print("- QDA estimates a separate covariance matrix per class, allowing quadratic boundaries.")
print("- QDA is more flexible but estimates more parameters and may overfit on small datasets.")
print("- This is an educational demonstration, not a medical diagnostic tool.")
