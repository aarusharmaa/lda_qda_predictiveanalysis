# LDA and QDA Predictive Analysis

## Project overview
This project compares Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) on the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn. The target is binary classification: malignant vs benign.

**Important:** This is an educational machine-learning project, not a clinical diagnostic tool.

## Research question
Which model performs better on held-out data, and how do their different assumptions affect predictive performance?

## Algorithms
- **LDA:** Assumes classes share a covariance matrix. Its decision boundary is linear.
- **QDA:** Estimates a separate covariance matrix for each class. Its decision boundary can be quadratic, making it more flexible but potentially more prone to overfitting.

## Evaluation
The script uses a stratified 80/20 train/test split and 5-fold stratified cross-validation on the training set. It reports test accuracy, precision, recall, F1-score, ROC-AUC, classification reports, and confusion matrices. It creates charts for metric comparison, confusion matrices, and ROC curves.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python lda_qda_analysis.py
```

## Output files
- `model_comparison.csv`
- `model_comparison.png`
- `confusion_matrices.png`
- `roc_curves.png`

## Suggested project-report structure
1. Abstract
2. Introduction and problem statement
3. Dataset description
4. Methodology and preprocessing
5. LDA and QDA theory
6. Experimental setup
7. Results and visualizations
8. Discussion and limitations
9. Conclusion
10. References

## Limitations
- Results depend on the selected train/test split and regularization setting.
- The dataset is relatively small and has correlated features.
- A higher score on this split does not prove general superiority.
- Any real medical use would require extensive clinical validation, governance, and expert oversight.
