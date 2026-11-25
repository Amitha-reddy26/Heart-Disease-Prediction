# Heart Disease Prediction Using Machine Learning

This project is a new portfolio implementation of my previous academic project on heart disease prediction, focusing on machine learning, predictive analytics, model evaluation, and explainable AI.
## Dataset
Use the public [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), which describes 918 observations and 11 predictors. Download `heart.csv` and place it in `data/`. The dataset is not bundled; consult its source terms.

## Approach
- Treat zero cholesterol and resting blood pressure as missing; scale numeric features and apply KNN imputation.
- Impute and one-hot encode categorical features; use SMOTE on the training folds only (ordinary SMOTE on encoded categories is a baseline limitation).
- Compare nine classifiers: logistic regression, Gaussian naive Bayes, random forest, gradient boosting, extra trees, decision tree, KNN, SVM and MLP.
- Select the model by five-fold stratified training cross-validation ROC-AUC; report held-out accuracy and ROC-AUC.
- Optionally generate a SHAP feature importance plot for the selected model.

## Run locally (Python 3.11 recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
# Place downloaded heart.csv in data/
python -m src.train --data data/heart.csv --explain
python -m pytest -q
```

## Outputs
The model training pipeline generates the following files:

- `outputs/metrics.csv` - Model performance and evaluation metrics.
- `outputs/model.joblib` - Trained machine learning model.
- `outputs/shap.png` - SHAP feature importance visualization (optional).

Models are evaluated using accuracy, ROC-AUC, and cross-validation scores.

## Limitations
- Model performance depends on dataset quality and diversity.
- Potential sampling bias may affect model generalization.
- Further validation is required before real-world clinical implementation.
- This project is intended for educational and research purposes.
