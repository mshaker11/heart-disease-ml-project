# Heart Disease Prediction using Logistic Regression and Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("cardio_train.csv")

# Simple EDA
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(14, 10))
plt.suptitle("Feature Distributions")
plt.show()

# Prepare features and labels
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Metric function
def show_metrics(name, y_true, y_pred, y_prob):
    print(f"\n{name} Results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

show_metrics("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
show_metrics("Random Forest", y_test, y_pred_rf, y_prob_rf)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve Comparison")
plt.show()

# Feature Importance
importances = rf.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance")
plt.show()
