# diabetes_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 🔁 Replace this with your actual file path
data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Target column (ground truth)
y = data.iloc[:, 0]

# Combine prediabetic (1.0) and diabetic (2.0) into a single class (1.0)
y = y.replace({2.0: 1.0})

# Features (everything except the first column)
x = data.iloc[:, 1:]

# Quick data check
print("First few target values:\n", y.head())
print("\nFirst few rows of features:\n", x.head())

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
sampling_strategy = {
    0: len(y_train[y_train == 0]),
    1: int(len(y_train[y_train == 1]) * 6)
}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Train a linear SVM using SGDClassifier
sgd_svm = SGDClassifier(
    loss='hinge',
    tol=1e-4,
    max_iter=1000,
    verbose=1,
    class_weight='balanced',
    random_state=42
)

sgd_svm.fit(X_train_res, y_train_res)

# Evaluate on test set
y_pred = sgd_svm.predict(X_test_scaled)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Save the model and scalar
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(sgd_svm, 'sgd_svm_model.pkl')
print("✅ Model saved successfully as 'sgd_svm_model.pkl'")
