import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the test dataset and the saved model
preprocessed_data_path = "data/preprocessed_data.csv"
model_path = "models/lgbm_model.pkl"

data = pd.read_csv(preprocessed_data_path)
model = joblib.load(model_path)

# Split the dataset into features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Split the dataset into train and test sets
train_ratio = 0.8
split_index = int(X.shape[0] * train_ratio)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt="d", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
