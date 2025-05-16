
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data.csv")

# Encode labels
le = LabelEncoder()
df["Risk Level"] = le.fit_transform(df["Risk Level"])

# Prepare features and labels
X = df.drop("Risk Level", axis=1)
y = df["Risk Level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Define models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(max_iter=500, random_state=42)
}

# Train, evaluate, and plot
model_names = []
accuracies = []
conf_matrices = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"{name}_model.pkl")

    cm = confusion_matrix(y_test, y_pred)
    model_names.append(name)
    accuracies.append(acc)
    conf_matrices.append((name, cm))

# Accuracy bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Confusion matrices
for name, cm in conf_matrices:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=["Low", "Medium", "High"], 
                yticklabels=["Low", "Medium", "High"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

print("\nAll models trained, evaluated, and visualized.")
