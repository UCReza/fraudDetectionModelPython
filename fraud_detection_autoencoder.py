import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# Loading Dataset
print("Loading dataset...")
df = pd.read_csv("creditcard.csv") 
print(f"Dataset shape: {df.shape}")

# Preprocessing
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalizing numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Spliting train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


# Building AutoEncoder Model
print("Training AutoEncoder model...")
clf = AutoEncoder()
clf.fit(X_train)

# Predictions
y_train_pred = clf.labels_  # Training predictions
y_train_scores = clf.decision_scores_

y_test_pred = clf.predict(X_test)  # 0 = normal, 1 = fraud
y_test_scores = clf.decision_function(X_test)


# Evaluation
print("\nModel Evaluation on Test Set:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


# Visualization
plt.figure(figsize=(8, 5))
sns.histplot(y_test_scores, bins=50, kde=True)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()
