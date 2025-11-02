# -*- coding: utf-8 -*-
"""
Train and save the mushroom classification model
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Fetch dataset
mushroom = fetch_ucirepo(id=73)

# Data (as pandas dataframes)
X = mushroom.data.features
X = X[['gill-size', 'odor', 'gill-spacing', 'stalk-surface-above-ring', 'spore-print-color', 'stalk-root']]
y = mushroom.data.targets.squeeze()  # Ensure y is 1D

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Identify categorical columns
categorical_features = X.columns

# Create a ColumnTransformer to apply OneHotEncoder to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply the ColumnTransformer to the training and testing data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("Original X_train shape:", X_train.shape)
print("Encoded X_train shape:", X_train_encoded.shape)

# Train the model
clf = SGDClassifier(
    loss='log_loss',        # Enables logistic regression behavior
    penalty='elasticnet',   # Better generalization (mix of L1/L2)
    alpha=1e-4,             # Regularization strength
    l1_ratio=0.15,
    class_weight={'e': 1, 'p': 5},  # Penalize misclassifying poisonous as edible
    max_iter=5000,
    random_state=42
)


clf.fit(X_train_encoded, y_train)

# Test the model
y_pred = clf.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the trained model and preprocessor
model_path = Path(__file__).parent / 'mushroom_model.pkl'
preprocessor_path = Path(__file__).parent / 'mushroom_preprocessor.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"\nModel saved to: {model_path}")
print(f"Preprocessor saved to: {preprocessor_path}")

# Save feature names for reference
feature_names = X.columns.tolist()
feature_names_path = Path(__file__).parent / 'mushroom_features.pkl'
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_names, f)

print(f"Feature names saved to: {feature_names_path}")
