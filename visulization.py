# -*- coding: utf-8 -*-
"""
Visualize the Mushroom Dataset
Shows distributions and relationships between features and classes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
print("Fetching mushroom dataset...")
mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets.squeeze()  # Convert to Series
df = X.copy()
df['class'] = y  # Add target column ('e' or 'p')

print(f"Dataset loaded successfully with {len(df)} samples and {len(df.columns)} columns.")
print(f"\nClass distribution:\n{df['class'].value_counts()}")

# -----------------------------------------------------------
# Basic class distribution (edible vs poisonous)
# -----------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='class', palette='coolwarm')
plt.title('Distribution of Edible vs Poisonous Mushrooms', fontsize=14, fontweight='bold')
plt.xlabel('Mushroom Type')
plt.ylabel('Count')
plt.xticks([0, 1], ['Edible (e)', 'Poisonous (p)'])
plt.tight_layout()
plt.savefig('data_exploration/class_distribution.png', dpi=300)
print("✓ Saved class distribution plot: class_distribution.png")
plt.close()

# -----------------------------------------------------------
# Visualize selected categorical features
# -----------------------------------------------------------
selected_features = ['odor', 'gill-size', 'gill-spacing', 'stalk-surface-above-ring', 'spore-print-color', 'stalk-root']

for feature in selected_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=feature, hue='class', palette='coolwarm')
    plt.title(f"{feature.capitalize()} vs Class", fontsize=13, fontweight='bold')
    plt.xlabel(feature.replace('-', ' ').capitalize())
    plt.ylabel('Count')
    plt.legend(title='Class', labels=['Edible (e)', 'Poisonous (p)'])
    plt.xticks(rotation=30)
    plt.tight_layout()
    filename = f"feature_{feature}.png"
    plt.savefig(f'data_exploration/{filename}', dpi=300)
    print(f"✓ Saved feature plot: {filename}")
    plt.close()

# -----------------------------------------------------------
# Correlation heatmap (based on one-hot encoding)
# -----------------------------------------------------------
print("\nCreating correlation heatmap...")

# One-hot encode categorical data for correlation analysis
df_encoded = pd.get_dummies(df, drop_first=True)

# Compute correlation with the target ('class_p' = poisonous)
corr = df_encoded.corr()['class_p'].sort_values(ascending=False)

plt.figure(figsize=(6, 10))
sns.heatmap(corr.to_frame(), annot=False, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation with Being Poisonous', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data_exploration/feature_correlation_heatmap.png', dpi=300)
print("✓ Saved feature correlation heatmap: feature_correlation_heatmap.png")

print("\nVisualization completed successfully!")
