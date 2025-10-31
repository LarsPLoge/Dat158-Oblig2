# -*- coding: utf-8 -*-
"""
Test the mushroom classification model
Evaluates model performance with various metrics
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    model_path = Path(__file__).parent / 'mushroom_model.pkl'
    preprocessor_path = Path(__file__).parent / 'mushroom_preprocessor.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print("✓ Model and preprocessor loaded successfully!\n")
        return model, preprocessor
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_mushroom_model.py first.")
        return None, None

def prepare_test_data():
    """Fetch and prepare test data"""
    print("Fetching dataset...")
    mushroom = fetch_ucirepo(id=73)
    
    X = mushroom.data.features
    # X = X[['gill-size', 'odor', 'gill-spacing', 'stalk-surface-above-ring', 'spore-print-color', 'stalk-root']]
    y = mushroom.data.targets.squeeze()
    
    # Use the same split as training to get the same test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"✓ Test set size: {len(y_test)} samples\n")
    return X_test, y_test

def evaluate_model(model, preprocessor, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    
    # Preprocess test data
    X_test_encoded = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_encoded)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='e')
    recall = recall_score(y_test, y_pred, pos_label='e')
    f1 = f1_score(y_test, y_pred, pos_label='e')
    
    # Print results
    print("=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\n📊 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🔍 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"⚖️  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print("\nNote: 'e' = edible, 'p' = poisonous\n")
    print(classification_report(y_test, y_pred, target_names=['Edible (e)', 'Poisonous (p)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['e', 'p'])
    
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("\n", cm)
    print("\nMatrix interpretation:")
    print(f"  True Positives (Edible correctly classified):  {cm[0][0]}")
    print(f"  False Negatives (Edible wrongly as Poisonous): {cm[0][1]}")
    print(f"  False Positives (Poisonous wrongly as Edible): {cm[1][0]}")
    print(f"  True Negatives (Poisonous correctly classified): {cm[1][1]}")
    
    # Calculate error rates
    total = len(y_test)
    correct = cm[0][0] + cm[1][1]
    incorrect = cm[0][1] + cm[1][0]
    
    print(f"\n  Total predictions: {total}")
    print(f"  Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, y_test, y_pred):
    """Create and save confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Edible', 'Poisonous']
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Confusion Matrix - Mushroom Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save the figure
    output_path = Path(__file__).parent / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix plot saved to: {output_path}")
    
    plt.close()

def analyze_feature_importance(model, preprocessor):
    """Analyze and visualize feature importance from the model"""
    
    # Load original feature names
    feature_names_path = Path(__file__).parent / 'mushroom_features.pkl'
    try:
        with open(feature_names_path, 'rb') as f:
            original_features = pickle.load(f)
    except FileNotFoundError:
        print("⚠️  Feature names file not found. Skipping feature importance analysis.")
        return
    
    # Get feature names from preprocessor
    try:
        encoded_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        print("⚠️  Cannot extract feature names from preprocessor.")
        return
    
    # Get coefficients from the model (SGDClassifier has coef_ attribute)
    if not hasattr(model, 'coef_'):
        print("⚠️  Model does not have coefficient attributes for feature importance.")
        return
    
    coefficients = model.coef_[0]  # Get coefficients for binary classification
    
    # Create a mapping of encoded features to original features
    feature_importance_dict = {}
    
    for idx, encoded_name in enumerate(encoded_feature_names):
        # Extract original feature name from one-hot encoded name
        # Format is typically: 'onehot__feature-name_value'
        if 'onehot__' in encoded_name:
            # Split by '__' and then by '_' to get the feature name
            parts = encoded_name.split('__')[1]  # Get part after 'onehot__'
            # The feature name is everything before the last underscore (which separates value)
            # Find the original feature by matching against known features
            original_feature = None
            for feat in original_features:
                if parts.startswith(feat + '_'):
                    original_feature = feat
                    break
            
            if original_feature is None:
                # If no match, use the part before last underscore
                original_feature = '_'.join(parts.split('_')[:-1]) if '_' in parts else parts
        else:
            original_feature = encoded_name
        
        # Aggregate importance by original feature
        if original_feature not in feature_importance_dict:
            feature_importance_dict[original_feature] = []
        feature_importance_dict[original_feature].append(abs(coefficients[idx]))
    
    # Calculate mean absolute importance for each original feature
    feature_importance = {
        feature: np.mean(importances) 
        for feature, importances in feature_importance_dict.items()
    }
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("\nTop 10 Most Important Features (by mean absolute coefficient):\n")
    
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"{i:2d}. {feature:30s} {importance:.4f}")
    
    print(f"\nTotal unique features analyzed: {len(sorted_features)}")
    print(f"Total encoded features: {len(encoded_feature_names)}")
    
    # Plot feature importance
    plot_feature_importance(sorted_features[:15])  # Top 15 features
    
    return sorted_features

def plot_feature_importance(sorted_features):
    """Create and save feature importance visualization"""
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Highest importance at the top
    ax.set_xlabel('Mean Absolute Coefficient', fontsize=12)
    ax.set_title('Top 15 Feature Importances - Mushroom Classification', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(__file__).parent / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature importance plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main function to run all tests"""
    print("\n" + "=" * 60)
    print("MUSHROOM MODEL TESTING")
    print("=" * 60 + "\n")
    
    # Load model
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        return
    
    # Prepare test data
    X_test, y_test = prepare_test_data()
    
    # Evaluate model
    metrics = evaluate_model(model, preprocessor, X_test, y_test)
    
    # Analyze feature importance
    feature_rankings = analyze_feature_importance(model, preprocessor)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60 + "\n")
    
    # Safety check
    if metrics['accuracy'] < 0.95:
        print("⚠️  WARNING: Model accuracy is below 95%!")
        print("   Consider retraining or adjusting model parameters.")
    else:
        print("✅ Model performance looks good!")
    
    if metrics['confusion_matrix'][1][0] > 0:
        print(f"\n⚠️  SAFETY WARNING: {metrics['confusion_matrix'][1][0]} poisonous mushrooms")
        print("   were incorrectly classified as edible!")
        print("   This model should NOT be used for real-world mushroom identification.")
    
    print("\n")

if __name__ == "__main__":
    main()
