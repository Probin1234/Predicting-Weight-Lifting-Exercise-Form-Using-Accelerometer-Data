#!/usr/bin/env python3
"""
Weight Lifting Exercises Dataset - Predicting Exercise Class
Equivalent analysis to the Coursera Practical Machine Learning project.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    train = pd.read_csv('/tmp/pml-training.csv', low_memory=False)
    test = pd.read_csv('/tmp/pml-testing.csv', low_memory=False)

    # Feature selection
    exclude = ['Unnamed: 0', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 
               'cvtd_timestamp', 'new_window', 'num_window', 'classe']
    feature_cols = [c for c in train.columns if c not in exclude and train[c].isna().mean() < 0.1]
    print(f"Selected {len(feature_cols)} features with low missingness")

    X = train[feature_cols]
    y = train['classe']
    X_test = test[feature_cols]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Final model and test predictions
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)
    test_preds = rf_final.predict(X_test)
    print("Test predictions:", list(test_preds))

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                                X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

if __name__ == "__main__":
    main()
