import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Load data
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/strides_all.pkl', 'rb') as f:
    strides_all = pickle.load(f)
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Filter to LEFT strides
left_mask = metadata['side'] == 'L'
X = strides_all[left_mask]
left_metadata = metadata[left_mask].copy()

# Patient-based splitting (same as before)
unique_patients = left_metadata[['patient_id', 'group']].drop_duplicates()
control_patients = unique_patients[unique_patients['group'] == 'control']['patient_id'].values
patient_patients = unique_patients[unique_patients['group'] == 'patient']['patient_id'].values

control_train, control_val = train_test_split(control_patients, test_size=0.2, random_state=42)
patient_train, patient_val = train_test_split(patient_patients, test_size=0.2, random_state=42)

train_patients = np.concatenate([control_train, patient_train])
val_patients = np.concatenate([control_val, patient_val])

train_mask = left_metadata['patient_id'].isin(train_patients)
val_mask = left_metadata['patient_id'].isin(val_patients)

X_train = X[train_mask]
X_val = X[val_mask]
y_train = left_metadata[train_mask]['group'].values
y_val = left_metadata[val_mask]['group'].values

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

print(f"Training: {len(X_train)} strides from {len(train_patients)} patients")
print(f"Validation: {len(X_val)} strides from {len(val_patients)} patients")

# Feature Engineering Functions
def extract_statistical_features(X):
    """Extract statistical features from time series data"""
    features = []
    
    # For each channel, extract statistical features
    for channel in range(X.shape[2]):
        channel_data = X[:, :, channel]  # Shape: (n_samples, time_points)
        
        # Basic statistics
        features.append(np.mean(channel_data, axis=1))  # Mean
        features.append(np.std(channel_data, axis=1))   # Std
        features.append(np.min(channel_data, axis=1))   # Min
        features.append(np.max(channel_data, axis=1))   # Max
        features.append(np.median(channel_data, axis=1)) # Median
        
        # Percentiles
        features.append(np.percentile(channel_data, 25, axis=1))  # 25th percentile
        features.append(np.percentile(channel_data, 75, axis=1))  # 75th percentile
        
        # Range and IQR
        features.append(np.max(channel_data, axis=1) - np.min(channel_data, axis=1))  # Range
        features.append(np.percentile(channel_data, 75, axis=1) - np.percentile(channel_data, 25, axis=1))  # IQR
        
        # Peak-to-peak and RMS
        features.append(np.sqrt(np.mean(channel_data**2, axis=1)))  # RMS
        
    return np.column_stack(features)

def extract_key_channel_features(X, key_channels=[8, 11, 35]):
    """Extract features from the key channels identified by Random Forest"""
    features = []
    
    for channel in key_channels:
        channel_data = X[:, :, channel]
        
        # Full time series for this channel
        features.append(channel_data)
        
        # Statistical features for this channel
        features.append(np.mean(channel_data, axis=1).reshape(-1, 1))
        features.append(np.std(channel_data, axis=1).reshape(-1, 1))
        features.append(np.min(channel_data, axis=1).reshape(-1, 1))
        features.append(np.max(channel_data, axis=1).reshape(-1, 1))
        
        # Key time points identified by RF
        key_timepoints = [26, 110, 125, 126, 139, 156, 163, 168, 169]
        for tp in key_timepoints:
            if tp < channel_data.shape[1]:
                features.append(channel_data[:, tp].reshape(-1, 1))
    
    return np.column_stack([f.reshape(f.shape[0], -1) for f in features])

def extract_temporal_features(X):
    """Extract temporal features like derivatives and patterns"""
    features = []
    
    # First and second derivatives (velocity and acceleration)
    first_deriv = np.diff(X, axis=1)  # Shape: (n_samples, time_points-1, channels)
    second_deriv = np.diff(first_deriv, axis=1)  # Shape: (n_samples, time_points-2, channels)
    
    # Statistical features of derivatives
    features.append(np.mean(first_deriv, axis=1))  # Mean velocity
    features.append(np.std(first_deriv, axis=1))   # Std velocity
    features.append(np.mean(second_deriv, axis=1)) # Mean acceleration
    features.append(np.std(second_deriv, axis=1))  # Std acceleration
    
    # Zero crossings
    zero_crossings = []
    for i in range(X.shape[2]):
        channel_data = X[:, :, i]
        zc = np.sum(np.diff(np.sign(channel_data), axis=1) != 0, axis=1)
        zero_crossings.append(zc)
    features.append(np.column_stack(zero_crossings))
    
    return np.column_stack(features)

# Extract different types of features
print("Extracting features...")
X_train_stats = extract_statistical_features(X_train)
X_val_stats = extract_statistical_features(X_val)

X_train_key = extract_key_channel_features(X_train)
X_val_key = extract_key_channel_features(X_val)

X_train_temporal = extract_temporal_features(X_train)
X_val_temporal = extract_temporal_features(X_val)

# Combine all features
X_train_combined = np.column_stack([X_train_stats, X_train_key, X_train_temporal])
X_val_combined = np.column_stack([X_val_stats, X_val_key, X_val_temporal])

# Standardize features
scaler = StandardScaler()
X_train_combined_scaled = scaler.fit_transform(X_train_combined)
X_val_combined_scaled = scaler.transform(X_val_combined)

print(f"Statistical features shape: {X_train_stats.shape}")
print(f"Key channel features shape: {X_train_key.shape}")
print(f"Temporal features shape: {X_train_temporal.shape}")
print(f"Combined features shape: {X_train_combined.shape}")

# Compare different approaches
models = {
    'Random Forest (Raw Flattened)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Random Forest (Statistical)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Random Forest (Key Channels)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Random Forest (Combined)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Logistic Regression (Combined)': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (Combined)': SVC(random_state=42, probability=True)
}

feature_sets = {
    'Random Forest (Raw Flattened)': (X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1)),
    'Random Forest (Statistical)': (X_train_stats, X_val_stats),
    'Random Forest (Key Channels)': (X_train_key, X_val_key),
    'Random Forest (Combined)': (X_train_combined_scaled, X_val_combined_scaled),
    'Logistic Regression (Combined)': (X_train_combined_scaled, X_val_combined_scaled),
    'SVM (Combined)': (X_train_combined_scaled, X_val_combined_scaled)
}

results = {}

print("\n=== MODEL COMPARISON ===")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    X_tr, X_va = feature_sets[model_name]
    
    # Train model
    model.fit(X_tr, y_train_encoded)
    
    # Predict
    y_pred = model.predict(X_va)
    y_prob = model.predict_proba(X_va)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    acc = accuracy_score(y_val_encoded, y_pred)
    auc = roc_auc_score(y_val_encoded, y_prob) if y_prob is not None else None
    
    results[model_name] = {
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    print(f"  Accuracy: {acc:.2%}")
    if auc is not None:
        print(f"  AUC: {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val_encoded, y_pred)
    print(f"  Confusion Matrix:\n{cm}")

# Print summary
print("\n=== SUMMARY ===")
for model_name, result in results.items():
    auc_str = f", AUC: {result['auc']:.3f}" if result['auc'] is not None else ""
    print(f"{model_name}: {result['accuracy']:.2%}{auc_str}")

# Feature importance for the best Random Forest model
best_rf_name = max([name for name in results.keys() if 'Random Forest' in name], 
                   key=lambda x: results[x]['accuracy'])
best_rf_model = models[best_rf_name]

print(f"\n=== FEATURE IMPORTANCE ({best_rf_name}) ===")
if hasattr(best_rf_model, 'feature_importances_'):
    importances = best_rf_model.feature_importances_
    top_indices = np.argsort(importances)[-20:][::-1]
    print("Top 20 most important features:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
