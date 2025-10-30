import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Dataset
class StrideDataset(Dataset):
    def __init__(self, strides, labels):
        self.strides = torch.tensor(strides, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.strides)

    def __getitem__(self, idx):
        return self.strides[idx], self.labels[idx]

# CNN with regularization
class GaitCNN(nn.Module):
    def __init__(self, input_channels=39, dropout_rate=0.5):
        super(GaitCNN, self).__init__()
        # Simpler architecture with fewer parameters
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Add batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simpler classifier with dropout
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 output classes: control/patient

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
        
        # Conv layers with batch norm and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Global pooling and classifier
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training Routine
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

# scaler function
def stride_scaler(stride):
    """
    Normalize the entire 300x39 stride to unit variance and zero mean,
    but treat the entire set of joints as a single vector.
    """
    mean = np.mean(stride)
    std = np.std(stride) + 1e-8
    return (stride - mean) / std


# Load data
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/strides_all.pkl', 'rb') as f:
    strides_all = pickle.load(f)
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Filter to LEFT strides
left_mask = metadata['side'] == 'L'
X = strides_all[left_mask]
left_metadata = metadata[left_mask].copy()

# Get unique patients per group for patient-based splitting
unique_patients = left_metadata[['patient_id', 'group']].drop_duplicates()
control_patients = unique_patients[unique_patients['group'] == 'control']['patient_id'].values
patient_patients = unique_patients[unique_patients['group'] == 'patient']['patient_id'].values

print(f"Total patients: {len(unique_patients)}")
print(f"Control patients: {len(control_patients)}")
print(f"Patient patients: {len(patient_patients)}")

# Split patients (not strides) into train/val while maintaining class balance
from sklearn.model_selection import train_test_split
control_train, control_val = train_test_split(control_patients, test_size=0.2, random_state=13)
patient_train, patient_val = train_test_split(patient_patients, test_size=0.2, random_state=13)

# Combine train and val patient lists
train_patients = np.concatenate([control_train, patient_train])
val_patients = np.concatenate([control_val, patient_val])

print(f"Training patients: {len(train_patients)} ({len(control_train)} control + {len(patient_train)} patient)")
print(f"Validation patients: {len(val_patients)} ({len(control_val)} control + {len(patient_val)} patient)")

# Create train/val masks based on patient IDs
train_mask = left_metadata['patient_id'].isin(train_patients)
val_mask = left_metadata['patient_id'].isin(val_patients)

# Split the data and apply stride scaling
X_train = X[train_mask]
X_val = X[val_mask]
X_train = np.array([stride_scaler(stride) for stride in X_train])
X_val = np.array([stride_scaler(stride) for stride in X_val])

y_train = left_metadata[train_mask]['group'].values
y_val = left_metadata[val_mask]['group'].values

print(f"Training strides: {len(X_train)}")
print(f"Validation strides: {len(X_val)}")

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # 'control' = 0, 'patient' = 1
y_val_encoded = le.transform(y_val)

# Let's add some debugging information
print("\n=== DEBUGGING INFORMATION ===")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Train labels distribution: {np.bincount(y_train_encoded)}")
print(f"Val labels distribution: {np.bincount(y_val_encoded)}")
print(f"Train patients: {sorted(train_patients)}")
print(f"Val patients: {sorted(val_patients)}")

# Check if there's any overlap (there shouldn't be!)
overlap = set(train_patients) & set(val_patients)
if overlap:
    print(f"WARNING: Patient overlap detected: {overlap}")
else:
    print("✓ No patient overlap between train and validation sets")

print("=== STARTING TRAINING ===\n")

# Dataloaders
train_ds = StrideDataset(X_train, y_train_encoded)
val_ds = StrideDataset(X_val, y_val_encoded)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

# Setup with regularization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaitCNN(dropout_rate=0.3).to(device)  # Less aggressive dropout
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Lower LR + weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

# Evaluate on validation set
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_correct += (logits.argmax(1) == y_batch).sum().item()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # Probability of being "patient"

    # Evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Control", "Patient"])

    print(f"\nValidation Accuracy: {acc:.2%}")
    print(f"AUC: {auc:.3f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    val_loss = total_loss / len(dataloader.dataset)
    val_acc = total_correct / len(dataloader.dataset)

    return val_loss, val_acc

# Training loop with early stopping and regularization
best_val_acc = 0.0
best_model_state = None
best_epoch = 0
patience = 10  # Early stopping patience
epochs_without_improvement = 0

# Let's also try a simple baseline for comparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n=== SIMPLE BASELINE COMPARISON ===")
# Flatten the stride data for traditional ML
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Train a simple Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=13)
rf.fit(X_train_flat, y_train_encoded)

# Predict on validation set
y_pred_rf = rf.predict(X_val_flat)
rf_acc = accuracy_score(y_val_encoded, y_pred_rf)
print(f"Random Forest baseline accuracy: {rf_acc:.2%}")

# Feature importance (top 10 features)
feature_importance = rf.feature_importances_
top_features = np.argsort(feature_importance)[-10:][::-1]
print(f"Top 10 most important features: {top_features}")
print("=== END BASELINE ===\n")

print("Starting training with patient-based split...")
print(f"Using device: {device}")

for epoch in range(100):  # Increased max epochs but with early stopping
    train_loss, train_acc = train_model(model, train_dl, optimizer, criterion, device)
    val_loss, val_acc = evaluate_model(model, val_dl, criterion, device)
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2%}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2%}, LR = {current_lr:.6f}")
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        print(f"  → New best model at epoch {best_epoch} with val acc: {best_val_acc:.2%}")
    else:
        epochs_without_improvement += 1
    
    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break
    
    # Stop if learning rate becomes too small
    if current_lr < 1e-7:
        print(f"Learning rate too small ({current_lr:.2e}), stopping training")
        break

print(f"\nBest model was at epoch {best_epoch} with validation accuracy: {best_val_acc:.2%}")

# Get final validation metrics
val_loss, val_acc = evaluate_model(model, val_dl, criterion, device)
print(f"Final Validation Loss = {val_loss:.4f}, Validation Acc = {val_acc:.2%}")

# Save model and related components
save_dir = '/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering'
os.makedirs(save_dir, exist_ok=True)

# Save final model state dict
torch.save(model.state_dict(), os.path.join(save_dir, 'gait_cnn_final_model.pth'))

# Save best model state dict
torch.save(best_model_state, os.path.join(save_dir, 'gait_cnn_best_model.pth'))

# Save complete model (alternative approach)
torch.save(model, os.path.join(save_dir, 'gait_cnn_complete.pth'))

# Save validation data and labels
torch.save(X_val, os.path.join(save_dir, 'X_val.pth'))
torch.save(y_val_encoded, os.path.join(save_dir, 'y_val.pth'))

# Save label encoder for future use
with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# Save training metadata
training_info = {
    'model_architecture': 'GaitCNN',
    'input_channels': 39,
    'num_classes': 2,
    'final_train_loss': train_loss,
    'final_train_acc': train_acc,
    'final_val_loss': val_loss,
    'final_val_acc': val_acc,
    'best_val_acc': best_val_acc,
    'best_epoch': best_epoch,
    'total_epochs': 50,
    'train_size': len(X_train),
    'val_size': len(X_val),
    'total_patients': len(unique_patients),
    'train_patients': len(train_patients),
    'val_patients': len(val_patients),
    'train_control_patients': len(control_train),
    'train_patient_patients': len(patient_train),
    'val_control_patients': len(control_val),
    'val_patient_patients': len(patient_val),
    'train_patient_list': train_patients.tolist(),
    'val_patient_list': val_patients.tolist(),
    'split_method': 'patient_based',  # Important: documents that we used patient-based splitting
    'label_mapping': dict(zip(le.classes_, le.transform(le.classes_)))
}

with open(os.path.join(save_dir, 'training_info.pkl'), 'wb') as f:
    pickle.dump(training_info, f)

print(f"CNN model and validation data saved to {save_dir}")
