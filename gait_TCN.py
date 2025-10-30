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
from sklearn.utils.class_weight import compute_class_weight

# Dataset
class StrideDataset(Dataset):
    def __init__(self, strides, labels):
        self.strides = torch.tensor(strides, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.strides)

    def __getitem__(self, idx):
        return self.strides[idx], self.labels[idx]

# TCN Components
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TCN Model
class GaitTCN(nn.Module):
    def __init__(self, input_channels=39, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(GaitTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 2)  # 2 output classes: control/patient

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
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

# Load data
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/trials_all.pkl', 'rb') as f:
    trials_all = pickle.load(f)
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/trials_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Filter to LEFT strides
"""left_mask = metadata['side'] == 'L'
X = strides_all[left_mask]"""
X = trials_all
y = metadata['group'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'control' = 0, 'patient' = 1

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Dataloaders
train_ds = StrideDataset(X_train, y_train)
val_ds = StrideDataset(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GaitTCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

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

# Training loop with best model tracking
best_val_acc = 0.0
best_model_state = None
best_epoch = 0

for epoch in range(50):
    train_loss, train_acc = train_model(model, train_dl, optimizer, criterion, device)
    val_loss, val_acc = evaluate_model(model, val_dl, criterion, device)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2%}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2%}")
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1
        print(f"  → New best TCN model at epoch {best_epoch} with val acc: {best_val_acc:.2%}")

print(f"\nBest TCN model was at epoch {best_epoch} with validation accuracy: {best_val_acc:.2%}")

# Get final validation metrics
val_loss, val_acc = evaluate_model(model, val_dl, criterion, device)

# Save model and related components
save_dir = '/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering'
os.makedirs(save_dir, exist_ok=True)

# Save final model state dict
torch.save(model.state_dict(), os.path.join(save_dir, 'gait_tcn_final_model.pth'))

# Save best model state dict
torch.save(best_model_state, os.path.join(save_dir, 'gait_tcn_best_model.pth'))

# Save complete model (alternative approach)
torch.save(model, os.path.join(save_dir, 'gait_tcn_complete.pth'))

# Save validation data and labels (same as CNN version)
torch.save(X_val, os.path.join(save_dir, 'X_val.pth'))
torch.save(y_val, os.path.join(save_dir, 'y_val.pth'))

# Save label encoder for future use
with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# Save training metadata
training_info = {
    'model_architecture': 'GaitTCN',
    'input_channels': 39,
    'num_classes': 2,
    'tcn_channels': [64, 128, 256],
    'kernel_size': 3,
    'dropout': 0.2,
    'final_train_loss': train_loss,
    'final_train_acc': train_acc,
    'final_val_loss': val_loss,
    'final_val_acc': val_acc,
    'best_val_acc': best_val_acc,
    'best_epoch': best_epoch,
    'total_epochs': 50,
    'train_size': len(X_train),
    'val_size': len(X_val),
    'label_mapping': dict(zip(le.classes_, le.transform(le.classes_)))
}

with open(os.path.join(save_dir, 'training_info_tcn.pkl'), 'wb') as f:
    pickle.dump(training_info, f)

print(f"TCN model and validation data saved to {save_dir}")
