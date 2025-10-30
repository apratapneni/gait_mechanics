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
import warnings
warnings.filterwarnings('ignore')

# Dataset for full trials
class TrialDataset(Dataset):
    def __init__(self, trials, labels):
        self.trials = torch.tensor(trials, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx], self.labels[idx]

# TCN Components (adapted for longer sequences)
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

# Enhanced TCN Model for longer sequences
class TrialTCN(nn.Module):
    def __init__(self, input_channels=39, num_channels=[64, 128, 256, 512], kernel_size=7, dropout=0.3):
        super(TrialTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Use larger dilations for longer sequences
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
        # Multi-scale pooling for better temporal aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention mechanism for longer sequences
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 4),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 4, 1),
            nn.Sigmoid()
        )
        
        # Final classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),  # *2 for avg+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 2 output classes: control/patient
        )

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Apply TCN layers
        x = self.network(x)
        
        # Multi-scale pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (B, C)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (B, C)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2*C)
        
        # Final classification
        output = self.classifier(pooled)
        
        return output

# Enhanced training routine with gradient clipping
def train_model(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss, total_correct = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability with long sequences
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
    
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

# Enhanced evaluation with more metrics
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

    return val_loss, val_acc, acc, auc, cm, report

def main():
    # Load trial data
    print("Loading trial data...")
    with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/trials_all.pkl', 'rb') as f:
        trials_all = pickle.load(f)
    with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/trials_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"Data shape: {trials_all.shape}")
    print(f"Metadata shape: {metadata.shape}")

    # Check data balance
    y = metadata['group'].values
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # 'control' = 0, 'patient' = 1
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Train/Val Split with stratification to handle imbalance
    X_train, X_val, y_train, y_val = train_test_split(
        trials_all, y_encoded, 
        test_size=0.2, 
        stratify=y_encoded, 
        random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Validation class distribution: {np.bincount(y_val)}")

    # Dataloaders with smaller batch size for memory efficiency
    batch_size = 16  # Smaller batch size for longer sequences
    train_ds = TrialDataset(X_train, y_train)
    val_ds = TrialDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  # Set to 0 to avoid multiprocessing issues
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)  # Set to 0 to avoid multiprocessing issues

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TrialTCN(
        input_channels=39,
        num_channels=[64, 128, 256, 512],  # Deeper network for longer sequences
        kernel_size=7,  # Larger kernel for longer sequences
        dropout=0.3  # More dropout for regularization
    ).to(device)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Optimizer with weight decay and learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Handle class imbalance with weighted loss
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Class weights: {class_weights}")

    # Training loop with enhanced tracking
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    max_patience = 15

    print("\nStarting training...")
    for epoch in range(100):  # More epochs for convergence
        train_loss, train_acc = train_model(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc, val_acc_sklearn, val_auc, cm, report = evaluate_model(model, val_dl, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2%}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2%}, Val AUC = {val_auc:.3f}")
        
        # Save best model based on validation AUC (better for imbalanced data)
        if val_auc > best_val_auc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  → New best model at epoch {best_epoch} with val AUC: {best_val_auc:.3f}, val acc: {best_val_acc:.2%}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {max_patience} epochs)")
            break

    print(f"\nBest model was at epoch {best_epoch} with validation AUC: {best_val_auc:.3f}, accuracy: {best_val_acc:.2%}")

    # Load best model and get final metrics
    model.load_state_dict(best_model_state)
    val_loss, val_acc, val_acc_sklearn, val_auc, cm, report = evaluate_model(model, val_dl, criterion, device)

    # Save model and related components
    save_dir = '/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering'
    os.makedirs(save_dir, exist_ok=True)

    # Save best model state dict
    torch.save(best_model_state, os.path.join(save_dir, 'trial_tcn_best_model.pth'))

    # Save complete model
    torch.save(model, os.path.join(save_dir, 'trial_tcn_complete.pth'))

    # Save validation data and labels
    torch.save(X_val, os.path.join(save_dir, 'X_val_trials.pth'))
    torch.save(y_val, os.path.join(save_dir, 'y_val_trials.pth'))

    # Save label encoder
    with open(os.path.join(save_dir, 'label_encoder_trials.pkl'), 'wb') as f:
        pickle.dump(le, f)

    # Save comprehensive training metadata
    training_info = {
        'model_architecture': 'TrialTCN',
        'input_channels': 39,
        'num_classes': 2,
        'tcn_channels': [64, 128, 256, 512],
        'kernel_size': 7,
        'dropout': 0.3,
        'batch_size': batch_size,
        'sequence_length': 2700,
        'total_params': total_params,
        'best_val_acc': best_val_acc,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'class_distribution': dict(zip(unique, counts)),
        'class_weights': class_weights.cpu().numpy().tolist(),
        'label_mapping': dict(zip(le.classes_, le.transform(le.classes_))),
        'final_confusion_matrix': cm.tolist(),
        'final_classification_report': report
    }

    with open(os.path.join(save_dir, 'training_info_trial_tcn.pkl'), 'wb') as f:
        pickle.dump(training_info, f)

    print(f"\nTrial TCN model and validation data saved to {save_dir}")
    print(f"Files saved:")
    print(f"  - trial_tcn_best_model.pth")
    print(f"  - trial_tcn_complete.pth")
    print(f"  - X_val_trials.pth")
    print(f"  - y_val_trials.pth")
    print(f"  - label_encoder_trials.pkl")
    print(f"  - training_info_trial_tcn.pkl")

if __name__ == '__main__':
    main()
