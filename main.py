# main.py - DGCNN Modularization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
import csv
import time
from datetime import datetime

# --- ایمپورت فایل‌های شما ---
from utils import log_to_csv, get_lr
from dataset import dataset
from model import DGCNN

# --- Configuration ---
RESULT_DIR = r'D:\Project\DGCNN\RESULTS'
CSV_FILE = os.path.join(RESULT_DIR, 'training_results_dreamer_losoOptW_5fold_lr001.csv')
LOG_DIR = f'runs/dreamer_experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

BATCH_SIZE = 128
EPOCHS = 20       
LEARNING_RATE = 0.001 
WEIGHT_DECAY = 1e-4   
LAMBDA_L1 = 1e-4      # Regularization weight (Alpha in Eq. 14) 
N_SPLITS = 18         # LeaveOneSubjectOut protocol for DREAMER 

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device} | TensorBoard Path: {LOG_DIR}")

# --- Metadata Initialization ---
df = dataset.info 
unique_subjects = sorted(df['subject_id'].unique())
writer = SummaryWriter(LOG_DIR)

# 

# ==========================================
# Subject-Dependent Training Loop
# ==========================================
for sub_id in unique_subjects:
    print(f"\n========== Processing Subject: {sub_id} ==========")
    all_trials = np.arange(18)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold_idx, (train_trials, test_trials) in enumerate(kf.split(all_trials)):
        start_time = time.time()
        
        # 1. Dataset Splitting
        train_idx = df[(df['subject_id'] == sub_id) & (df['trial_id'].isin(train_trials))].index.tolist()
        test_idx = df[(df['subject_id'] == sub_id) & (df['trial_id'].isin(test_trials))].index.tolist()
        
        # 2. Imbalance Handling (WeightedRandomSampler)
        y_train = (df.iloc[train_idx]['valence'] > 3.0).astype(int).values
        class_counts = np.bincount(y_train)
        w0 = 1. / class_counts[0] if class_counts[0] > 0 else 0
        w1 = 1. / class_counts[1] if (len(class_counts) > 1 and class_counts[1] > 0) else 0
        samples_weights = torch.tensor([w1 if t == 1 else w0 for t in y_train], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        
        # 3. DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
        
        model = DGCNN(in_channels=3, num_electrodes=14, k_adj=3, out_channels=32, num_classes=2).to(device)
        rho = 1e-3
        base_params = [param for name, param in model.named_parameters() if name != 'A']
        optimizer = optim.Adam(base_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # 5. Training Phase
        for epoch in range(EPOCHS):
            model.train()
            running_loss, running_ce, running_l1 = 0.0, 0.0, 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                model.zero_grad() # instead of optimizer.zero_grad() to avoid zeroing A's gradients
                output = model(batch_x)
                
                # Loss Calculation (Equation 14) 
                ce_loss = criterion(output, batch_y)
                l1_loss = torch.norm(model.A, p=1)
                total_loss = ce_loss + (LAMBDA_L1 * l1_loss)
                
                total_loss.backward()
                optimizer.step()
                
                # Non-negative adjacency matrix constraint 
                with torch.no_grad():
                    if model.A.grad is not None:
                        # implementing Exponential Moving Average (EMA) update for 'A'
                       model.A.data = (1 - rho) * model.A.data - rho * model.A.grad
                    
                    model.A.data = torch.relu(model.A.data)
                
                running_loss += total_loss.item()
                running_ce += ce_loss.item()
                running_l1 += l1_loss.item()

            # Logging metrics
            avg_loss = running_loss / len(train_loader)
            avg_ce = running_ce / len(train_loader)
            avg_l1 = running_l1 / len(train_loader)
            current_lr = get_lr(optimizer)

            writer.add_scalars(f'Loss/Sub_{sub_id}_Fold_{fold_idx+1}', {'Total': avg_loss, 'CE': avg_ce, 'L1': avg_l1}, epoch)
            writer.add_scalar(f'LR/Sub_{sub_id}', current_lr, epoch)
            
            scheduler.step()
            
        # 6. Evaluation Phase
        model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        fold_acc = 100 * correct / total
        fold_f1 = f1_score(all_targets, all_preds, average='weighted') * 100
        fold_accuracies.append(fold_acc)
        
        elapsed = time.time() - start_time
        print(f"  > Fold {fold_idx+1}/{N_SPLITS} | Acc: {fold_acc:.2f}% | F1: {fold_f1:.2f}% | Time: {elapsed:.1f}s")
        
        # 7. Final CSV Logging per Fold
        log_to_csv({
            'subject_id': sub_id, 
            'fold': fold_idx + 1, 
            'accuracy': round(fold_acc, 2),
            'f1_score': round(fold_f1, 2), 
            'avg_ce_loss': round(avg_ce, 4),
            'avg_l1_loss': round(avg_l1, 4),
            'final_lr': current_lr,
            'epochs': EPOCHS
        }, CSV_FILE)

    # Subject Summary
    avg_sub_acc = np.mean(fold_accuracies)
    print(f"⭐ Subject {sub_id} Mean Accuracy: {avg_sub_acc:.2f}%")
    log_to_csv({'subject_id': sub_id, 'fold': 'AVERAGE', 'accuracy': avg_sub_acc}, CSV_FILE)

writer.close()