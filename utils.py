# utils.py - DGCNN Modularization
import os
import csv

# --- Helper Functions ---
def log_to_csv(results_dict, filepath):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        if not file_exists: 
            writer.writeheader()
        writer.writerow(results_dict)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']