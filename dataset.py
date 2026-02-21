# dataset.py - DGCNN Modularization
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
import pandas as pd
import torch
import os
import shutil
import numpy as np 

# 1. Defining the feature as PSD of each band for each electrode. 

# We use three main bands according to the article 
sampling_rate =128
seconds_to_keep = 60
chunk_size = 256
overlap = 128     

num_samples_to_keep = seconds_to_keep * sampling_rate
Band_dict = {'alpha': [8, 13], 'beta': [13, 20], 'theta': [4, 8]}

Num_Features = len(Band_dict.keys())

# 2. Cropping the last 60 seconds of the signal for each trial.
def crop_last_60s(signal_dict):
    total_samples = signal_dict.shape[1]
    if total_samples > num_samples_to_keep:
        return signal_dict[:, -num_samples_to_keep:]
    return signal_dict
    
# 3. Apply PSD transformation 
psd_transform = transforms.Compose([
    transforms.BandPowerSpectralDensity(
        sampling_rate=sampling_rate,
        band_dict=Band_dict  
    ),
    transforms.MeanStdNormalize() 
])

dataset_path = r"D:\DREAMER.mat"
save_path = r'.\processed_data_DGCNN_paper'

# Loading the dataset & applying transformations

print(f"⏳ Loading dataset from {save_path}...")

dataset = DREAMERDataset(
    io_path=save_path, 
    mat_path=dataset_path, 
    chunk_size=chunk_size,
    overlap=overlap,       
    before_trial=crop_last_60s,
    offline_transform=psd_transform,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('arousal'),
        transforms.Binary(3.0)
    ]),
    num_worker=4, # To prevent memory crashes set 0 or 1 
    verbose=True
)

print(f"✅ Data Loaded! Total samples: {len(dataset)}")