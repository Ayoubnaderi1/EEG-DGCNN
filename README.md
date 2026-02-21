🧠 EEG Emotion Recognition using DGCNN This project implements a Dynamical Graph Convolutional Neural Network (DGCNN) to classify emotional states from EEG signals.



* Project Overview 



The model utilizes dynamic graphs to represent the functional connectivity between brain electrodes. Unlike static graphs, the Adjacency Matrix in this implementation is a learnable parameter, allowing the network to discover the most relevant neural connections during training.



* Project details



Model: DGCNN with Chebyshev polynomial-based graph convolution.



Feature Extraction: Power Spectral Density computed by torcheeg PyTorch, utilizing a learnable adjacency matrix.



Dynamic Adjacency Matrix: Implements a custom EMA-based update for the graph structure (rho=0.001) to ensure stable topology learning.



Dataset: DREAMER (14 electrodes, sampled at 128Hz).



Evaluation: Subject-dependent training with 18-fold cross-validation.



* Project Structure



dataset.py: Handles DREAMER dataset loading and PSD feature extraction.



model.py: Contains the DGCNN and Chebynet architecture.



utils.py: Helper functions for logging and learning rate tracking.



main.py: The main execution script for the 5-fold training loop.





In this project I am trying to implement DGCNN. For more details, please refer to the following information.



Paper: Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks\[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532-541.



URL: https://ieeexplore.ieee.org/abstract/document/8320798



Related Project: https://github.com/xueyunlong12589/DGCNN

