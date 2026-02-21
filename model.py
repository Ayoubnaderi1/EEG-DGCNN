# model.py - DGCNN Modularization
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):
        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Intialization and requires_grad=True
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(device))
        # Smart weight Initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).to(device))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out
# output shape = (Batch, N, out_channels)

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        # weight matrix shape = (in_channels, out_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

# Output shape = (Batch, N, out_channels)

def normalize_A(A, lmax=2):
    
    dev = A.device # cuda or cpu
    
    # We must eliminate the negative values in the adj Matrix
    A = F.relu(A) 
    N = A.shape[0] # a square mitrix [N, N]
    # Making A symmetric and removing self-loops
    A = A * (torch.ones(N, N, device=dev) - torch.eye(N, N, device=dev))
    A = A + A.T
    # Normalization
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N, N, device=dev) - torch.matmul(torch.matmul(D, A), D)
    # all eigenvalues of L are in the range [0, 2], so we can scale by lmax=2
    # evals_L - evals_Lnorm = 1.0
    
    Lnorm = (2 * L / lmax) - torch.eye(N, N, device=dev)
    
    return Lnorm
# output shape = (N, N)

# Chebyshev polynomials up to order K-1 (K supports)

def generate_cheby_adj(Lnorm, K):
    support = []
    for i in range(K):
        if i == 0:
           
            support.append(torch.eye(Lnorm.shape[-1], device=Lnorm.device, dtype=Lnorm.dtype))
        elif i == 1:
            support.append(Lnorm)
        else:
           
            temp = torch.matmul(2*Lnorm, support[-1]) - support[-2]
            support.append(temp)
    return support # a list
    
# if K = 3, Support = [I, Lnorm , 2*(Lnorm^2) - I]
# len(support) = K , shape of each = [N, N]


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result
# output shape = (Batch, N, out_channels)

class DGCNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=2):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.num_electrodes = num_electrodes
        self.out_channels = out_channels
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes))
      
        nn.init.uniform_(self.A, 0.0, 0.1)
        
        
        #nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
   
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        #self.layer2 = Chebynet(out_channels, k_adj, out_channels * 2) 
        #self.bn2 = nn.BatchNorm1d(out_channels*2)
        
        # --- 1Dim Convolution ---
        # conv.shape = (32,32) # transform 32 features into 1 scalar 
        self.conv1x1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1) 
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')

        self.flatten_dim = num_electrodes * out_channels 
      
        self.fc1 = Linear(self.flatten_dim, 128) 
        self.fc2 = Linear(128, 64)               
        self.fc_out = Linear(64, num_classes)    
        nn.init.xavier_normal_(self.fc_out.linear.weight) # As we don't have activation after the last layer, we can use Xavier initialization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
     
    #data can also be standardized offline
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)
        
        L = normalize_A(self.A)
        x = self.layer1(x, L)  
       
         # --- 1x1 Convolution ---
        # Conv1d Expects input as shape: (Batch, Channels, Electrodes) 
        
        x = x.permute(0, 2, 1) # -> (Batch, out_channels, Electrodes)
        x = self.conv1x1(x)
        x = F.relu(x)          # [1] Relu activation after conv
        
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)     
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)    
        
        logits = self.fc_out(x)
        return logits