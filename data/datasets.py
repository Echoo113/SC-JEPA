import torch
from torch.utils.data import Dataset

# ============================================================================
# Dataset Classes
# ============================================================================
# Dataset implementations for time series data processing

class PastFutureDataset(Dataset):
    """Get time-series data, form (past, future) pairs. Shape: (T, num_patches, patch_len, in_channels)"""
    def __init__(self, data_tensor: torch.Tensor):
        self.data = data_tensor

    def __len__(self):
        # the last patch has no corresponding future
        return len(self.data) - 1

    def __getitem__(self, idx):
        # past = idx, future = idx+1
        return self.data[idx], self.data[idx + 1]

# ============================================================================
# Multi-Variable to Univariate Strategy
# ============================================================================
# Critical: Converts multi-variable time series to independent univariate series
# This ensures channel-independent training and prevents cross-variable contamination

class ChannelIndependentDataset(Dataset):
    """Safe version: pair within same variable (n -> n+1), avoid cross-variable frame mixing. Input: (N,P,L,V), Output: (P,L,1)"""
    def __init__(self, data_tensor: torch.Tensor):
        assert data_tensor.ndim == 4, f"Need (N,P,L,V), got {data_tensor.shape}"
        N, P, L, V = data_tensor.shape
        
        # Support both V=1 (combined single-variable format) and V>1 (multi-variable format)
        if V == 1:
            # Single variable format: directly create pairs from N dimension
            pairs = []
            for n in range(N - 1):
                past = data_tensor[n]  # (P, L, 1)
                future = data_tensor[n + 1]  # (P, L, 1)
                pairs.append((past, future))
            self.pairs = pairs  # length (N-1)
        else:
            # Multi-variable format: process each variable separately
            x = data_tensor.permute(3, 0, 1, 2).unsqueeze(-1).contiguous()  # (V,N,P,L,1)
            pairs = []
            for v in range(V):
                for n in range(N - 1):
                    past = x[v, n]
                    future = x[v, n + 1]
                    pairs.append((past, future))
            self.pairs = pairs  # length V*(N-1)
        
        self.P, self.L = P, L

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]