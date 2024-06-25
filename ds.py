from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class DS(Dataset):
    def __init__(self, data, weights=None, resampled=True):
        self.data = data
        self.weight = weights
        
        if resampled:
            self.idxs = np.argsort(self.weight)[::-1]
        else:
            self.idxs = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.idxs is None:
            return self.data.iloc[idx]
        else: return self.data[self.idxs[idx]]
    
    
def collate(batch):
    src_batchs = []
    reals = []
    
    for src in batch:
        src_batchs.append(src[["log_vol_pct", "amount_pct"]].values.astype(float).T)
        reals.append(src)
        
    return torch.tensor(np.array(src_batchs), dtype=torch.float32).unsqueeze(-1), pd.DataFrame(reals)