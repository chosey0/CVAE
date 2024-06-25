import numpy as np
import torch
import pandas as pd

def collate(batch):
    src_batchs = []
    reals = []
    
    for src in batch:
        src_batchs.append(src[["log_vol_pct", "amount_pct"]].values.astype(float).T)
        reals.append(src)
        
    return torch.tensor(np.array(src_batchs), dtype=torch.float32).unsqueeze(-1), pd.DataFrame(reals)


def multi_collate(batch):
    batch_temp = []
    batch_reals = []
    for tensor, reals in batch:
        batch_temp.append(tensor)
        batch_reals.append(reals)
        
    batch_tensor = torch.cat(batch_temp, dim=0)
    return batch_tensor, batch_reals