import torch
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import numpy as np

def load_dataset(keys, store, resample=True):
    dataset = [] # name, subset
    
    for key in tqdm(keys, total=len(keys), desc="Dataset"):
        sub_set = store.get(key)
        
        subset = [] # 1s_item
        length = []
        
        sub_set["date"] = pd.to_datetime(sub_set["date"].astype(str) + " " + sub_set["time"], format="%Y%m%d %H%M%S")
        sub_set["amount"] = sub_set["close"] * sub_set["jdiff_vol"]
        #########
        sub_set["amount_pct"] = np.log(sub_set["amount"]).pct_change().fillna(0)
        sub_set["log_vol_pct"] = np.log((sub_set["jdiff_vol"] / sub_set["jdiff_vol"].shift(1))).fillna(0) * 0.1 
        #########
        
        if resample:
            [[subset.append(item), length.append(len(item))] for _, item in tqdm(sub_set.resample("1s", on="date"), total=len(sub_set.resample("1s", on="date")), desc="subset") if not item.empty]
            dataset.append((key, subset, length))
            
        else:
            dataset.append((key, sub_set, len(sub_set)))
        
    return dataset

def init_dataset(train, val, collate, batch_size):
    
    def get_sampler(data):
        num_data = np.shape(data)[0]
        sampler = SubsetRandomSampler(torch.from_numpy(np.arange(0, num_data)))

        return sampler
    
    cuda = torch.cuda.is_available()
    
    train_loader = DataLoader(train, batch_size=batch_size, pin_memory=cuda,
                                             collate_fn=collate, sampler=WeightedRandomSampler(torch.DoubleTensor(train.df["weights"].values), int(len(train))))
                                            #  sampler=get_sampler(train.keys))
    valid_loader = DataLoader(val, batch_size=batch_size, pin_memory=cuda,
                                             collate_fn=collate, sampler=WeightedRandomSampler(torch.DoubleTensor(val.df["weights"].values), int(len(val)))) #shuffle=False, drop_last=True)
                                            #  sampler=get_sampler(val.keys))
    
    return train_loader, valid_loader