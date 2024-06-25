from torch.utils.data import Dataset, BatchSampler, Sampler, IterableDataset, DataLoader
import numpy as np

class eBestDataset(Dataset):
    def __init__(self, data, key):
        self.data = data
        self.key = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
    
    
class eBestMultiDataset(IterableDataset):
    def __init__(self, subsets):
        self.subsets = subsets
        self.length = sum([len(subset) for subset in self.subsets])
        self.terminated = [False] * len(subsets)
        
    def __iter__(self):
        while self.terminated.count(False) > 0:
            
            for i in range(len(self.subsets)):
                subset = self.subsets[i]
                try:
                    tensor, reals = next(subset._get_iterator())
                    yield tensor, reals
                    
                except:
                    self.terminated[i] = True
                    yield np.array([[0.], [0.]]), None