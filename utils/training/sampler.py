from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, dataset, multi=True):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.is_multi = multi
        if multi:
            self.num_dataset = len(dataset)
            self.dataset_lengths = [len(subset) for subset in dataset]
            self.indices = [0] * self.num_datasets
        
    def __iter__(self):
        while True:
            batch = []
            for i in range(self.num_datasets):
                if self.current_indices[i] < self.dataset_lengths[i]:
                    batch.append(self.current_indices[i])
                    self.current_indices[i] += 1
                else:
                    batch.append([-0])
            if all(index == -1 for index in batch):
                break
            yield batch

    def __len__(self):
        if not self.is_multi:
            return len(self.dataset)
        else:
            return max(self.dataset_lengths)