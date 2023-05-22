import random
import numpy as np
from torch.utils.data import BatchSampler
'''
Eg:
    test_data = Sep2SepDataset(data_file, config.input_l, config.output_l,tokenizer=tokenizer,test=True)
    test_chunked_batch_sampler = ChunkedBatchSampler(test_data, config.valid_batch, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_data,  num_workers=1, batch_sampler=test_chunked_batch_sampler)
'''
class ChunkedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if isinstance(self.dataset[0],tuple):
            source_lengths = [len(self.dataset[idx][0].split()) for idx in indices]
        else:
            source_lengths = [len(self.dataset[idx].split()) for idx in indices]

        sorted_indices = np.argsort(source_lengths)
        chunked_indices = []

        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size]
            if self.shuffle:
                random.shuffle(batch)
            chunked_indices.append(batch)

        if self.drop_last and len(chunked_indices[-1]) < self.batch_size:
            chunked_indices.pop()

        if self.shuffle:
            random.shuffle(chunked_indices)

        return iter(chunked_indices)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size