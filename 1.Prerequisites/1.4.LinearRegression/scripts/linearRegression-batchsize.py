
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, file):
        self.file = file
        self.initialize()
        
    def initialize(self):
        meta = pd.read_csv(self.file).to_numpy()
        self.labels = meta[:, 0]
        self.features = meta[:, 1:]
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)


class DataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(np.arange(len(dataset)))
        self.cursor  = 0
        if shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cursor >= len(self):
            raise StopIteration
        
        batch_features = []
        batch_targets  = []
        remain = min(self.batch_size, len(self.dataset)-self.cursor)
        for n in range(remain):
            index = self.indices[self.cursor]
            features, targets = self.dataset[index]
            batch_features.append(features)
            batch_targets.append(targets)
            
            self.cursor += 1
        features = np.stack(batch_features)
        targets  = np.stack(batch_targets)

        return features, targets


if __name__ == '__main__':
    file = '../datasets/上海二手房价.csv'
    dataset = Dataset(file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(dataset.labels[0])
    