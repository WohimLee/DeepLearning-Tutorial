
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, file):
        self.file = file
        self.initialize()
        
    def initialize(self):
        # meta = pd.read_csv(self.file).to_numpy()
        meta = pd.read_csv(self.file)
        self.targets_mean = meta.mean()[0]
        self.targets_std  = meta.std()[0]
        meta = ((meta - meta.mean()) / meta.std()).to_numpy()
        self.targets = meta[:, 0]
        self.features = meta[:, 1:]
        
    def __getitem__(self, index):
        return self.features[index], self.targets[index]
        
    def __len__(self):
        return len(self.targets)


class DataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return DataLoaderIterator(self)
    
    
    
    
class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataset = dataloader.dataset
        self.shuffle = dataloader.shuffle
        self.batch_size = dataloader.batch_size
        self.cursor  = 0
        self.indices = list(np.arange(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration()
    
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
        targets  = np.stack(batch_targets).reshape(-1, 1)

        return features, targets
    


class Model:
    def __init__(self, num_features=6):
        self.num_features = num_features
        self.weights = np.random.randn(1, num_features)
        self.bias = 0
        
    def forward(self, x):
        self.x = x
        return x@self.weights.T + self.bias
    
    
    def backward(self, G, lr=1e-2):
        d_b = np.sum(G)
        d_W = self.x.T @ G
        self.weights = self.weights - lr*d_W.T
        self.bias = self.bias - lr*d_b
    
    def __call__(self, x):
        return self.forward(x)
    



if __name__ == '__main__':
    file = '/Users/azen/Desktop/myAir/Work/Workspace/Others/DeepLearning-Tutorial/1.Prerequisites/1.4.LinearRegression/datasets/上海二手房价.csv'
    num_features = 6
    batch_size = 16
    dataset = Dataset(file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    epochs = 2000
    lr_schedule = {
        0: 1e-1,
        100: 1e-2, 
        300: 1e-3, 
        500: 1e-4
    }
    model = Model(num_features=num_features)
    for epoch in range(epochs):
        if epoch in lr_schedule:
            lr = lr_schedule[epoch]
        for i, (features, targets) in enumerate(dataloader):
            predict = model(features)
            loss = 0.5*np.sum((predict - targets)**2) /batch_size
            G = (predict - targets) / batch_size
            model.backward(G, lr=lr)
        if (epoch+1) % 200 == 0:
            print("Epoch: {}, lr: {}, Loss: {}".format(epoch+1, lr, loss))
        
    # Test
    idx = np.random.randint(len(dataset))
    feature, target = dataset[idx]
    predict = model(feature)
    mean = dataset.targets_mean
    std = dataset.targets_std
    print("Predict: {:.1f}, Ground True: {}".format(float(predict*std + mean), target*std + mean))
    
    

    
    