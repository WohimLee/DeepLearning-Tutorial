
import numpy as np

class DataLoader:
    # shuffle 打乱
    def __init__(self, dataset, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.count_data = len(dataset)
        
    def __iter__(self):
        return DataLoaderIterator(self)

class DataLoaderIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.dataset    = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.cursor = 0
        self.indexs = list(range(len(self.dataset)))  # 0, ... 60000
        if self.dataloader.shuffle:
            # 打乱一下
            np.random.shuffle(self.indexs)
            
    def __next__(self):
        if self.cursor >= (n := len(self.dataset)):
            raise StopIteration()
            
        batch_size = min(self.batch_size, n - self.cursor)  #  256, 128
        batch_labels = self.dataset.labels[self.cursor: self.cursor+batch_size]
        batch_images = self.dataset.images[self.cursor: self.cursor+batch_size]
        
        self.cursor += batch_size
        return batch_labels, batch_images




if __name__ == '__main__':
    pass
