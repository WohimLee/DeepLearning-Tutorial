
import numpy as np
   
class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cursor = 0
        self.indexs = list(range(self.dataloader.count_data))  # 0, ... 60000
        if self.dataloader.shuffle:
            # 打乱一下
            np.random.shuffle(self.indexs)
            
    def __next__(self):
        if self.cursor >= self.dataloader.count_data:
            raise StopIteration()
            
        batch_data = []
        remain = min(self.dataloader.batch_size, self.dataloader.count_data - self.cursor)  #  256, 128
        for n in range(remain):
            index = self.indexs[self.cursor]
            data = self.dataloader.dataset[index]
            
            # 如果batch没有初始化，则初始化n个list成员
            if len(batch_data) == 0:
                batch_data = [[] for i in range(len(data))]
                
            #直接append进去
            for index, item in enumerate(data):
                batch_data[index].append(item)
            self.cursor += 1
            
        # 通过np.vstack一次性实现合并，而非每次一直在合并
        for index in range(len(batch_data)):
            batch_data[index] = np.vstack(batch_data[index])
        return batch_data

class DataLoader:
    
    # shuffle 打乱
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.count_data = len(dataset)
        self.batch_size = batch_size
        
    def __iter__(self):
        return DataLoaderIterator(self)


if __name__ == '__main__':
    pass
