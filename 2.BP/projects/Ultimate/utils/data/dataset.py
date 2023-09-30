
import yaml
import numpy as np

class Dataset:
    def __init__(self, config='config/MNIST.yaml'):
        self.config = config
        self.initialize()

    def initialize(self):
        with open(self.config, 'r') as f:
            data = yaml.safe_load(f)
        print()
        pass
    
    
    # 获取他的一个item，  dataset = Dataset(),   dataset[index]
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)
     
     
if __name__ == '__main__':
    pass
