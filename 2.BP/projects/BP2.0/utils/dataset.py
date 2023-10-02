
import yaml
import numpy as np
import os.path as osp

from .utils import load_labels, load_images, one_hot
    
class Dataset:
    def __init__(self, config='config/MNIST.yaml', train=True):
        self.config = config
        self.train = train
        self.initialize()

    def initialize(self):
        with open(self.config, 'r') as f:
            data = yaml.safe_load(f)
        if self.train:
            labels_path = osp.join(data['root'], data['train']['labels'])
            images_path = osp.join(data['root'], data['train']['images'])

        else:
            labels_path = osp.join(data['root'], data['test']['labels'])
            images_path = osp.join(data['root'], data['test']['images'])
            
        self.labels = one_hot(load_labels(labels_path))
        self.images = load_images(images_path)
        self.images = (self.images - np.mean(self.images)) / np.var(self.images)
        # self.images = self.images / 255 - 0.5
    
    def __getitem__(self, index):
        return self.labels[index], self.images[index] 
    
    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)