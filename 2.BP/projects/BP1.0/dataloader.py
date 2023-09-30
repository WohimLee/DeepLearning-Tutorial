
import numpy as np
import os.path as osp

from utils import mnist_labels, mnist_images, one_hot
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, root, train=True):
        self.root  = root
        self.train = train
        # Test
        if not self.train:
            self.image_path = osp.join(root, "t10k-images-idx3-ubyte")
            self.label_path = osp.join(root, "t10k-labels-idx1-ubyte")
        
        self.image_path = osp.join(root, "train-images-idx3-ubyte")
        self.label_path = osp.join(root, "train-labels-idx1-ubyte")
        
        self.initialize()
        
    def initialize(self):
        self.images = mnist_images(self.image_path)
        self.labels = one_hot(mnist_labels(self.label_path))
    
    def __len__(self):
        return len(self.images)
        
    def __getitem(self, index):
        return self.labels[index], self.images[index]
   

class MNISTLoader:
    def __init__(self, dataset: MNISTDataset, batch_size=32, shuffle=False):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.dataset_size = len(dataset)
        self.labels       = dataset.labels
        self.images       = dataset.images
        self.indice       = np.arange(self.dataset_size)
        
    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indice)
        return self
    
    def __next__(self):
        if self.cursor >= self.dataset_size:
            raise StopIteration
        
        range = min(self.batch_size, self.dataset_size - self.cursor)
        indice = self.indice[self.cursor : self.cursor+range]
        labels = self.labels[indice]
        images = self.images[indice]
        self.cursor += range
        return images, labels
        

        
if __name__ == '__main__':
    root = '/Users/azen/Desktop/myAir/Work/Workspace/Others/Dataset/MNIST'
    trainset    = MNISTDataset(root, train=True)
    trainloader = MNISTLoader(trainset, batch_size=64, shuffle=True)
    
    for images, labels in trainloader:
        # print(images.shape)
        # print(labels.shape)
        print(trainloader.cursor)
    
    