
import os
import torch
import os.path as osp
import torch.nn as nn



from utils import xml_parse


class VOCDataset:
    def __init__(self, root, train=True):
        self.root = root
        self.labels = 0
        self.images = 0
        self.initialize()
        pass
    
    
    def initialize(self):
        annotations_path = osp.join(self.root,'Annotations')
        JPEGImages_path  = osp.join(self.root,'JPEGImages')
        annotations = [osp.join(annotations_path, item) for item in os.listdir(annotations_path) if item.endswith('.xml')]
        JPEGImages  = [os.replace('xml', 'jpg', item) for item in annotations_path]
        
        
        
    
    def __getitem__(self, index):
        return self.labels[index], self.images[index]
    
    
    def __len__(self):
        return len(self.labels)
    
    


if __name__ == '__main__':
    root = '/Users/azen/Desktop/myAir/Work/Workspace/Others/Dataset/VOC2007/VOCdevkit/VOC2007'
    vocdataset = VOCDataset(root=root)
    pass




