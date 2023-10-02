


class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    # 获取他的一个item，  dataset = Dataset(),   dataset[index]
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)