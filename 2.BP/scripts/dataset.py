import random
import torch
from torch.utils.data import DataLoader, Dataset

from utils import *

# torch.Dataset
# dataloader = DataLoader()
'''
dataset           : Dataset, 
batch_size        : int | None = 1, 
shuffle           : bool       = False, 
sampler           : Sampler[int] | None           = None, 
batch_sampler     : Sampler[Sequence[int]] | None = None, 
num_workers       : int        = 0, 
collate_fn        : (List[T@_collate_fn_t]) -> Any | None = None, 
pin_memory        : bool       = False, 
drop_last         : bool       = False, 
timeout           : float      = 0, 
worker_init_fn    : _worker_init_fn_t | None      = None, 
multiprocessing_context        = None, 
generator                      = None, *, 
prefetch_factor   : int        = 2, 
persistent_workers: bool       = False
'''

class MNISTDataset:
    def __init__(self, images, labels, overwrite = False):
        self.images = load_images(images)
        self.labels = load_labels(labels)
        self.overwrite = overwrite

        self.all_data = []
        cache_name  = get_md5(images)
        cache_file  = f"runs/dataset_cache/{cache_name}.cache"
        if overwrite:
            self.build_cache(cache_file)
        else :
            if os.path.exists(cache_file):
                self.load_cache(cache_file)
            else :
                self.build_cache()

    def build_cache(self, cache_file):
        
        images = norm_images(self.images)
        # labels = one_hot(labels)
        self.all_data = list(zip(images, self.labels))
        mkparent(cache_file)
        torch.save(self.all_data, cache_file)

    def load_cache(self, cache_file):
        self.all_data = torch.load(cache_file)

    def __getitem__(self, index):
        return self.all_data[index]
    
    def __len__(self):
        return len(self.images)


class MNISTDataLoader:
    def __init__(self, dataset, batch_size, shuffle = True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        
    def __iter__(self):
        return MNISTDataLoaderIterator(self)
    
    def __len__(self):
        return len(self.dataset)
    
class MNISTDataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset    = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.shuffle    = dataloader.shuffle
        self.collate_fn = dataloader.collate_fn
        self.cursor = 0
        self.indice = list(range(len(dataloader))) 
        if self.shuffle:
            random.shuffle(self.indice)
            
    def __next__(self):
        if self.cursor >= len(self.dataloader):
            raise StopIteration()
            
        batch_data = []
        batch_size = min(self.batch_size, len(self.dataloader) - self.cursor)  

        for n in range(batch_size):

            index = self.indice[self.cursor]
            data = self.dataset[index]
          
            # 如果batch没有初始化，则初始化n个list成员
            if len(batch_data) == 0:
                batch_data = [[] for i in range(len(data))]
                
            # 遍历2次，第0次是image，第1次是label
            for index, item in enumerate(data):
                batch_data[index].append(item)
            self.cursor += 1
        # 遍历2次，第0次是batch_size 个 image，第1次是 batch_size 个 label
        for index in range(len(batch_data)):
            batch_data[index] = np.vstack(batch_data[index])
        return batch_data

if __name__ == "__main__":
    train_images = "/datav/Dataset/MNIST/train-images-idx3-ubyte"
    train_labels = "/datav/Dataset/MNIST/train-labels-idx1-ubyte"
    test_images  = "/datav/Dataset/MNIST/t10k-images-idx3-ubyte"
    test_labels  = "/datav/Dataset/MNIST/t10k-labels-idx1-ubyte"

    # train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset  = MNISTDataset(test_images, test_labels)
    test_loader   = MNISTDataLoader(test_dataset, 32, True) # image(32, 784) label(32, 1)
    i = 0
    for batch_data in test_loader:
        print(i, batch_data[0].shape, batch_data[1].shape)
        i+=1






    





