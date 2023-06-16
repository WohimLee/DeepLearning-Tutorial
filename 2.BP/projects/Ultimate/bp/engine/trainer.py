



import numpy as np
import os.path as osp

from dataloader import Dataset, DataLoader
from nn import Module, ModuleList, Linear, ReLU, Dropout
from loss import SigmoidCrossEntropy
from optim import Adam
from utils import load_labels, load_images, one_hot, estimate_val


class Model(Module):
    def __init__(self, num_feature, num_hidden, num_classes):
        super().__init__("Model")
        self.backbone = ModuleList(
            Linear(num_feature, num_hidden),
            ReLU(),
            Dropout(),
            Linear(num_hidden, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def backward(self, G):
        return self.backbone.backward(G)
    
    
def train(epochs = 10, lr=1e-2, batch_size = 64, classes = 10):
    np.random.seed(3)
    numdata, data_dims = train_images.shape  # 60000, 784

    # 定义dataloader和dataset，用于数据抓取
    train_data = DataLoader(Dataset(train_images, one_hot(train_labels, classes)), batch_size, shuffle=True)
    model = Model(data_dims, 256, classes)
    #loss_func = SoftmaxCrossEntropy()
    loss_func = SigmoidCrossEntropy(model.params(), 0)
    optim = Adam(model, lr)
    iters = 0   # 定义迭代次数，因为我们需要展示loss曲线，那么x将会是iters

    lr_schedule = {
        5: 1e-3,
        15: 1e-4,
        18: 1e-5
    }

    # 开始进行epoch循环，总数是epochs次
    for epoch in range(epochs):
        
        if epoch in lr_schedule:
            lr = lr_schedule[epoch]
            optim.set_lr(lr)
        
        model.train()
        # 对一个批次内的数据进行迭代，每一次迭代都是一个batch（即256）
        for index, (images, labels) in enumerate(train_data):
            
            x = model(images)
            
            # 计算loss值
            loss = loss_func(x, labels)
            
            optim.zero_grad()
            G = loss_func.backward()
            model.backward(G)
            optim.step()   # 应用梯度，更新参数
            iters += 1
            if index % 200 == 0:
                print("Epoch: {} / {}, Iter: {}, Loss: {:.3f}, LR: {:g}".format(
                    epoch, epochs, iters, loss, lr
                ))
        
        model.eval()
        val_accuracy, val_loss = estimate_val(model(test_images), test_labels, classes, loss_func)
        print("\nTest Result: Acc: {:.2f}%, Loss: {:.3f}\n".format(
            val_accuracy*100, val_loss
        ))
        
        

    

if __name__ == '__main__':
    root = "/Users/azen/Desktop/myAir/Work/Workspace/Others/Dataset/MNIST"
    test_labels = load_labels(osp.join(root, "t10k-labels-idx1-ubyte"))   #  10000,
    test_images = load_images(osp.join(root, "t10k-images-idx3-ubyte"))   #  10000, 784
    test_images = (test_images - np.mean(test_images)) / np.var(test_images)


    train_labels = load_labels(osp.join(root, "train-labels-idx1-ubyte")) # 60000,
    train_images = load_images(osp.join(root, "train-images-idx3-ubyte")) # 60000, 784
    train_images = (train_images - np.mean(train_images)) / np.var(train_images)
    
    train()



    