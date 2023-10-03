import numpy as np

from utils.utils import estimate_val
from utils.dataset import Dataset
from utils.dataloader import DataLoader
from utils.loss import SigmoidCrossEntropy
from utils.optim import Adam
from utils.model import Model



config_file = './cfg/MNIST.yaml'
trainset = Dataset(config=config_file, train=True)
testset  = Dataset(config=config_file, train=False)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(trainset, batch_size=64, shuffle=True)

np.random.seed(3) # 保证每次实验结果一样

features = 784
classes  = 10   

lr = 1e-2
lr_schedule = {
    5 : 1e-3,
    15: 1e-4,
    18: 1e-5
}
epochs = 20 
batch_size = 64


model = Model(features, 1024, classes)
#loss_func = SoftmaxCrossEntropy()
loss_func = SigmoidCrossEntropy(model.params(), 0)
optim = Adam(model, lr)
iters = 0   # 定义迭代次数，因为我们需要展示loss曲线，那么 x 将会是iters



print("Start Training...")
# 开始进行epoch循环，总数是epochs次
for epoch in range(epochs):
    
    if epoch in lr_schedule:
        lr = lr_schedule[epoch]
        optim.set_lr(lr)
    
    model.train()
    # 对一个批次内的数据进行迭代，每一次迭代都是一个batch（即256）
    for index, (labels, images) in enumerate(trainloader):
        x = model(images)
        
        # 计算loss值
        loss = loss_func(x, labels)
        
        optim.zero_grad()
        G = loss_func.backward()
        model.backward(G)
        optim.step()   # 应用梯度，更新参数
        iters += 1
        if iters % 200 == 0:
            print(f"Epoch: {epoch} / {epochs}, Iters: {iters}, Loss: {loss:.3f}, LR: {lr:g}")
    
    model.eval()
    val_accuracy, val_loss = estimate_val(model(testset.images), testset.labels, classes, loss_func)
    print(f"Val set, Accuracy: {val_accuracy*100:.3f}%, Loss: {val_loss:.5f}")