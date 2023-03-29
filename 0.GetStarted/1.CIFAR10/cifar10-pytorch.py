import time
import torch
import torchvision

from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


 
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
 
    def forward(self,x):
        x = self.model(x)
        return x
    
if __name__ == '__main__':

    # 准备数据集
    train_data = torchvision.datasets.CIFAR10("dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data  = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_data_size = len(train_data)
    test_data_size  = len(test_data)
    print("训练数据集的长度为{}".format(train_data_size))
    print("测试数据集的长度为{}".format(test_data_size))
    
    # 利用DataLoader来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader  = DataLoader(test_data,  batch_size=64)

    # 定义训练设备
    device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Model()
    model = model.to(device)
    
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    
    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step  = 0
    epoch = 30
    
    # 添加Tensorboard
    writer = SummaryWriter("logs_train")


    start_time = time.time()
    for i in range(epoch):
        print("-----第{}轮训练开始------".format(i+1))
    
        # 训练步骤开始
        model.train()
        for data in train_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            loss = loss_fn(output, targets)
    
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("训练次数{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
    
        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy
    
        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
        total_test_step += 1
    
        torch.save(model, "test_{}.pth".format(i))
        print("模型已保存")
    
    writer.close()


    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    image_path = "./imgs/dog.png"
    image = Image.open(image_path)
    # print(image)
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])
    
    image = transform(image)
    # print(image.shape)
    
    
    model = torch.load("test_99.pth", map_location=torch.device('cpu'))
    # print(model)
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    ret = output.argmax(1)
    ret = ret.numpy()
    print("预测结果为:{}".format(CLASSES[ret[0]]))