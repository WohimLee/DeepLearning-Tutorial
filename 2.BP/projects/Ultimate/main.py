

from utils.data import Dataset
from utils.data import DataLoader







if __name__ == '__main__':
    config_file = './config/MNIST.yaml'
    trainset = Dataset(config=config_file, train=True)
    testset  = Dataset(config=config_file, train=False)
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader  = DataLoader(trainset, batch_size=32, shuffle=True)

    for num, (labels, images) in enumerate(testloader):
        print(num, labels.shape, images.shape)

    

    pass