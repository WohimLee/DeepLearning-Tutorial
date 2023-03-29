

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


from torchvision import transforms




if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(root='datasets/', train=True, download=True, transform=transform)
    validset = torchvision.datasets.MNIST(root='datasets/', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=2)
    testloader  = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=2)
    

    