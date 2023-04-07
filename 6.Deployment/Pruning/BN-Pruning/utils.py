import os
import torch
import shutil

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth'), os.path.join(filepath, 'model_best.pth'))


def get_training_dataloader(batch_size=16, **kwargs):
    
    train_dataset = datasets.CIFAR10(root='./data.cifar10', train=True, download=True)
    mean = train_dataset.data.mean(axis=(0,1,2)) / 255
    std  = train_dataset.data.std(axis=(0,1,2))  / 255
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset.transform = transform_train
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader

def get_test_dataloader(batch_size=16, **kwargs):

    test_dataset = datasets.CIFAR10(root='./data.cifar10', train=False, download=True)
    mean = test_dataset.data.mean(axis=(0,1,2)) / 255
    std  = test_dataset.data.std(axis=(0,1,2))  / 255
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset.transform = transform_test
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return test_loader

