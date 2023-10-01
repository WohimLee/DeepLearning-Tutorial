
import re
import torch
import torchvision
import numpy as np
import torchvision.datasets as datasets

from torchvision import transforms 
from torch.utils.data import DataLoader


# trainset  = torchvision.datasets.MNIST(root="mnist/",train=True, download=True,transform=transform)
# validset  = torchvision.datasets.MNIST(root="mnist/",train=False,download=True,transform=transform)

# trainloader  =  torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=2)
# validloader  =  torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=2)


def xml_parse(file, label_map):
    with open(file) as f:
        data = f.read().replace("\t", "").replace("\n", "")
    objs = re.findall(r"<object>(.*?)</object>", data)
    obj_bboxes = []
    for obj in objs:
        xmin = re.findall(r"<xmin>(.*?)</xmin>", obj)[0]
        ymin = re.findall(r"<ymin>(.*?)</ymin>", obj)[0]
        xmax = re.findall(r"<xmax>(.*?)</xmax>", obj)[0]
        ymax = re.findall(r"<ymax>(.*?)</ymax>", obj)[0]
        name = re.findall(r"<name>(.*?)</name>", obj)[0]
        obj_bboxes.append((xmin, ymin, xmax, ymax, label_map.index(name)))
    res = np.zeros((0, 5), dtype = np.float32)
    if len(obj_bboxes) > 0:
        res = np.array(obj_bboxes, dtype=np.float32)
    return res