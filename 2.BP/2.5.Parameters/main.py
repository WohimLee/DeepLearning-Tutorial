

import torch.nn as nn
from torch.nn.parameter import Parameter

linear = nn.Linear(32, 64)
print(type(linear.weight))
print(linear.weight.data.shape())
print(linear.weight.grad.shape())

p = Parameter()