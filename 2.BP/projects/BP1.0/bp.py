
import torch

import numpy as np
import torch.nn as nn


class Module:
    name : str
    training : bool
    def __init__(self):
        super().__setattr__('name', self.__class__.__name__)
        super().__setattr__('training', True)
        
    def forward(self, *args, **kwargs):
        return
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __repr__(self):
        return self.name
    

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
        

class Linear(Module):
    def __init__(self, in_fearures, out_features):
        self.in_features  = in_fearures
        self.out_features = out_features
        self.weight = Parameter(np.ones((out_features, in_fearures)))
        self.bias   = Parameter(np.zeros(out_features))
        
    def forward(self, x):
        self.__x = x
        return x@self.weight.data.T + self.bias.data
    # Y = X@W.T + B
    def backward(self, G):
        # G.T @ X
        self.weight.grad = G.T @ self.__x
        # G
        self.bias.grad   = np.sum(G, axis=0)
        return G @ self.weight.data


class Linear(Module):
    def __init__(self, input_feature, output_feature):
        super().__init__("Linear")
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weights = Parameter(np.ones((input_feature, output_feature)))
        self.bias = Parameter(np.zeros((1, output_feature)))
        
        
    def forward(self, x):
        self.x_save = x.copy()
        return x @ self.weights.value + self.bias.value
    
    #AB = C  G
    #dB = A.T @ G
    #dA = G @ B.T
    def backward(self, G):
        self.weights.delta += self.x_save.T @ G
        self.bias.delta += np.sum(G, 0)  #值复制
        return G @ self.weights.value.T
        
        
l1 = Linear(2, 3)
input = np.arange(5*2).reshape(5, 2)
output = l1(input)
print(output)

    

        
        


