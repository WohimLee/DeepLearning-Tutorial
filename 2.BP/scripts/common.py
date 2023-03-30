
import numpy as np
import torch.nn as nn


class Module:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.weights = np.zeros((in_features, out_features))
        self.bias    = np.zeros((1, out_features))
        self.d_w     = np.zeros(self.weights.shape)
        self.d_b     = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x_save = x.copy()
        return x @ self.weights + self.bias

    def backward(self, G):
        self.d_w = self.x_save.T @ G
        self.d_b = np.sum(G, 0)
        return G @ self.weights.T


















 
    