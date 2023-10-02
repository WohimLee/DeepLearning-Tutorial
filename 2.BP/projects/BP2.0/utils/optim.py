
import numpy as np


class Optimizer:
    def __init__(self, name, model, lr):
        self.name = name
        self.model = model
        self.lr = lr
        self.params = model.params()
                
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
            
    def set_lr(self, lr):
        self.lr = lr
        
class SGD(Optimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__("SGD", model, lr)
    
    def step(self):
        for param in self.params:
            param.value -= self.lr * param.delta
            
class SGDMomentum(Optimizer):
    def __init__(self, model, lr=1e-3, momentum=0.9):
        super().__init__("SGDMomentum", model, lr)
        self.momentum = momentum
        
        for param in self.params:
            param.v = 0
    
    # 移动平均
    def step(self):
        for param in self.params:
            param.v = self.momentum * param.v - self.lr * param.delta
            param.value += param.v
            
class Adam(Optimizer):
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, l2_regularization = 0):
        super().__init__("Adam", model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.l2_regularization = l2_regularization
        self.t = 0
        
        for param in self.params:
            param.m = 0
            param.v = 0
            
    # 指数移动平均
    def step(self):
        eps = 1e-8
        self.t += 1
        for param in self.params:
            g = param.delta
            param.m = self.beta1 * param.m + (1 - self.beta1) * g
            param.v = self.beta2 * param.v + (1 - self.beta2) * g ** 2
            mt_ = param.m / (1 - self.beta1 ** self.t)
            vt_ = param.v / (1 - self.beta2 ** self.t)
            param.value -= self.lr * mt_ / (np.sqrt(vt_) + eps) + self.l2_regularization * param.value
    