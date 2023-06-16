
import numpy as np
from nn import Module

class SigmoidCrossEntropy(Module):
    def __init__(self, params, weight_decay=1e-5):
        super().__init__("CrossEntropyLoss")
        self.params = params
        self.weight_decay = weight_decay
        
    def sigmoid(self, x):
        #return 1 / (1 + np.exp(-x))
        p0 = x < 0
        p1 = ~p0
        x = x.copy()
        x[p0] = np.exp(x[p0]) / (1 + np.exp(x[p0]))
        x[p1] = 1 / (1 + np.exp(-x[p1]))
        return x
    
    def decay_loss(self):
        loss = 0
        for p in self.params:
            loss += np.sqrt(np.sum(p.data ** 2)) / (2 * p.data.size) * self.weight_decay
        return loss
    
    def decay_backward(self):
        for p in self.params:
            eps = 1e-8
            p.grad += 1 / (2 * np.sqrt(np.sum(p.data ** 2)) + eps) / (2 * p.data.size) * self.weight_decay * 2 * p.data

    def forward(self, x, label_onehot):
        eps = 1e-6
        self.label_onehot = label_onehot
        self.predict = self.sigmoid(x)
        self.predict = np.clip(self.predict, a_max=1-eps, a_min=eps)  # 裁切
        self.batch_size = self.predict.shape[0]
        return -np.sum(label_onehot * np.log(self.predict) + (1 - label_onehot) * 
                        np.log(1 - self.predict)) / self.batch_size + self.decay_loss()
    
    def backward(self):
        self.decay_backward()
        return (self.predict - self.label_onehot) / self.batch_size


class SoftmaxCrossEntropy(Module):
    def __init__(self):
        super().__init__("SoftmaxCrossEntropy")
        
    def softmax(self, x):
        #return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x, label_onehot):
        eps = 1e-6
        self.label_onehot = label_onehot
        self.predict = self.softmax(x)
        self.predict = np.clip(self.predict, a_max=1-eps, a_min=eps)  # 裁切
        self.batch_size = self.predict.shape[0]
        return -np.sum(label_onehot * np.log(self.predict)) / self.batch_size
    
    def backward(self):
        return (self.predict - self.label_onehot) / self.batch_size
    