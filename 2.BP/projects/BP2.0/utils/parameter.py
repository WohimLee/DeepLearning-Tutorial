
import numpy as np

class Parameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = np.zeros(data.shape)
        
    def zero_grad(self):
        self.grad[...] = 0