from common import Module

class ReLULayer(Module):
    def __init__(self, inplace=True):
        super().__init__("ReLU")
        self.inplace = inplace
        
    def forward(self, x):
        self.negative_position = x < 0
        if not self.inplace:
            x = x.copy()
            
        x[self.negative_position] = 0
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()
            
        G[self.negative_position] = 0
        return G