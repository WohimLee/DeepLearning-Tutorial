
import numpy as np

class Module:
    def __init__(self, name):
        self.name = name
        self.train_mode = False
        
    def __call__(self, *args):
        return self.forward(*args)
    
    def train(self):
        self.train_mode = True
        for m in self.modules():
            m.train()
        
    def eval(self):
        self.train_mode = False
        for m in self.modules():
            m.eval()
        
    def modules(self):
        ms = []
        for attr in self.__dict__:
            m = self.__dict__[attr]
            if isinstance(m, Module):
                ms.append(m)
        return ms
    
    def params(self):
        ps = []
        for attr in self.__dict__:
            p = self.__dict__[attr]
            if isinstance(p, Parameter):
                ps.append(p)
            
        ms = self.modules()
        for m in ms:
            ps.extend(m.params())
        return ps
    
    def info(self, n):
        ms = self.modules()
        output = f"{self.name}\n"
        for m in ms:
            output += ('  '*(n+1)) + f"{m.info(n+1)}\n"
        return output[:-1]
    
    def __repr__(self):
        return self.info(0)
    
class Initializer:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, *args):
        return self.apply(*args)
        
class GaussInitializer(Initializer):
    # where :math:`\mu` is the mean and :math:`\sigma` the standard
    # deviation. The square of the standard deviation, :math:`\sigma^2`,
    # is called the variance.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def apply(self, value):
        value[...] = np.random.normal(self.mu, self.sigma, value.shape)
    
class Parameter:
    def __init__(self, value):
        self.value = value
        self.delta = np.zeros(value.shape)
        
    def zero_grad(self):
        self.delta[...] = 0
        
class Linear(Module):
    def __init__(self, input_feature, output_feature):
        super().__init__("Linear")
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weights = Parameter(np.zeros((input_feature, output_feature)))
        self.bias = Parameter(np.zeros((1, output_feature)))
        
        # 权重初始化 
        initer = GaussInitializer(0, np.sqrt(2 / input_feature))  # np.sqrt(2 / input_feature)
        initer.apply(self.weights.value)
        
    def forward(self, x):
        self.x_save = x.copy()
        return x @ self.weights.value + self.bias.value
    
    # XW.T = C,  G
    # dW.T = G @ W
    # dX = G.T @ W
    def backward(self, G):
        self.weights.delta += self.x_save.T @ G
        self.bias.delta += np.sum(G, 0)  #值复制
        return G @ self.weights.value.T
    
class ReLU(Module):
    def __init__(self, inplace=True):
        super().__init__("ReLU")
        self.inplace = inplace
        
    # 亿点点
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

def sigmoid(x):
    p0 = x < 0
    p1 = ~p0
    x = x.copy()

    # 如果x的类型是整数，那么会造成丢失精度
    x[p0] = np.exp(x[p0]) / (1 + np.exp(x[p0]))
    x[p1] = 1 / (1 + np.exp(-x[p1]))
    return x

class SWish(Module):
    def __init__(self):
        super().__init__("SWish")
        
    def forward(self, x):
        self.x_save = x.copy()
        self.sx = sigmoid(x)
        return x * self.sx
    
    def backward(self, G):
        return G * (self.sx + self.x_save * self.sx * (1 - self.sx))
    
class Dropout(Module):
    def __init__(self, prob_keep=0.5, inplace=True):
        super().__init__("Dropout")
        self.prob_keep = prob_keep
        self.inplace = inplace
        
    def forward(self, x):
        if not self.train_mode:
            return x
        
        self.mask = np.random.binomial(size=x.shape, p=1 - self.prob_keep, n=1)
        if not self.inplace:
            x = x.copy()
            
        x[self.mask] = 0
        x *= 1 / self.prob_keep
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()
        G[self.mask] = 0
        G *= 1 / self.prob_keep
        return G
    
class ModuleList(Module):
    def __init__(self, *args):
        super().__init__("ModuleList")
        self.ms = list(args)
        
    def modules(self):
        return self.ms
    
    def forward(self, x):
        for m in self.ms:
            x = m(x)
        return x
    
    def backward(self, G):
        for i in range(len(self.ms)-1, -1, -1):
            G = self.ms[i].backward(G)
        return G
    