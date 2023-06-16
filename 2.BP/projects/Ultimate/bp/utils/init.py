
import numpy as np

    
# nn.init.xavier_normal()
# nn.init.kaiming_normal()
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
        
    def apply(self, data):
        data[...] = np.random.normal(self.mu, self.sigma, data.shape)
