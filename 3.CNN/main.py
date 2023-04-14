import torch.nn as nn

nnConv = nn.Conv2d()
'''
in_channels : int, 
out_channels: int, 
kernel_size : _size_2_t, 
stride      : _size_2_t       = 1, 
padding     : _size_2_t | str = 0, 
dilation    : _size_2_t       = 1, 
groups      : int             = 1, 
bias        : bool            = True, 
padding_mode: str             = 'zeros', 
device                        = None, 
dtype                         = None
'''


class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Conv2d_1:
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       padding_mode = 'zeros',
                       device       = None,
                       dtype        = None):
        pass


pool = nn.MaxPool2d(
    kernel_size=2, 
    stride=1,
    padding=0
)

