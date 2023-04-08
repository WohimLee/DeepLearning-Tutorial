

import torch.nn as nn





if __name__ == '__main__':
    groups = [1, 2, 4]
    for g in groups:
        conv = nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=g
        )
        print("Groups: {}\nConv Weights: {}".format(g, conv.weight.data.size()))
    pass