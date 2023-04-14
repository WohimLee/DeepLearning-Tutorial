
import torch
import torch.nn as nn





if __name__ == '__main__':
    groups = [1, 2, 4]
    x = torch.randn((1, 4, 32, 32))
    for g in groups:
        conv = nn.Conv2d(
            in_channels=4,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=g
        )
        output = conv(x)
        print("Groups: {}\nConv Weights: {}\nOutput: {}\n".format(
            g, conv.weight.data.size(), output.shape))
    pass