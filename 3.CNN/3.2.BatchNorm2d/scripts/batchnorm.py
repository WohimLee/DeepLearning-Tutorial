
import torch.nn as nn


conv = nn.Conv2d(
    in_channels=3, 
    out_channels=12,
    kernel_size=3
)

bn = nn.BatchNorm2d(
    num_features=32, 
    eps=1e-5,
    affine=True,
    track_running_stats=True
)