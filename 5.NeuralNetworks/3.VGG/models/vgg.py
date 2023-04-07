
import torch.nn as nn

defaultcfg = {
    11 : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512          ],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512          ],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512     ],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=11, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, l, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]

                in_channels = l
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    net = VGG()
