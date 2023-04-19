&emsp;
# Identity


nn.Identity is a PyTorch module that represents an identity function. It simply returns its input tensor as its output tensor, without applying any transformation.

The nn.Identity module is useful in cases where you want to create a neural network architecture that passes its input through a sequence of layers, but you want to skip certain layers and pass the input directly to the output. In this case, you can use nn.Identity as a placeholder module for the skipped layers.

>Example
```py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.identity(x)  # skip conv3
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x
```




In this example, we define a PyTorch model MyModel that consists of three convolutional layers and a max pooling layer. We use nn.Identity as a placeholder module to skip the third convolutional layer. The forward() method of the model applies the convolutional and pooling layers in sequence, and uses the nn.Identity module to skip the third convolutional layer.

Note that in this example, we use the torch.relu() function to apply the ReLU activation function to the output of the convolutional layers. Alternatively, we could use the nn.ReLU module to achieve the same result.


# 拟 Identity
>Example
```py
class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super().__init__(in_channels  = channels,
                         out_channels = channels,
                         kernel_size  = 1,
                         stride       = 1,
                         padding      = 0,
                         groups       = groups,
                         bias         = False)
        assert channels % groups == 0
        input_dim = channels // groups
        id_value  = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):

        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input,
                          kernel,
                          None,
                          stride=1,
                          padding=0,
                          dilation=self.dilation,
                          groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)
```


This class sets the kernel variable as self.weight + self.id_tensor.to(self.weight.device) in the forward method to add an identity matrix to the weight kernel, allowing it to act as an identity operation in addition to the normal convolutional operation.

The self.id_tensor attribute is a diagonal matrix with the value of 1 for the diagonal elements and 0 for other elements. By adding this identity matrix to the weight tensor, the convolution operation will behave as the identity operation for some input samples. This is because the convolution operation with the identity matrix results in the same input tensor as the output tensor.

Therefore, by adding the identity matrix to the weight tensor, the IdentityBasedConv1x1 module can perform two operations simultaneously: normal convolution operation and identity operation. This makes the module more flexible and powerful, and it can be useful in many applications, such as in residual networks and in neural architecture search.





