&emsp;
# Pool2d

- input: $(batch size，channels，H_{in}，W_{in})$
- output: $(batch size，channels，H_{out}，W_{out})$
$$H_{out} = \frac{(H - pool_{size})}{stride} + 1$$
$$W_{out} = \frac{(W - pool_{size})}{stride} + 1$$

&emsp;
# MaxPool2d

Given an input tensor of shape (batch_size, in_channels, H, W), where batch_size is the number of input samples in a batch, in_channels is the number of input channels, and H and W are the spatial dimensions of the input, nn.MaxPool2d performs the following operation:

1. Divides the input into non-overlapping rectangular regions, or windows, of size (kernel_size, kernel_size).

2. For each window, takes the maximum value of the elements inside it.

3. Outputs: (batch_size, in_channels, H'/stride, W'/stride), where H' and W' are the spatial dimensions of the output tensor, which are computed as the formula above:


&emsp;
# AvgPool2d