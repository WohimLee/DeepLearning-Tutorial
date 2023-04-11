&emsp;
# nn.Conv2d



The padding parameter can take different values:

- padding=0 (default): no padding is added to the input tensor before the convolution operation.
- padding=(pad_h, pad_w): adds pad_h zeros to the top and bottom of the input tensor and pad_w zeros to the left and right of the input tensor.
- padding=same: adds padding to the input tensor such that the output size of the convolution operation is the same as the input size. This is useful for designing convolutional networks that do not change the spatial dimensions of the input tensor.
- padding=valid: no padding is added to the input tensor, and the convolution operation is applied only to the pixels where the filter fits entirely within the input tensor. This is equivalent to zero-padding of size (filter_size - 1) / 2 when the filter size is odd.