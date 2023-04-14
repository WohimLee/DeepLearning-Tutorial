



## Padding

The condition padding - kernel_size // 2 >= 0 is used to check whether the specified padding value is large enough to ensure that the convolution operation can be applied to all pixels in the input tensor without exceeding its boundaries.

If padding - kernel_size // 2 >= 0, it means that there is enough padding to ensure that the convolution operation can be applied to all pixels in the input tensor without running out of bounds. In this case, the hor_padding and ver_padding variables are set such that the same amount of padding is added to both sides of the input tensor along the horizontal and vertical dimensions.

If padding - kernel_size // 2 < 0, it means that the specified padding value is not enough to apply the convolution operation to all pixels in the input tensor without running out of bounds. In this case, the hor_padding and ver_padding variables are set such that the convolution operation is only applied to a cropped version of the input tensor, with self.crop pixels removed from the left and right sides (along the horizontal dimension) or top and bottom sides (along the vertical dimension).

By checking the condition padding - kernel_size // 2 >= 0, the code ensures that the convolution operation is applied to as much of the input tensor as possible, while also avoiding errors caused by attempting to apply the convolution operation to pixels that lie outside the boundaries of the input tensor.


```py
if padding - kernel_size // 2 >= 0:
    self.crop    = 0
    hor_padding  = [padding - kernel_size // 2, padding]
    ver_padding  = [padding, padding - kernel_size // 2]
else:
    self.crop    = kernel_size // 2 - padding
    hor_padding  = [0, padding]
    ver_padding  = [padding, 0]
```

The self.crop attribute is used to store the amount of cropping that needs to be applied to the input tensor before the convolution operation can be applied.

If self.crop is zero, it means that there is enough padding to ensure that the convolution operation can be applied to all pixels in the input tensor without running out of bounds, and therefore no cropping is necessary.

If self.crop is greater than zero, it means that the specified padding value is not enough to apply the convolution operation to all pixels in the input tensor without running out of bounds, and cropping needs to be applied to the input tensor to ensure that the convolution operation can be applied to a valid region of the input tensor.

The self.crop attribute is used later in the forward method to crop the input tensor along the horizontal and vertical dimensions if self.crop is greater than zero. Specifically, the ver_input and hor_input variables are created by slicing the input tensor to remove self.crop pixels from the left and right sides (along the horizontal dimension) or top and bottom sides (along the vertical dimension). This ensures that the convolution operation is applied only to the valid region of the input tensor, and not to pixels that lie outside the boundaries of the input tensor.


In cases where padding is not sufficient to prevent the filter from extending beyond the boundaries of the input tensor, cropping may be used to remove the outer rows and/or columns of the input tensor. Cropping involves removing a specified number of rows and/or columns from the edges of the input tensor to ensure that the filter remains fully overlaid with the input tensor at each position

>padding after a BN layer

The reason why some people set padding after a BatchNorm layer using BatchNorm statistics instead of setting padding while operating a Conv layer or an AvgPool layer is to avoid information loss.

When padding is applied to a Conv layer or an AvgPool layer, the padded values are typically set to zero. This can cause information loss because the mean and variance of the input data may change due to the zero-padding. When the input data distribution changes, the statistics of the BatchNorm layer can become inaccurate and lead to degraded performance.

To avoid this issue, some people choose to apply padding after the BatchNorm layer using the statistics computed during training. By doing so, the padding values are added to the input data after normalization, and the BatchNorm statistics are preserved. This can help to maintain the accuracy of the BatchNorm layer and improve the overall performance of the model.

However, it is worth noting that this approach may not always be necessary or beneficial for all models and tasks. It is important to carefully consider the specific requirements of each model and experiment to determine the most appropriate approach for applying padding and BatchNorm layers.

## Stride

>卷积后的 $W、H$
$$W_{out} = \frac{(W_{in} + 2*Padding - K_{size})}{Stride} +1$$

$$H_{out} = \frac{(H_{in} + 2*Padding - K_{size})}{Stride} +1$$
- 这里的 $K_{size}$ 宽高一样，如果卷积核宽高不一样，对应使用卷积核的 $K_{width}、K_{height}$
- 如果想让输入不被下采样或者想让输出都一致，这个公式很有用

## Weights



## Bias

People often set the bias to False while training neural networks because it can help to reduce overfitting and improve generalization performance.

A bias term is a learnable parameter in a neural network that is added to the output of each convolutional or fully connected layer. It allows the network to learn a shift in the activation function, which can improve the ability of the network to fit the training data. However, if the network is overfitting to the training data, it may learn to rely too heavily on the bias term, which can hurt its ability to generalize to new data.

By setting the bias to False, the network is forced to learn representations that do not rely on a shift in the activation function. This can help to prevent overfitting and improve the network's ability to generalize. Additionally, setting the bias to False can reduce the number of parameters in the model, which can help to improve computational efficiency and reduce the risk of overfitting.

It's worth noting that setting the bias to False is not always necessary or desirable, and it depends on the specific problem and architecture. In some cases, the bias term can be important for achieving good performance. Therefore, it's important to experiment with different configurations and evaluate their performance on a validation set to determine the best approach for a particular problem.



