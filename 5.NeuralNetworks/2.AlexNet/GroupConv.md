

- [分组卷积 Group Converlution](https://zhuanlan.zhihu.com/p/490685194)
&emsp;
## Groups
Group Conv 最早出现在 AlexNet 中，因为显卡显存不够，只好把网络分在两块卡里，于是产生了这种结构；Alex 认为 group conv 的方式能够增加 filter 之间的对角相关性，而且能够减少训练参数，不容易过拟合，这类似于正则的效果



n PyTorch, the condition if groups < out_channels: is used to check if the number of groups used in a convolutional layer is less than the number of output channels in the layer.

In a convolutional layer with out_channels output channels and in_channels input channels, the layer can be divided into groups number of groups along the channel dimension. Each group will have out_channels / groups output channels and in_channels / groups input channels.

If groups == out_channels, then each output channel will only depend on a single input channel. This is known as depthwise convolution.

On the other hand, if groups == 1, then the layer will behave like a standard convolutional layer where each output channel depends on all the input channels.

The condition if groups < out_channels: checks if the number of groups used in the layer is less than the number of output channels. If this condition is true, it means that each output channel will depend on multiple input channels, which is known as a grouped convolution.

Grouped convolutions can help in reducing the computational cost and memory usage of the model, while also improving the model's performance. They have been used in various deep learning models, such as Google's Inception and ResNeXt architectures, to improve the accuracy of the models while keeping the computational cost reasonable.

- in_channels 和 out_channels 都必须能够被 groups 整除

普通卷积

<div align=center>
    <image src='imgs/conv.png' width=600>
</div>



分组卷积

<div align=center>
    <image src='imgs/group-conv.png' width=600>
</div>


