&emsp;
# BatchNorm2d

- 相关论文：Batch Normalization- Accelerating Deep Network Training b y Reducing Internal Covariate Shift (2015)
- 知乎解读：[Batch Normalization 原理与实战
](https://zhuanlan.zhihu.com/p/34879333)

$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$$

>Batch Normalize 的作用
- 加快收敛、提升精度：对输入进行归一化，从而使得优化更加容易
- 减少过拟合：可以减少方差的偏移
- 可以使得神经网络使用更高的学习率：BN 使得神经网络更加稳定，从而可以使用更大的学习率，加速训练过程
- 甚至可以减少 Dropout 的使用：因为 BN 可以减少过拟合，所以有了 BN，可以减少其他正则化技术的使用
- input_size: (batch_size, channels, H, W)
- output_size: (batch_size, channels, H, W)


&emsp;
>nn.BatchNorm2d 参数
- num_features: int
- eps: float = 0.00001
- momentum: float = 0.1
- affine: bool = True
    - True (the default): 有两个可学习参数: 
      - weight: scales the normalized outputs of the layer, 
      - bias: shifts the normalized outputs
    - False: 不设置可学习参数，没有 scaled 和 shifted，只做 normalize
- track_running_stats: bool = True
- device


&emsp;
>nn.BatchNorm2d 属性
- weights: 也就是 $\gamma$，(channels)
- bias: 也就是 $\beta$，(channels)
- running_mean: 
- running_var: 
