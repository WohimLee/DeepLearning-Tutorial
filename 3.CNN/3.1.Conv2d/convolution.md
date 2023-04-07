



## Padding

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
