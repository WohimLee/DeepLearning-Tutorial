



## Padding

## Stride

>卷积后的 $W、H$
$$W_{out} = \frac{(W_{in} + 2*Padding - K_{size})}{Stride} +1$$

$$H_{out} = \frac{(H_{in} + 2*Padding - K_{size})}{Stride} +1$$
- 这里的 $K_{size}$ 宽高一样，如果卷积核宽高不一样，对应使用卷积核的 $K_{width}、K_{height}$
- 如果想让输入不被下采样或者想让输出都一致，这个公式很有用

## Weights



## Bias


