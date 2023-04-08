

- [分组卷积 Group Converlution](https://zhuanlan.zhihu.com/p/490685194)
&emsp;
## Groups
Group Conv 最早出现在 AlexNet 中，因为显卡显存不够，只好把网络分在两块卡里，于是产生了这种结构；Alex 认为 group conv 的方式能够增加 filter 之间的对角相关性，而且能够减少训练参数，不容易过拟合，这类似于正则的效果



- in_channels 和 out_channels 都必须能够被 groups 整除

普通卷积

<div align=center>
    <image src='imgs/conv.png' width=600>
</div>



分组卷积

<div align=center>
    <image src='imgs/group-conv.png' width=600>
</div>


