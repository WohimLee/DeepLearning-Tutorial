&emsp;
# Intro
- Reference: [一文看尽深度学习中的15种损失函数](https://zhuanlan.zhihu.com/p/377799012)

我们来思考一个非常简单的问题：给定两个数 $a$ 和 $b$，我们怎么去衡量他们有多相近？给大家列几个常见的方法
- $a-b$: 第一个是作差，但是会有正负问题，引出绝对值
- $|a-b|$: 又被称为 "曼哈顿距离", 如果觉得这个词难理解，直接忽视
- $(a-b)^2$: 又被称为 "欧式距离", 备注同上

大家看一下上面这几种方法，各有什么特点？

似乎有点难看出来，那我们换成深度学习里面的例子，我们通过网络输出了一个预测值（predict），如何判断它有多接近数据集的真值（target, ground true）？上面的例子就变成了：
- $predict-target$
- $|predict-target|$
- $(predict-target)^2$



&emsp;
# 1 Loss Function 损失函数
- Loss function，即损失函数： `单个训练样本` 与 $Target$ 之间的误差；
- Cost function，即代价函数：`单个batch/整个trainset` 与 $Target$ 之间的误差，可以理解为 Loss function 的 Vector-Matrix Form
  
损失函数（Loss function）是用来度量 model 的预测值 $Predict$ 与真实值 $Target$ 的差异程度的运算函数，它是一个非负实值函数，我们期望损失函数尽可能的小，回想梯度下降，是不是觉得有点相似？

>为什么使用损失函数
- 损失函数主要在模型的训练阶段使用
- `Step1-Forward`: 每个 batch 的 train data 输入模型，通过前向传播输出预测值 $Predict$
- `Step2-Loss`: 然后损失函数会计算出 $Predict$ 和 $Target$ 之间的差异值，也就是损失值
- `Step3-Backward`: 得到损失值之后，模型通过反向传播（Back Propagation）去更新各个参数，来降低真实值与预测值之间的损失，使得模型的 $Predict$ 尽可能接近 $Target$，从而达到学习的目的


&emsp;
# 2 常见的 Loss
## 2.1 MAE（L1 Loss）
>平均绝对误差（Mean Absolute Error）
$$Loss = \frac{1}{n}\sum\limits^{N}_{i=1}|Y_{predict} - Y_{target}|$$

$L_1$ 损失又称为曼哈顿距离，表示残差的绝对值之和。$L_1$ 损失函数对离群点有很好的鲁棒性，但它在残差为零处却不可导。另一个缺点是更新的梯度始终相同，也就是说，即使很小的损失值，梯度也很大，这样不利于模型的收敛。针对它的收敛问题，一般的解决办法是在优化算法中使用变化的学习率，在损失接近最小值时降低学习率

&emsp;
## 2.2 MSE（L2 Loss）
>均方差（Mean Squred Error）

$$Loss = \frac{1}{n}\sum\limits^{N}_{i=1}(Y_{predict} - Y_{target})^2$$
- $n$: $(Y_{predict} - Y_{target})$ 这样的对数

$L_2$ 损失又被称为欧氏距离，是一种常用的距离度量方法，通常用于度量数据点之间的相似度。由于 $L_2$ 损失具有凸性和可微性，且在独立、同分布的高斯噪声情况下，它能提供最大似然估计，使得它成为回归问题、模式识别、图像处理中最常使用的损失函数

&emsp;
## 2.3 IoU Loss
- 用于目标检测的 Loss
>变种
- GIoU
- DIoU
- CIoU
  
&emsp;
## 2.4 Entropy Loss
- Entropy
- Cross Entropy

