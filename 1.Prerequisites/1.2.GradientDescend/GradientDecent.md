&emsp;
# Gradient Desend 梯度下降
- 为甚么神经网络能进行 "学习"？它的底层原理到底是什么？
- 梯度下降就是所谓的神经网络、或者说模型它能够 "学习" 的原因，随着越来越多的优化器（Optimizer）的出现，或许会有很多变种，但是，只要我们理解了梯度下降，就能够清楚的明白优化器（Optimizer）的工作原理。

>提醒
- 以上的这段话大家可以直接忽略。如果里面的一些词听着很难懂，请直接忽视。因为我们后面会挨个去分析并且在代码上实现，现在我们只需要专注一个点：梯度下降

&emsp;
# 1 Intro
我们看一个熟悉的普通的方程：
$$y = 3x^2 + 6x + 5$$

<div align=center>
    <image src="imgs/function.png" width=400/>
</div>

如果我们要对这个函数求最小值
>方法一
- 求出对称轴：$x=-b/2a=-1$
- 然后求极值：$y=3\times(-1)^2 + 6\times (-1)+5=2$

>方法二
- 求出导数：$\frac{dy}{dx}=6x+6$
- 令导数为 $0$，求出 $x$ 的值：$x=-1$（为了快速演示忽略 $2$ 阶导数）
- 求出极值


&emsp;
# 1 One-Dimentional GD
我们直接来看更新公式
$$x_{new} = x - lr*gradient$$

看它的几何意义：
<div align=center>
    <image src="imgs/gradient-descent.png" width=500>
</div>

>牛顿法
- 没有学习率（learning rate），更新的 $x$ 是与 y 轴的交点


&emsp;
# 2 Two-Dimentional GD
如果现在输入是一个 $2D$ 的矩阵，这里先提一个新名词：batch size，大家先对他有个印象：
$$X = \begin{bmatrix}
x_{1} & x_{2} & x_{3} & x_{4}
\end{bmatrix}$$

经过方程 $y = 3x^2 + 6x + 5$ 后的结果为：
$$Y = \begin{bmatrix}
y_{1} & y_{2} & y_{3} & y_{4}
\end{bmatrix}$$

我们写出对应的所有方程：
$$\begin{matrix}y_1 = 3x_1^2 + 6x_1 + 5 \\ \\
y_2 = 3x_2^2 + 6x_2 + 5 \\ \\
y_3 = 3x_3^2 + 6x_3 + 5 \\ \\
y_4 = 3x_4^2 + 6x_4 + 5
\end{matrix}$$ 


&emsp;
# 2 Multivariate GD




