&emsp;
# Gradient Desend 梯度下降
- 为甚么神经网络能进行 "学习"？它的底层原理到底是什么？
- 梯度下降就是所谓的神经网络、或者说模型它能够 "学习" 的原因，随着越来越多的优化器（Optimizer）的出现，或许会有很多变种，但是，只要我们理解了梯度下降，就能够清楚的明白优化器（Optimizer）的工作原理。

>Notice
- 以上的这段话大家可以直接忽略。如果里面的一些词听着很难懂，请直接忽视。因为我们后面会挨个去分析并且在代码上实现，现在我们只需要专注一个点：梯度下降

&emsp;
# 1 Intro
我们看一个熟悉的普通的方程：
$$y = 3x^2 + 6x + 5 = 3(x+1)^2 + 4$$

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
# 2 Gradient Descent
我们直接来看更新公式
$$x_{new} = x - lr*gradient$$

看它的几何意义：
<div align=center>
    <image src="imgs/gradient-descent.png" width=800>
</div>

&emsp;
>牛顿法
- 没有学习率（learning rate），更新的 $x$ 是与 y 轴的交点


到这里我们已经知道了人工智能能够 "学习" 的秘密了，后面的无非就是更多的链式求导，更多的参数更新

&emsp;
# 3 Others
- B站: [梯度下降3D可视化讲解通俗易懂](https://www.bilibili.com/video/BV18P4y1j7uH/?spm_id_from=333.337.search-card.all.click&vd_source=ead820d10887c21595d014f264bcbb35)
- BGD，Batch Gradient Descent
- SGD，Stochastic Gradient Descent
- MBGD，Mini-Batch Gradient Descent
- Adam



