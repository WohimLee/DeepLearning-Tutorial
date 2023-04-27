&emsp; 
# Logistic Regression 逻辑回归
- 知乎：[【机器学习】逻辑回归（非常详细）](https://zhuanlan.zhihu.com/p/74874291)

>Logistic 回归的本质
- 人话版：线性回归 + sigmoid 函数
- 做作版：假设数据服从 Logistic 分布，然后使用极大似然估计做参数的估计。
>Notice
- 虽然逻辑回归名称带 "回归"，但它是用来解决分类问题的，也就是说它的输出是离散的

以下使用
&emsp;
# 1 One-Dimension




&emsp;
# 2 Muti-Dimension


&emsp;
# 1 人话版


&emsp;
## 1.2 线性回归转分类
思考一个问题，对于左图，我们能不能将它们分为两类？
- $y = xw + b$
<table><tr>
    <td><img src="imgs/classify-raw.png" border=0></td>
    <td><img src="imgs/classify-done.png" border=0></td>
</tr></table>



&emsp;
## 1.3 sigmoid 函数


&emsp;
# 2 做作版
Logistic 分布是一种连续型的概率分布
- $\mu$: 位置参数
- $\gamma$: 形状参数
>概率分布函数
$$F(x) = P(X \leq x) = \frac{1}{1+e^{-(x-\mu)/\gamma}}$$

>概率密度函数

$$f(x) = F'(X \leq x)= \frac{e^{-(x-\mu)/ \gamma}}{\gamma(1+e^{-(x-\mu)/ \gamma})^2}$$
