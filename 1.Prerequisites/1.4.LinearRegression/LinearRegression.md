&emsp;
# Linear Regression


# 1 One-Dimension

&emsp;
# 2 Multivariate 
- 神经网络最后的输出：(batch_size, predict)
  - batch size: 多少个输入（如图片）
  - predict: 每张图片的输出

- 接下来我们将仿照神经网络计算 Loss 的过程组织运算

&emsp;
## 2.1 Batch Size 为 1
- 假设数据是 1 张图片，将它输入网络，输出了 4 个值：

<div align=center>
    <image src='imgs/batchsize1.png' width=500/>
</div>

- 经过方程 $L = (X - Y)^2$ 后的结果为：
    $$L = [L_1，L_2，L_3，L_4]=\begin{bmatrix}
    (x_1 - y_{1})^2， (x_2 - y_{2})^2 ， (x_3 - y_{3})^2 ，(x_4 - y_{4})^2
    \end{bmatrix}$$

- 将他们加和并求均值：

    $$loss = (L_1 + L_2 + L_3 +L_4 )/4$$

- 要使得 loss 最小，问题转变成了 loss 对每个分量进行梯度下降了，也就是说我们要用这个公式
    $$x_{new} = x - lr*gradient$$
- 将每一个 $X$ 的分量更新到与 $Y$ 的每一个分量接近，其中的关键就是 $gradient$， 也就是要求出 $loss$ 对 $X$ 的导数: $\frac{d\ loss}{d\ X}$，用来批量更新 $x$
    $$X_{new} = X - lr*\frac{d\ loss}{d\ X}$$

- 求 $\frac{d\ loss}{d\ X}$

    $$\frac{d\ loss}{d\ X} = \frac{d\ loss}{d\ L}\frac{d\ L}{d\ X} =
    [\frac{\partial loss}{\partial L_1} \frac{\partial L_1}{\partial x_1}，
    \frac{\partial loss}{\partial L_2} \frac{\partial L_2}{\partial x_2}，
    \frac{\partial loss}{\partial L_3} \frac{\partial L_3}{\partial x_3}，
    \frac{\partial loss}{\partial L_4} \frac{\partial L_4}{\partial x_4}]
    $$

    $$= [\frac{1}{4} \times 2x_1，\frac{1}{4} \times 2x_2，\frac{1}{4} \times 2x_3，\frac{1}{4} \times 2x_4] $$
    $$= \frac{1}{2}X$$

&emsp;
## 2.2 Batch Size 为 n
- 假设数据是 $batch\ size=2$ 张图片，将它输入网络，输出了 4 个值：
  
<div align=center>
    <image src='imgs/batchsize2.png' width=600/>
</div>

- 经过方程 $L = (X - Y)^2$ 后的结果为：
    $$L = \begin{bmatrix}L_{11}，L_{12}，L_{13}，L_{14} \\
    L_{21}，L_{22}，L_{23}，L_{24}
    \end{bmatrix}$$
    $$=\begin{bmatrix}
    (x_{11} - y_{11})^2， (x_{12} - y_{12})^2 ， (x_{13} - y_{13})^2 ，(x_{14} - y_{14})^2 \\
    (x_{21} - y_{21})^2， (x_{22} - y_{22})^2 ， (x_{23} - y_{23})^2 ，(x_{24} - y_{24})^2
    \end{bmatrix}$$

- 将他们加和并求均值：

    $$loss = (L_{11} + L_{12} + L_{13} +L_{14} + L_{21} + L_{22} + L_{23} +L_{24} )/8$$

- 要使得 loss 最小，问题转变成了 loss 对每个分量进行梯度下降了，也就是说我们要用这个公式
    $$x_{new} = x - lr*gradient$$
- 将每一个 $X$ 的分量更新到与 $Y$ 的每一个分量接近，其中的关键就是 $gradient$， 也就是要求出 $loss$ 对 $X$ 的导数: $\frac{d\ loss}{d\ X}$，用来批量更新 $x$
    $$X_{new} = X - lr*\frac{d\ loss}{d\ X}$$

- 求 $\frac{d\ loss}{d\ X}$

    $$\frac{d\ loss}{d\ X} = \frac{d\ loss}{d\ L}\frac{d\ L}{d\ X} $$
    $$=\begin{bmatrix}
    \frac{\partial loss}{\partial L_{11}} \frac{\partial L_{11}}{\partial x_{11}}，
    \frac{\partial loss}{\partial L_{12}} \frac{\partial L_{12}}{\partial x_{12}}，
    \frac{\partial loss}{\partial L_{13}} \frac{\partial L_{13}}{\partial x_{13}}，
    \frac{\partial loss}{\partial L_{14}} \frac{\partial L_{14}}{\partial x_{14}} \\ \\
    \frac{\partial loss}{\partial L_{21}} \frac{\partial L_{21}}{\partial x_{21}}，
    \frac{\partial loss}{\partial L_{22}} \frac{\partial L_{22}}{\partial x_{22}}，
    \frac{\partial loss}{\partial L_{23}} \frac{\partial L_{23}}{\partial x_{23}}，
    \frac{\partial loss}{\partial L_{24}} \frac{\partial L_{24}}{\partial x_{24}}
    \end{bmatrix}$$

    $$= \begin{bmatrix}
    \frac{1}{8} \times 2x_{11}，\frac{1}{8} \times 2x_{12}，\frac{1}{8} \times 2x_{13}，\frac{1}{8} \times 2x_{14} \\ \\
    \frac{1}{8} \times 2x_{21}，\frac{1}{8} \times 2x_{22}，\frac{1}{8} \times 2x_{23}，\frac{1}{8} \times 2x_{24}
    \end{bmatrix}$$
    $$= \frac{1}{4}X$$

