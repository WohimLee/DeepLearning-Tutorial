&emsp;
# Vector Calculus
>矩阵相乘求导公式
$$\begin{align}
    C = A @ B，G = \frac{\nabla Loss}{\nabla C}\\
\end{align}$$

$$\begin{align}
    \nabla A = G @ B^T \\
    \nabla B = A^T @ G
\end{align}$$


>Notice:
- 下面的例子全部使用 PyTorch 框架使用的运算形式及运算顺序

&emsp;
# 1 Batch Size=1
$$X =\begin{bmatrix}
x_{1} & x_{2} & x_{3} & x_{4}
\end{bmatrix}$$

$$W =\begin{bmatrix}
w_{1} & w_{2} & w_{3} & w_{4}
\end{bmatrix}$$

$$y^{predict} = X @ W^T + b = [x_1, x_2, x_3, x_4] \begin{bmatrix}w_1\\w_2\\w_3\\w_4\end{bmatrix} + b$$

&emsp;
# 2 Batch Size=n
- batch_size=2
>Input
$$X = \begin{bmatrix}
x_{11} & x_{12} & x_{13} & x_{14} \\
x_{21} & x_{22} & x_{23} & x_{24}\\
\end{bmatrix}$$

>Ground True
$$Y^{target} =\begin{bmatrix}
y^{target}_1 \\ \\ y^{target}_2
\end{bmatrix}$$

>Weight
$$W =\begin{bmatrix}
w_{1} & w_{2} & w_{3} & w_{4}
\end{bmatrix}$$

>Forward
$$Y^{predict} = X @ W^T + b = \begin{bmatrix}
x_{11} & x_{12} & x_{13} & x_{14} \\
x_{21} & x_{22} & x_{23} & x_{24}\\
\end{bmatrix} 
\begin{bmatrix}w_1\\w_2\\w_3\\w_4\end{bmatrix} + b = \begin{bmatrix}z_1+b \\ z_2+b\end{bmatrix} = \begin{bmatrix}y^{predict}_1 \\ \\ y^{predict}_2\end{bmatrix}$$

>Update

$$W = W - lr*\nabla W$$
$$b = b - lr*\nabla b$$


&emsp;
## 2.1 Loss Function
$$E = Y^{predict} - Y^{target}$$

>Cost Function（标量）
$$Cost=\sum\frac{1}{2}(E)^2=\frac{1}{2}e_1^2 + \frac{1}{2}e_2^2$$

$$Cost=\sum\frac{1}{2}(Y^{predict} - Y^{target})^2=\frac{1}{2}(y^{predict}_1-y^{target}_1)^2 + \frac{1}{2}(y^{predict}_2-y^{target}_2)^2$$

>Loss Function（标量）
- 如果 batch size 过大，那么 $Cost$ 的值就会很大，求导的时候会导致梯度很大，产生 "梯度爆炸" 的问题，无法收敛
- 且后面对 $w$ 和 $b$ 的求导需要求均值
- 所以这里将 $Cost$ 除以 batch_size=n，这样 Loss 就会变小，同时后面对 $w$ 和 $b$ 的求导求均值就只需要 sum 就可以了
$$Loss=\frac{1}{n}Cost$$


&emsp;
## 2.2 $G$
- 链式求导

$$\frac{\partial L}{\partial C} = \frac{1}{n}$$
&emsp;

$$\frac{\partial L}{\partial e_1} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial e_1}=\frac{1}{n}\times2\times \frac{1}{2}e_1 = \frac{1}{n}e_1$$
&emsp;
$$\frac{\partial L}{\partial y^{predict}_1} =
\frac{\partial L}{\partial e_1}\frac{\partial e_1}{\partial y^{predict}_1} = \frac{1}{n}\times e_1\times 1 = 
\frac{1}{n}\Big(y^{predict}_1 - y^{target}_1\Big)$$

- 所以

$$G = \frac{\nabla L}{\nabla Y^{predict}}=\frac{1}{n}\Big(Y^{predict} - Y^{target}\Big)$$

&emsp;
## 2.2 Bias 的梯度
- 把最后的部分拿出来

$$Y^{predict} = \begin{bmatrix}z_1+b \\ z_2+b\end{bmatrix} = \begin{bmatrix}y^{predict}_1 \\ \\ y^{predict}_2\end{bmatrix}$$

-  对 $b$ 进行求导
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y^{predict}}\frac{\partial Y^{predict}}{\partial b}= \begin{bmatrix}
\frac{\partial L}{\partial y^{predict}_1}\frac{\partial y^{predict}_1} {\partial b} \\ \\
\frac{\partial L}{\partial y^{predict}_2}\frac{\partial y^{predict}_2} {\partial b}
\end{bmatrix}$$

- 但是我们只有一个 $b$，所以这个时候我们希望它取个平均值，由于前面我们已经除以 batch size，所以：
$$np.sum \Big(\begin{bmatrix}
\frac{\partial L}{\partial y^{predict}_1}\frac{\partial y^{predict}_1} {\partial b} \\ \\
\frac{\partial L}{\partial y^{predict}_2}\frac{\partial y^{predict}_2} {\partial b}
\end{bmatrix} \Big) = 
np.sum \Big(\begin{bmatrix}
\frac{1}{n}(y^{predict}_1 - y^{target}_1) \\ \\
\frac{1}{n}(y^{predict}_1 - y^{target}_1)
\end{bmatrix}\Big) = 
np.sum
\Big(\frac{1}{n}(Y^{predict} - Y^{target})\Big)
= np.sum(G)$$

- 所以
$$\nabla b = np.sum(G)$$

&emsp;
## 2.3 Weight 的梯度

$$\frac{\partial L}{\partial Z} =\begin{bmatrix}
\frac{\partial L}{\partial z_1} \\ \\
\frac{\partial L}{\partial z_2} 
\end{bmatrix}= \frac{1}{n}\begin{bmatrix}
(y^{predict}_1 - y^{target}_1) \\ \\
(y^{predict}_1 - y^{target}_1)
\end{bmatrix}= G$$

$$Z = \begin{bmatrix}z_1 \\ z_2\end{bmatrix} = \begin{bmatrix}
x_{11}w_1 + x_{12}w_2 + x_{13}w_3 + x_{14}w_4 \\
x_{21}w_1 + x_{22}w_2 + x_{23}w_3 + x_{24}w_4 
\end{bmatrix}$$

$$\frac{\partial L}{\partial W} =\frac{\partial L}{\partial Z}\frac{\partial Z}{\partial W} = \begin{bmatrix}
\frac{\partial L}{\partial z_1}\frac{\partial z_1}{\partial w_1} & 
\frac{\partial L}{\partial z_1}\frac{\partial z_1}{\partial w_2} & 
\frac{\partial L}{\partial z_1}\frac{\partial z_1}{\partial w_3} & 
\frac{\partial L}{\partial z_1}\frac{\partial z_1}{\partial w_4} \\ \\

\frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial w_1} & 
\frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial w_2} & 
\frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial w_3} & 
\frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial w_4} 
\end{bmatrix} = \begin{bmatrix}
\frac{\partial L}{\partial z_1}x_{11} & 
\frac{\partial L}{\partial z_1}x_{12} & 
\frac{\partial L}{\partial z_1}x_{13} & 
\frac{\partial L}{\partial z_1}x_{14} \\ \\

\frac{\partial L}{\partial z_2}x_{21} & 
\frac{\partial L}{\partial z_2}x_{22} & 
\frac{\partial L}{\partial z_2}x_{23} & 
\frac{\partial L}{\partial z_2}x_{24} 
\end{bmatrix}$$

- 我们希望每个分量取平均（前面已经除以 batch size，相加就可以了），即：
$$\begin{bmatrix}
\frac{\partial L}{\partial z_1}x_{11}+\frac{\partial L}{\partial z_2}x_{21}，
\frac{\partial L}{\partial z_1}x_{12}+\frac{\partial L}{\partial z_2}x_{22}，
\frac{\partial L}{\partial z_1}x_{13}+\frac{\partial L}{\partial z_2}x_{23}，
\frac{\partial L}{\partial z_1}x_{14}+\frac{\partial L}{\partial z_2}x_{24}
\end{bmatrix}$$

- 将上面写成矩阵相乘的形式：

$$ \begin{bmatrix}
x_{11} & x_{21} \\ x_{12} & x_{22} \\ x_{13} & x_{23} \\ x_{14} & x_{24} 
\end{bmatrix}
\begin{bmatrix}
    \frac{1}{n}\frac{\partial L}{\partial y^{predict}_1} \\ \\ \frac{1}{n}\frac{\partial L}{\partial y^{predict}_2}
\end{bmatrix}= X^T @ G $$

- 所以：
$$\nabla W = X^T @ G $$