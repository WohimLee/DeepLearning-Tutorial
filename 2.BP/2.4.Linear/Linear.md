&emsp;
# Linear
- input: X.shape = (batch_size, in_features)
- output: Y.shape = (batch_size, out_features)
    $$Y = X@W^T + B$$
    <div align=center>
        <image src='imgs/linear.png' width=800>
    </div>
- in_features: 输入的特征数
- out_features: 输出特征数
- weight: 权重，(out_features, in_features)
- bias: 噪声/偏置，(out_features,)

&emsp;
## forward 过程




&emsp;
## backward 过程

>矩阵相乘求导公式


$$\begin{align}
    Y = X @ W^T + B，G = \frac{\nabla Loss}{\nabla Y}\\
\end{align}$$

$$\nabla X = G @ W$$ 

>weight 和 bias 的梯度
$$\nabla W = (X^T @ G)^T = G^T @ X\\
 \nabla B = G$$


&emsp;
## 代码实现

```py
class Linear(Module):
    def __init__(self, in_fearures, out_features):
        self.in_features  = in_fearures
        self.out_features = out_features
        self.weight = Parameter(np.zeros(out_features, in_fearures))
        self.bias   = Parameter(np.zeros(out_features))
        
    def forward(self, x):
        self.__x = x
        return x@self.weight.T + self.bias
    # Y = X@W.T + B
    def backward(self, G):
        # G.T @ X
        self.weight.grad = G.T @ self.__x
        # G
        self.bias.grad   = G
        # G @ W
        return G @ self.weight.data
```



