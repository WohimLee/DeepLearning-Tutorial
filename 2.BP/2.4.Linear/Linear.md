&emsp;
# LINEAR
CLASS torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None) [[SOURCE]](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)

Applies a linear transformation to the incoming data: $y=x A^T+b$
$$Y = X@W^T + B$$
This module supports TensorFloat32.

On certain ROCm devices, when using float16 inputs this module will use different precision for backward.

&emsp;
## Parameters
- in_features (int) - size of each input sample
- out_features (int) - size of each output sample
- bias (bool) - If set to False, the layer will not learn an additive bias. Default: True

>Shape
- Input: $\left(*, H_{i n}\right)$ where $*$ means any number of dimensions including none and $H_{\text {in }}=$ in_features.
- Output: $\left(*, H_{\text {out }}\right)$ where all but the last dimension are the same shape as the input and $H_{\text {out }}=$ out_features.

## Variables
- weight (torch.Tensor) - the learnable weights of the module of shape `(out_features, in_features)`. The values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$, where $k=\frac{1}{\text{ in\_features}}$
- bias - the learnable bias of the module of shape `(out_features)`. If bias is True, the values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k=\frac{1}{\text {in\_features }}$


&emsp;
## forward 过程
<div align=center>
    <image src='imgs/linear.png' width=800>
</div>



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
## Implementation
>Code
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



