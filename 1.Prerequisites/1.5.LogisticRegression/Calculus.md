&emsp;
# Calculus

# 1 sigmoid 导数
- 知乎：[Sigmoid函数求导](https://zhuanlan.zhihu.com/p/452684332)
$$S(x) = sigmoid = \frac{1}{1+e^{-x}}$$

>求导
$$\quad\ \ \ S'(x) = -\frac{1}{(1+e^{-x})^2}\times (1+e^{-x})'$$
$$\qquad\quad   =-\frac{1}{(1+e^{-x})^2}\times (-e^{-x})$$
$$\quad\ \ =\frac{1}{1+e^{-x}} \times \frac{e^{-x}}{1+e^{-x}}$$
$$\qquad\quad\ =\frac{1}{1+e^{-x}} \times \frac{1+e^{-x}-1}{1+e^{-x}}$$
$$=S(x) (1 - S(x))$$


&emsp;
# 2 


