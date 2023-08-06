&emsp;
# Jacobian Matrix 雅可比矩阵

>$\pmb{X}@\pmb{Y}$ 矩阵运算
$$\pmb{X} = \begin{bmatrix} 
x_0 & x_1 \\ x_2 & x_3 \\ x_4 & x_5 
\end{bmatrix}， shape=(3, 2)$$

$$\pmb{Y} = \begin{bmatrix} 
y_0 & y_1 \\ y_2 & y_3 
\end{bmatrix}，shape=(2, 2)$$


$$\pmb{Z}=\pmb{X} @\pmb{Y}=(3,2)@(2,2)=(3,2)=\begin{bmatrix}z_0 & z_1 \\ z_2 & z_3 \\ z_4 & z_5 
\end{bmatrix}$$


$z_0 = [x_0， x_1] \ [y_0， y_2]^T = x_0y_0 + x_1y_2 
= x_0\cdot y_0 + 0\cdot y_1 + x_1\cdot y_2 + 0\cdot y_3$

$z_1 = [x_0， x_1] \ [y_1， y_3]^T= x_0y_1 + x_1y_3
= 0\cdot y_0 + x_0\cdot y_1 + 0\cdot y_2 + x_1\cdot y_3$

$z_2 = [x_2， x_3] \ [y_0， y_2]^T= x_2y_0 + x_3y_2
= x_2\cdot y_0 + 0\cdot y_1 + x_3\cdot y_2 + 0\cdot y_3$

$z_3 = [x_2， x_3] \ [y_1， y_3]^T= x_2y_1 + x_3y_3
= 0\cdot y_0 + x_2\cdot y_1 + 0\cdot y_2 + x_3\cdot y_3$

$z_4 = [x_4， x_5] \ [y_0， y_2]^T= x_4y_0 + x_5y_2
= x_4\cdot y_0 + 0\cdot y_1 + x_5\cdot y_2 + 0\cdot y_3$

$z_5 = [x_4， x_5] \ [y_1， y_3]^T= x_4y_1 + x_5y_3
= 0\cdot y_0 + x_4\cdot y_1 + 0\cdot y_2 + x_5\cdot y_3$

&emsp;
>$\pmb{X}@\pmb{Y}$ 变换

$$\pmb{X} = \begin{bmatrix} 
x_0 & 0 & x_1 & 0 \\
0 & x_0 & 0 & x_1 \\
x_2 & 0 & x_3 & 0 \\
0 & x_2 & 0 & x_3 \\
x_4 & 0 & x_5 & 0 \\
0 & x_4 & 0 & x_5 \\
\end{bmatrix}，shape=(6,4)$$

$$\pmb{Y} = \begin{bmatrix} 
y_0 & y_1 & y_2 & y_3
\end{bmatrix}^T，shape=(4,1)$$


$$\pmb{Z}=\pmb{X} @\pmb{Y}=(6,4)@(4,1)=(6,1)=\begin{bmatrix}z_0 , z_1 , z_2 , z_3 , z_4 , z_5\end{bmatrix}^T$$



&emsp;
>$\nabla \pmb{Y}$

$$\nabla \pmb{Y} = \pmb{X}^T @ \pmb{G}$$