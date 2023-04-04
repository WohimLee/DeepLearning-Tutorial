&emsp;
# Confusion Matrix 混淆矩阵


混淆矩阵有两个维度：
- Actual Values: 真实值
- Predicted Values: 模型预测的结果

<div align=center>
    <image src="imgs/confusionMatrix.png" width=400>
</div>

&emsp;
>Example-二分类
- 我们考虑一个二分类的问题：给定图片是不是老虎？
<div align=center>
    <image src="imgs/conMat_example.png" width=>
</div>


&emsp;
>Example-多分类
- 思考多分类的问题：如何判断 TN、FN？
<div align=center>
    <image src="imgs/conMat_example2.png" width=>
</div>

&emsp;
# Accuracy（ACC）
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$


&emsp;
# Precision（PPV）
- 表格行方向的统计
$$Precision = \frac{TP}{TP+FP}$$


&emsp;
# Recall（TPR）
- Sensitivity
- 表格列方向的统计
  
$$Recall = \frac{TP}{TP + FN}$$



&emsp;
# F score
- $Precision$ 与 $Recall$ 的调和平均, $Recall$ 的重要性是 $Precision$ 的 $\beta$ 倍
- $\beta=1$: 称为 $F1\ score$
$$F = \frac{(1 + \beta^2)\times Precision \times Recall}{\beta^2 \times Precision + Recall}$$


&emsp;
# ROC 
- TPR 和 FPR 的曲线

&emsp;
# AUC（Area Under Curve）


&emsp;
# PR 
- Precision 和 Recall 的曲线