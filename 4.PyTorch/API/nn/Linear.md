&emsp;
# Linear


In PyTorch, the nn.Linear layer performs a matrix multiplication between the input tensor x and the weight matrix w, and adds the bias term if bias=True.

More specifically, if x is a tensor of shape (batch_size, in_features) and w is a weight matrix of shape (out_features, in_features), then the output of the nn.Linear layer can be computed as:


```py
output = x @ w.t() + b
```

where @ represents matrix multiplication, w.t() is the transpose of the weight matrix w, and b is the bias vector of shape (out_features,).

Note that x @ w.t() results in a tensor of shape (batch_size, out_features), and adding the bias term b broadcasts b along the first dimension of the output tensor.

So, to summarize, nn.Linear applies a linear transformation to the input tensor x by computing x @ w.t() + b, where w is the weight matrix and b is the bias vector, and returns the output tensor of the same shape as (batch_size, out_features).

