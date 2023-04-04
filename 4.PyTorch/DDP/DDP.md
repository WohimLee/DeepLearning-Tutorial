&emsp;
# DDP（Distributed Data Parallel）
- 官方: [torchrun (Elastic Launch)](https://pytorch.org/docs/stable/elastic/run.html)
- 官方：[Multi GPU training with DDP](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#multi-gpu-training-with-ddp)
- 知乎：[Pytorch DDP分布式训练介绍](https://zhuanlan.zhihu.com/p/453798093)

&emsp;
# 1 基本概念
>基本概念
- group: 进程组，一般就需要一个默认的
- world size: 所有的进程数量
- rank: 全局的进程id
- local rank：某个节点上的进程id
- local_word_size: 某个节点上的进程数 (相对比较少见)

一个进程可以对应若干个GPU。 所以world_size 并不是等于所有的GPU数量，而人为设定的

- rank的取值范围：[0, W-1]，rank=0的进程为主进程，会负责一些同步分发的工作
- local_rank的取值：[0, L-1]

&emsp;
>Packages
```py
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
```

&emsp;
# 2 环境设置

```py
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
```

&emsp;
# 3 DDP 模型训练




