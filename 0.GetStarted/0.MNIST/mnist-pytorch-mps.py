import torch 
import datetime 
import os,sys,time
import torchvision 

import numpy as np
import pandas as pd
import torch.nn.functional as F 

from torch import nn 
from tqdm import tqdm 
from copy import deepcopy
from torchmetrics import Accuracy
from torchvision import transforms 


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")
    
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 64,out_channels = 512,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,10)
        ]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


# 评估指标
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0] 
        self.correct += m 
        self.total += n
        
        return m/n

    def compute(self):
        return self.correct.float() / self.total 
    
    def reset(self):
        self.correct -= self.correct
        self.total -= self.total
        
          
def train():
    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------  
        net.train()
        
        total_loss, step = 0, 0
        
        loop = tqdm(enumerate(trainloader), total =len(trainloader),ncols=100)
        train_metrics_dict = deepcopy(metrics_dict) 
        
        for i, batch in loop: 
            
            features,labels = batch
            
            # =========================移动数据到mps上==============================
            features = features.to(device)
            labels = labels.to(device)
            # ====================================================================
            
            #forward
            preds = net(features)
            loss = loss_fn(preds,labels)
            
            #backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
            #metrics
            step_metrics = {"train_"+name:metric_fn(preds, labels).item() 
                            for name,metric_fn in train_metrics_dict.items()}
            
            step_log = dict({"train_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            
            step+=1
            if i!=len(trainloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                                for name,metric_fn in train_metrics_dict.items()}
                epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in train_metrics_dict.items():
                    metric_fn.reset()
                    
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]
            

        # 2，validate -------------------------------------------------
        net.eval()
        
        total_loss,step = 0,0
        loop = tqdm(enumerate(validloader), total =len(validloader),ncols=100)
        
        val_metrics_dict = deepcopy(metrics_dict) 
        
        with torch.no_grad():
            for i, batch in loop: 

                features,labels = batch
                
                # =========================移动数据到mps上==============================
                features = features.to(device)
                labels = labels.to(device)
                # ====================================================================
                
                #forward
                preds = net(features)
                loss = loss_fn(preds,labels)

                #metrics
                step_metrics = {"val_"+name:metric_fn(preds, labels).item() 
                                for name,metric_fn in val_metrics_dict.items()}

                step_log = dict({"val_loss":loss.item()},**step_metrics)

                total_loss += loss.item()
                step+=1
                if i!=len(validloader)-1:
                    loop.set_postfix(**step_log)
                else:
                    epoch_loss = (total_loss/step)
                    epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                    for name,metric_fn in val_metrics_dict.items()}
                    epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                    loop.set_postfix(**epoch_log)

                    for name,metric_fn in val_metrics_dict.items():
                        metric_fn.reset()
                        
        epoch_log["epoch"] = epoch           
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:
            torch.save(net.state_dict(),ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                arr_scores[best_score_idx]),file=sys.stderr)
        if len(arr_scores)-best_score_idx>patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor,patience),file=sys.stderr)
            break 
        net.load_state_dict(torch.load(ckpt_path))
        
    dfhistory = pd.DataFrame(history)

if __name__ == '__main__':
#================================================================================
# 一，准备数据
#================================================================================

    transform = transforms.Compose([transforms.ToTensor()])

    trainset  = torchvision.datasets.MNIST(root="mnist/",train=True, download=True,transform=transform)
    validset  = torchvision.datasets.MNIST(root="mnist/",train=False,download=True,transform=transform)

    trainloader  =  torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=2)
    validloader  =  torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=2)
    
#================================================================================
# 二，定义模型
#================================================================================

    net = Model()
    print(net)
    
#================================================================================
# 三，训练模型
#================================================================================   
    loss_fn      = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(net.parameters(),lr = 0.01)   
    metrics_dict = nn.ModuleDict({"acc":Accuracy()})


    # =========================移动模型到mps上==============================
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    loss_fn.to(device)
    metrics_dict.to(device)
    # ====================================================================

    epochs    = 20 
    ckpt_path = 'checkpoint.pt'

    #early_stopping相关设置
    monitor   = "val_acc"
    patience  = 5
    mode      = "max"

    history = {}
    train()