
from .nn import Module, ModuleList, Linear, ReLU, Dropout
        
            
class Model(Module):
    def __init__(self, num_feature, num_hidden, num_classes):
        super().__init__("Model")
        self.backbone = ModuleList(
            Linear(num_feature, num_hidden),
            ReLU(),
            Dropout(),
            Linear(num_hidden, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def backward(self, G):
        return self.backbone.backward(G)
  