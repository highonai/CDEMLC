
import torch
import torch.nn as nn




class MLCNNet(nn.Module):
    
        def __init__(self,backbone,n_classes):
            super(MLCNNet,self).__init__();
            self.model = backbone
            self.classifier = nn.Sequential(nn.Linear(2048,256),
                                            nn.ReLU(),
                                            nn.Linear(256,n_classes))
        def forward(self,x):
            x = self.model(x)
            x = self.classifier(x)
            return x










     
     





