import pytorch_lightning as pl
from voc_dataset import PascalVOCDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as ET
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import timm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from model import MLCNNet


#please change the root_dir to the path of VOC2012 folder in PASCAL-VOC dataset

root_dir = '/home/arka/Pascal_voc/VOCdevkit/VOC2012'
transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


#creating dataset 
train_dataset = PascalVOCDataset(root_dir, image_set='train', transform=transforms)
test_val_dataset= PascalVOCDataset(root_dir, image_set='val', transform= transforms)

val_size= int(0.5*len(test_val_dataset))
test_size= len(test_val_dataset)- val_size

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
gen= torch.Generator().manual_seed(123)
val_dataset, test_dataset= torch.utils.data.random_split( test_val_dataset, [val_size, test_size], generator= torch.Generator().manual_seed(123))

train_loader= DataLoader(dataset= train_dataset, batch_size= 32, shuffle=True)
val_loader= DataLoader(dataset= val_dataset, batch_size=32, shuffle=False)
test_loader= DataLoader(dataset= test_dataset, batch_size=1, shuffle=False)

backbone = timm.create_model("resnetv2_50",pretrained=False,num_classes = 0)
model= MLCNNet(backbone, 20)

class VOCNet(pl.LightningModule):
     def __init__(self, model):
          super().__init__()
          self.model= model
    

     def training_step(self, batch, batch_idx):
          x,y = batch
          outputs = self.model(x)
          loss= F.binary_cross_entropy_with_logits(outputs, y)
          self.log("train_loss", loss.item()/len(y))
          return loss
     
     def validation_step(self,batch,batch_idx):
            x,y = batch
            outputs = self.model(x)
            loss = F.binary_cross_entropy_with_logits(outputs,y)
            self.log("val_loss",loss.item() / len(y))

     def predict_step(self,batch,batch_idx,dataloader_idx=0):
            x,y = batch
            preds  = self.model(x)
            return preds,y
     
     def configure_optimizers(self):
            optim = torch.optim.AdamW(self.parameters(),lr = 8e-6)
            return optim
     

pl_model= VOCNet(model)
#tb_logger= TensorBoardLogger(save_dir='/home/arka/Pascal_voc/tb_logs', name='experiment' )


"""change the DIRPATH to the directory where you want to save the trained model """
checkpoint_callback= ModelCheckpoint(
      monitor= 'val_loss',
      mode= 'min',
      save_top_k= 1,
      dirpath= '/home/arka/Pascal_voc',
      filename= 'best_model_seed_123' 

)

trainer= pl.Trainer(callbacks=[checkpoint_callback],  max_epochs=100, log_every_n_steps=5,  accelerator='gpu')
trainer.fit(pl_model, train_loader, val_loader)

















