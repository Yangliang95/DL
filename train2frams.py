#!/usr/bin/env python
# coding: utf-8
"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
Author：Team Li

"""
"vehicle speed estimation with two frams"
import tqdm
import os
from collections import OrderedDict
from pathlib2 import Path
from PIL import Image
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split, Subset
from torchvision import transforms
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)

# check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('############# USE', device, ' ###############')

wandb.login()

p = {
    'project_name': 'vehicle-speed-estimation',
    'batch_size': 32,
    'w': 240,  # 240 480
    'h': 320,  # 320 640
    'model': 'efficientnet-b0',
    'split': 0.2,
    'mean': [0.23578693, 0.27709611, 0.32325169],
    'std': [0.31154287, 0.34198315, 0.36109759],
    'divide_y': 10
}

IMAGES = r'H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\train_img'
LABELS = r'H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\train.txt'

transform = transforms.Compose([
    transforms.Resize((p['w'], p['h'])),
    transforms.ToTensor(),
    transforms.Normalize(p['mean'],p['std'])
])


################# prepare data ###############
class DS(Dataset):
    def __init__(self, images, labels):
        self.images = Path(images)
        self.labels = open(labels).readlines()
        self.n_images = len(list(self.images.glob('*.jpg')))

    def __len__(self): return self.n_images

    def __getitem__(self, idx):
        'Returns a random batch of len `seq`'
        if idx != 0: idx -= 1
        f1 = f'{self.images}\\frame_{str(idx)}.jpg'
        f2 = f'{self.images}\\frame_{str(idx + 1)}.jpg'
        image1 = Image.open(f1)
        image2 = Image.open(f2)
        img1 = transform(image1)
        img2 = transform(image2)
        x = torch.cat((img1, img2))
        y = float(self.labels[idx + 1].split()[0])
        return x, torch.Tensor([y])

ds = DS(IMAGES, LABELS)

train_idx = int(len(ds) * (1 - p['split']))
valid_idx = int(len(ds) * p['split'])
train_ds = Subset(ds, list(range(train_idx)))
valid_ds = Subset(ds, list(range(train_idx, train_idx + valid_idx)))

# train_ds, valid_ds = random_split(ds, [train_idx, valid_idx])
print('len of train_ds: ', len(train_ds), 'len of valid_ds: ', len(valid_ds))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=p['batch_size'], num_workers=0,
                      pin_memory=True)  # num_workers=6
valid_dl = DataLoader(valid_ds, shuffle=True, batch_size=p['batch_size'], num_workers=0,
                      pin_memory=True)  # num_workers=6
test_dl = DataLoader(ds, shuffle=False, batch_size=p['batch_size'], num_workers=0,
                      pin_memory=True)  # num_workers=6

################# Design Model ###############

class Model(pl.LightningModule):
    def __init__(self, p):
        super().__init__()
        self.hparams = p
        self.save_hyperparameters()
        self.en = EfficientNet.from_pretrained(p['model'], in_channels=6, num_classes=1)

    def forward(self, x):
        return self.en(x)

    def step(self, batch, batch_idx):
        x, y = batch
        y /= p['divide_y']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        abs_loss = torch.abs(y_hat - y).mean().sum()
        return OrderedDict({
            'loss': loss,
            'abs_loss': abs_loss
        })

    def training_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('train_batch_loss', out['loss'])
        self.log('train_batch_abs_loss', out['abs_loss'])
        return out

    def training_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs]).float().mean()
        abs_loss = torch.stack([output['abs_loss'] for output in outputs]).float().mean()
        self.log('train_loss', loss)
        self.log('train_abs_loss', abs_loss)

    def validation_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('val_batch_loss', out['loss'])
        self.log('val_batch_abs_loss', out['abs_loss'])
        return out

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs]).float().mean()
        abs_loss = torch.stack([output['abs_loss'] for output in outputs]).float().mean()
        self.log('val_abs_loss', abs_loss)
        self.log('val_loss', loss)

    def configure_optimizers(self): return optim.Adam(self.parameters())


################# logger ###############

wandb_logger = WandbLogger(project="vehicle-speed-estimation")

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    filename='cnn-{epoch:02d}-{val_loss:.4f}',
    save_top_k=2,
    mode='min'
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

trainer = pl.Trainer(gpus=1,
                     fast_dev_run=False,
                     log_every_n_steps=10,
                     logger=wandb_logger,
                     # overfit_batches=5,
                     deterministic=True,
                     callbacks=[checkpoint, early_stopping]
                     )


m = Model(p)
#trainer.fit(m, train_dl, valid_dl)
wandb.finish()

def text_save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.

  file = open(filename, 'a')

  for i in range(len(data)):
    s = str(data[i][0]) + ',' + str(data[i][1]) + '\n'
    # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符

    file.write(s)
  file.close()
  print("保存文件成功")

def inference(f1, f2):

    ckp = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\vehicle-speed-estimation\xqi52duk\checkpoints\cnn-epoch=04-val_loss=0.0681.ckpt'

    m = Model(p)
    m = m.load_from_checkpoint(ckp)
    m.eval() # 防止改变权值

    image1 = Image.open(f1)
    image2 = Image.open(f2)
    img1 = transform(image1)
    img2 = transform(image2)
    x = torch.cat((img1, img2)).unsqueeze(0)
    return m(x).item()

def test():
    res = []
    ckp = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\vehicle-speed-estimation\xqi52duk\checkpoints\cnn-epoch=04-val_loss=0.0681.ckpt'
    m = Model(p)
    m = m.load_from_checkpoint(ckp)
    m.eval()  # 防止改变权值
    for x,y in test_dl:
        print(m(x).shape,y.shape)
        # print(list(map(lambda x,y:[x.item(),y.item()],m(x),y)))
        res+=list(map(lambda x,y:[x.item()*10,y.item()],m(x),y))
    text_save('./res.txt',res)



if __name__ == "__main__":

    f1=r'H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\test_img\frame_100.jpg'
    f2=r'H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\test_img\frame_101.jpg'
    #print(inference(f1, f2))
    test()
    # text_save('./res.txt', [[1,2],[3,4]])