import os.path as path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import skimage.io as skio
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


root_data_dir = './Data/50'

ds_type = ['train','test','val']

class sim_ds(Dataset):

    def __init__(self,root_data_dir = root_data_dir, ds_type = 'train',ir_num = 16,ntiles=500,transforms=None):
        

        data_dir = path.join(root_data_dir,ds_type,'1')

        self.data_dir647 = path.join(data_dir,'647nm, Raw')
        self.data_dir750 = path.join(data_dir,'750nm, Raw')
        
        self.irs = [i for i in range(ir_num)]
        self.fovs = [i for i in range(ntiles)]

        self.gt = path.join(data_dir,'groundtruths')
        
        self.img_format = 'merFISH_{:02d}_{:03d}_01.TIFF'
        self.freq_format = 'frequency_{}.csv'
        self.loc_format = 'groundtruth_{}.csv'
        #get imagesize

        with open(path.join(data_dir,'config.yml')) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)

        self.img_size = self.params['simulation']['image_size']
        self.ds_size = np.minimum(self.params['simulation']['tile_count'],ntiles)
        self.emitters = self.params['simulation']['emitter_count']
        self.non_emitters = self.img_size**2 - self.params['simulation']['emitter_count']
        self.transforms = transforms

    def __len__(self):
        return self.ds_size
    def __getitem__(self,idx):
        
        # Create the targets
        bc_mat = torch.zeros((len(self.irs),self.img_size,self.img_size),requires_grad=False)
        loc_mat = torch.zeros((1,self.img_size,self.img_size),requires_grad=False)
        
        gt_file = pd.read_csv(path.join(self.gt,self.loc_format).format(idx+1))
        
        rows = gt_file['row'].tolist()
        
        columns = gt_file['column'].tolist()
        
        loc_mat[len(rows)*[0],
                rows,
                columns,
                ]=1.

        bcs = gt_file['barcode'].tolist()

        for ibc,bc in enumerate(bcs):
            bc = bc[1:-1]
            _bc = [int(s) for s in bc]
            
            c_list = [i*s for i,s in zip(range(len(self.irs)),_bc)]

            bc_mat[c_list,rows[ibc],columns[ibc]] = 1.
        
        #extract the data

        ir_idxs = len(self.irs)//2
        
        x647s = torch.zeros((ir_idxs,self.img_size,self.img_size),dtype=float)
        x750s = torch.zeros((ir_idxs,self.img_size,self.img_size),dtype=float)
        
        for i in range(1,ir_idxs+1):
            x647s[i-1,:,:] = torch.from_numpy(
                skio.imread(path.join(self.data_dir647,self.img_format).format(i,idx+1)).astype(float)[:,:,0]
                )
            x750s[i-1,:,:] = torch.from_numpy(skio.imread(path.join(self.data_dir750,self.img_format).format(i,idx+1)).astype(float)
            [:,:,0])
        

        if not (self.transforms is None):
            x647s = self.transforms(x647s)
            x750s = self.transforms(x750s)

        return x647s,x750s,loc_mat,bc_mat


class downsampleStage(nn.Module):
    def __init__(self,in_channels,out_channels=None):

        super(downsampleStage,self).__init__()
        if out_channels is None:
            out_channels = 2*in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        skip = x
        x = self.pool(x)

        return x,skip

class upsampleStage(nn.Module):
    def __init__(self,in_channels,out_channels = None,last_channel_out=None):

        super(upsampleStage,self).__init__()
        if out_channels is None:
            out_channels=int(in_channels//2)
        if last_channel_out is None:
            last_channel_out=out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=last_channel_out,
                                kernel_size=3,
                                padding = 1,
                                padding_mode='reflect')
        
        self.upsample = nn.Upsample(scale_factor=(2,2),mode='nearest')

    def forward(self,x,skip):
        x = F.interpolate(x,scale_factor=2)
        x = torch.concat((x,skip),dim=1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        return x
class bottleneck(nn.Module):
    def __init__(self,in_channels=96,out_channels=192):
        super(bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                padding_mode='reflect',
                                )
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                padding_mode='reflect',
                                )


    def forward(self,x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return x

class unet(nn.Module):
    def __init__(self,in_channels,last_out_channels=1):
        super(unet,self).__init__()
        
        self.add_module('down1',downsampleStage(in_channels,48))
        self.add_module('down2',downsampleStage(48,96))
        
        self.add_module('bottleneck',bottleneck(96,192))

        self.add_module('up1',upsampleStage(192+96,96))
        self.add_module('up2',upsampleStage(96+48,48,last_out_channels))
    
    def forward(self,x):
        x,skipx1 = self.down1(x)
        x,skipx2 = self.down2(x)
        x = self.bottleneck(x)
        x = self.up1(x,skipx2)
        x = self.up2(x,skipx1)
        return x

class multiStage(nn.Module):

    def __init__(self,in_channels=8,last_out_first_stage=1):
        super(multiStage,self).__init__()

        stage647 = unet(in_channels,last_out_first_stage)
        stage750 = unet(in_channels,last_out_first_stage)
        self.add_module('first_stage_647',stage647)
        self.add_module('first_stage_750',stage750)

        stage2 = unet(last_out_first_stage+last_out_first_stage,48)
        self.add_module('second_stage',stage2)

    def forward(self,x647,x750):
        x1 = self.first_stage_647(x647)
        x2 = self.first_stage_647(x750)

        x = torch.cat((x1,x2),dim=1)

        y = self.second_stage(x)

        return y

class anglerFISH(nn.Module):

    def __init__(self,in_channels=8,first_stage_channels=8):
        super(anglerFISH,self).__init__()
        self.add_module('twostage',multiStage(in_channels,first_stage_channels))
        
        self.add_module('head_p', nn.Sequential(nn.Conv2d(in_channels = 48,
                                                            out_channels = 48,
                                                            kernel_size=3,
                                                            padding=1,
                                                            padding_mode='reflect')
                                                            ,
                                            nn.Conv2d(in_channels = 48,
                                                            out_channels = 1,
                                                            kernel_size=3,
                                                            padding=1,
                                                            padding_mode='reflect')
                                                ))
                                            

        self.add_module('head_bc',nn.Sequential(nn.Conv2d(in_channels = 48,
                                                            out_channels = 48,
                                                            kernel_size=3,
                                                            padding=1,
                                                            padding_mode='reflect')
                                                            ,
                                            nn.Conv2d(in_channels = 48,
                                                            out_channels = 16,
                                                            kernel_size=3,
                                                            padding=1,
                                                            padding_mode='reflect')
                                                ))
    def forward(self,x647,x750):

        x = self.twostage(x647,x750)
        out_p = torch.sigmoid(self.head_p(x))
        out_bc = torch.sigmoid(self.head_bc(x))

        return out_p,out_bc

            
def prob_loss(pred,target):

    return F.binary_cross_entropy(pred,target,reduction='mean')


def valid_barcode_loss(pred:torch.tensor,target:torch.tensor,weight=1,reduction=torch.mean):
    limit = torch.tensor(16*[-100])
    _t = weight*target * torch.maximum(torch.log(pred),limit) + (1-target)*torch.maximum(torch.log(1-pred),limit)
    return reduction(-_t)

def barcode_loss(pred:torch.tensor,target:torch.tensor,weight=1,valid_weight=12/4):

    target = torch.permute(target,(0,2,3,1))
    target = target.reshape(-1,16)
    

    pred = torch.permute(pred,(0,2,3,1))
    pred = pred.reshape(-1,16)

    flags = target.sum(dim=1)>0
    loss = torch.tensor(0.)
    for i in range(pred.shape[0]):
 
        loss += weight*flags[i]*valid_barcode_loss(pred[i],target[i],valid_weight) + (1-1*flags[i])*F.binary_cross_entropy(pred[i],target[i],reduction='mean')
    loss /=pred.shape[0] 
    return loss

def train(model,dataset,dataloader,optimizer_fn,epochs=2,print_rate=10):
    loss_history =list()
    for e in range(epochs):
        count = 0
        epoch_loss= 0
        for x647s,x750s,loc_mat,bc_mat in tqdm(dataloader):
            count+=1
            optimizer_fn.zero_grad()
            out_p,out_bc = model(x647s.float(),x750s.float())
            if torch.any(torch.isnan(out_p)):
                print('prediction is nan')
                if torch.any(torch.isnan(x647s)) or torch.any(torch.isnan(x750s)):
                    print('input is nan')
            loss = barcode_loss(out_bc,bc_mat,dataset.non_emitters/dataset.emitters,12/4) + prob_loss(out_p,loc_mat)
            #print(loss)
            loss.backward()
            optimizer_fn.step()
            epoch_loss +=loss.item()
            if count % print_rate==0:
                denom = count*x647s.shape[0]
                print('Epoch: {} | Avg Running Loss per sample: {}'.format(e,epoch_loss/denom))

        denom = count*x647s.shape[0]
        loss_history.append(epoch_loss/denom)


    return loss_history





if __name__=="__main__":

    model = anglerFISH(8,8)

    optimizer_fn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_ds = sim_ds(ds_type='train',ntiles=50)
    test_ds = sim_ds(ds_type='test',ntiles=50)
    val_ds = sim_ds(ds_type='val',ntiles=50)

    train_dl = DataLoader(train_ds,batch_size=2,shuffle=True)
    test_dl = DataLoader(test_ds,batch_size=1,shuffle=True)
    val_dl = DataLoader(val_ds,batch_size=1,shuffle=True)

    train(model=model,dataset=train_ds,dataloader=train_dl,optimizer_fn=optimizer_fn)