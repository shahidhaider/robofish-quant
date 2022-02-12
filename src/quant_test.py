from model import anglerFISH
from dataset import sim_ds
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import metrics as m
import torch.quantization as quant

import os

# set the qconfig for PTQ
qconfig = quant.get_default_qconfig('qnnpack')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'qnnpack'

model = anglerFISH()

dict_path = './best_state_dicts/feb_11'
dict_name = 'state_dict.pth'

statedict = path.join(dict_path,dict_name)
img_out_dir = path.join(dict_path,'static_quant_test_images')
datapath = './Data/50/'

tf = transforms.Compose([
    transforms.Normalize((0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5),(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)),
    transforms.ConvertImageDtype(torch.float)])

test_ds = sim_ds(root_data_dir = datapath,ds_type='test',ntiles=50,transforms=tf)

test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)

model.load_state_dict(torch.load(statedict))

model.eval()

# Dyanmic Quant
quant_robofish = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d}, dtype=torch.qint8
)


# print('Here is the floating point version of this module:')
# print(model)
# print('')
# print('and now the quantized version:')
# print(quant_robofish)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(model,"fp32")
q=print_size_of_model(quant_robofish,"int8")
print("{0:.2f} times smaller".format(f/q))

## Static Quantization

class robofish_readyforquant(nn.Module):
    def __init__(self):
        super(robofish_readyforquant,self).__init__()
        self.quant647 = quant.QuantStub()
        self.quant750 = quant.QuantStub()
        self.main = model
        self.dequant_p = quant.DeQuantStub()
        self.dequant_bc = quant.DeQuantStub()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x647,x750):
        q647 = self.quant647(x647)
        q750 = self.quant750(x750)
        out_p,out_bc = self.main(q647,q750)
        out_p,out_bc= self.dequant_p(out_p),self.dequant_bc(out_bc)
        out_p = self.sigmoid(out_p)
        out_bc = self.sigmoid(out_bc)
        return out_p,out_bc


model_quant_ready = robofish_readyforquant()
# set the qconfig for PTQ
model_quant_ready.qconfig = quant.get_default_qconfig('qnnpack')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'qnnpack'



model_fused = model_quant_ready#torch.quantization.fuse_modules(model,[]) #[['conv1','bn1', 'elu']])

model_prepared = torch.quantization.prepare(model_quant_ready)

x647s,x750s,loc_mat,bc_mat = next(iter(test_dl))
model_prepared(x647s.float(),x750s.float())
model_int8 = torch.quantization.convert(model_prepared)
# compare the sizes
f=print_size_of_model(model,"fp32")
q=print_size_of_model(model_int8,"int8")
print("{0:.2f} times smaller".format(f/q))

# print('Here is the floating point version of this module:')
# print(model)
# print('')
# print('and now the quantized version:')
# print(model_int8)

import time

start = time.time()
model(x647s.float(),x750s.float())
duration = time.time()-start
print(f'Floating point mdoel time: {duration}s')


start = time.time()
model_int8(x647s.float(),x750s.float())
duration = time.time()-start
print(f'Int mdoel time: {duration}s')

##################
model_int8.eval()
count = 0
for x647s,x750s,loc_mat,bc_mat in tqdm(test_dl):

    f,ax = plt.subplots(1,1,figsize=(20,20))
    
    out_p,out_bc = model_int8(x647s.float(),x750s.float())

    p_res = out_p.squeeze().detach().numpy()

    p_res_high = np.where(p_res>=0.9,1,0)
    p_res_mid = np.where(p_res<0.9,p_res,0)
    p_res_mid = np.where(p_res_mid>=0.6,1,0)
    
    p_res_high_locs = np.nonzero(p_res_high)
    p_res_mid_locs = np.nonzero(p_res_mid)

    jac_50 = m.jaccard_index(p_res,loc_mat.squeeze().detach().numpy(),0.5)
    acc_50 = m.accuracy(p_res,loc_mat.squeeze().detach().numpy(),0.5)

    p_gt_locs = np.nonzero(loc_mat.squeeze().detach().numpy())
    marker_size = 25
    ax.imshow(p_res,cmap='gray')
    ax.set_title(f'Prob map \n Jacc: {jac_50} Acc: {acc_50}')
    ax.plot(p_gt_locs[1],p_gt_locs[0],'ro',markersize=marker_size,label='GT')
    ax.plot(p_res_high_locs[1],p_res_high_locs[0],'g*',markersize=marker_size,label='High')
    ax.plot(p_res_mid_locs[1],p_res_mid_locs[0],'y*',markersize=marker_size,label='Mid')

    plt.legend()

    plt.tight_layout()

    plt.savefig(path.join(img_out_dir,f'prob_map{count}.png'))


    bc_res = out_bc.squeeze().detach().numpy()
    bc_gt = bc_mat.squeeze().detach().numpy()
    f,ax = plt.subplots(2,int(np.ceil(bc_res.shape[0]/2)),figsize=(50,20))

    marker_size=10
    axs = ax.flatten()

    for i,ax in enumerate(axs):
        _bc_res = bc_res[i]
        _gt = bc_gt[i]
        bc_res_high = np.where(_bc_res>=0.9,1,0)
        bc_res_mid = np.where(_bc_res<0.9,p_res,0)
        bc_res_mid = np.where(bc_res_mid>=0.6,1,0)
        bc_res_high_locs = np.nonzero(bc_res_high)
        bc_res_mid_locs = np.nonzero(bc_res_mid)
        bc_gt_locs = np.nonzero(_gt)

        ax.imshow(_bc_res,cmap='gray')
        ax.set_title(f'Bit {i}')
        ax.plot(bc_gt_locs[1],bc_gt_locs[0],'ro',markersize=marker_size,label='GT')
        ax.plot(bc_res_high_locs[1],bc_res_high_locs[0],'g*',markersize=marker_size,label='High')
        ax.plot(bc_res_mid_locs[1],bc_res_mid_locs[0],'y*',markersize=marker_size,label='Mid')

    plt.legend()

    plt.tight_layout()

    plt.savefig(path.join(img_out_dir,f'bc_map{count}.png'))
    count+=1
    break