from model import anglerFISH
from dataset import sim_ds
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import metrics as m
import torch.quantization as quant
import argparse
import os
import time
import warnings

warnings.simplefilter('ignore')
class robofish_quant(nn.Module):
        def __init__(self):
            super(robofish_quant,self).__init__()
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

def test_run(model,test_dl,img_out_dir):
    os.makedirs(img_out_dir, mode=511, exist_ok=True)
    model.eval()
    count = 0
    for x647s,x750s,loc_mat,bc_mat in tqdm(test_dl):

        f,ax = plt.subplots(1,1,figsize=(20,20))
        
        out_p,out_bc = model(x647s.float(),x750s.float())

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


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print(f'{label}: \t Size (KB): {size/1e3}')
    os.remove('temp.p')
    return size

def time_model(model,test_dl,label=''):
    x647s,x750s,loc_mat,bc_mat = next(iter(test_dl))

    start = time.time()
    model(x647s.float(),x750s.float())
    duration = time.time()-start
    print(f'{label}: \t {duration}s')

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser.add_argument(
        '--qbackend',
        type=str,
        help='Quantisation Backend',
        default='qnnpack'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data',
        default = './Data/50/'
    )

    parser.add_argument(
        '--state_dict',
        type=str,
        help='Path to the training data',
        default='./best_state_dicts/feb_11/state_dict.pth'

    )

    args = parser.parse_args()
    statedict = args.state_dict
    
    datapath = args.data_path

    torch.backends.quantized.engine = args.qbackend

    model = anglerFISH()


    tf = transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5),(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)),
        transforms.ConvertImageDtype(torch.float)])

    test_ds = sim_ds(root_data_dir = datapath,ds_type='test',ntiles=50,transforms=tf)

    test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)

    model.load_state_dict(torch.load(statedict))

    model.eval()

    # Dyanmic Quant
    dyquant_robofish = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d}, dtype=torch.qint8
    )


    # print('Here is the floating point version of this module:')
    # print(model)
    # print('')
    # print('and now the quantized version:')
    # print(quant_robofish)


   

    time_model(model,test_dl,"Floating Point Model")
    time_model(model,test_dl,"Dynamic Quant Model")
    img_out_dir = path.join(path.dirname(statedict),'dynamic_quant_test_images')

    test_run(model=dyquant_robofish,test_dl=test_dl,img_out_dir=img_out_dir)


    ## Static Quantization
    model_quant_ready = robofish_quant()
    # set the qconfig for PTQ
    model_quant_ready.qconfig = quant.get_default_qconfig(args.qbackend)
    # set the qengine to control weight packing
    torch.backends.quantized.engine = args.qbackend

    model_fused = model_quant_ready # Would fuse, but difficult with multiple submodules

    model_prepared = torch.quantization.prepare(model_quant_ready)

    x647s,x750s,loc_mat,bc_mat = next(iter(test_dl))
    model_prepared(x647s.float(),x750s.float())
    model_int8 = torch.quantization.convert(model_prepared)

    # print('Here is the floating point version of this module:')
    # print(model)
    # print('')
    # print('and now the quantized version:')
    # print(model_int8)

    time_model(model_int8,test_dl,label="Static Quant Model")
    img_out_dir = path.join(path.dirname(statedict),'static_quant_test_images')
    test_run(model=model_int8,test_dl=test_dl,img_out_dir=img_out_dir)


    # compare the sizes
    f=print_size_of_model(model,"Floating")
    dq=print_size_of_model(dyquant_robofish,"Dynamic Quant")
    print("{0:.2f} times smaller".format(f/dq))
    sq=print_size_of_model(model_int8,"Static Quant")
    print("{0:.2f} times smaller".format(f/sq))