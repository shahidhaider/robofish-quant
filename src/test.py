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
import argparse
import metrics as m

if __name__=="__main__":
    model = anglerFISH()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
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
    img_out_dir = path.join(path.dirname(statedict),'test_images')
    datapath = args.data_path

    tf = transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5),(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)),
        transforms.ConvertImageDtype(torch.float)])

    test_ds = sim_ds(root_data_dir = datapath,ds_type='test',ntiles=50,transforms=tf)

    test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)

    model.load_state_dict(torch.load(statedict))

    model.eval()
    count = 0
    for x647s,x750s,loc_mat,bc_mat in tqdm(test_dl):

        f,ax = plt.subplots(1,1,figsize=(20,20))
        
        out_p,out_bc = model(x647s.float(),x750s.float())

        p_res = out_p.squeeze().detach().numpy()


        jac_50 = m.jaccard_index(p_res,loc_mat.squeeze().detach().numpy(),0.5)
        acc_50 = m.accuracy(p_res,loc_mat.squeeze().detach().numpy(),0.5)

        p_res_high = np.where(p_res>=0.9,1,0)
        p_res_mid = np.where(p_res<0.9,p_res,0)
        p_res_mid = np.where(p_res_mid>=0.6,1,0)
        
        p_res_high_locs = np.nonzero(p_res_high)
        p_res_mid_locs = np.nonzero(p_res_mid)

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

