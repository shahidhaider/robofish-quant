import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
import os.path as path
import skimage.io as skio
import numpy as np
root_data_dir = './Data/50'

ds_type = ['train','test','val']

class sim_ds(Dataset):

    def __init__(self,root_data_dir = root_data_dir, ds_type = 'train',ir_num = 16,ntiles=500,transforms=None,recompile=False):
        

        self.data_dir = path.join(root_data_dir,ds_type,'1')

        self.data_dir647 = path.join(self.data_dir,'647nm, Raw')
        self.data_dir750 = path.join(self.data_dir,'750nm, Raw')
        
        self.irs = [i for i in range(ir_num)]
        self.fovs = [i for i in range(ntiles)]

        self.gt = path.join(self.data_dir,'groundtruths')
        
        self.img_format = 'merFISH_{:02d}_{:03d}_01.tiff'
        self.freq_format = 'frequency_{}.csv'
        self.loc_format = 'groundtruth_{}.csv'
        #get imagesize

        with open(path.join(self.data_dir,'config.yml')) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)

        self.img_size = self.params['simulation']['image_size']
        self.ds_size = np.minimum(self.params['simulation']['tile_count'],ntiles)
        self.emitters = self.params['simulation']['emitter_count']
        self.non_emitters = self.img_size**2 - self.params['simulation']['emitter_count']
        self.transforms = transforms
        self.fname_root = 'img_{}.ds'
        if recompile:
            
            for idx in range(self.ds_size):

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

                torch.save({'x647s':x647s,'x750s':x750s,'locmat':loc_mat,'bcmat':bc_mat},path.join(self.data_dir,self.fname_root.format(idx)))
    def __len__(self):
        return self.ds_size
    def __getitem__(self,idx):
    
        sample = torch.load(path.join(self.data_dir,self.fname_root.format(idx))) 

        if not (self.transforms is None):
            x647s = self.transforms(sample['x647s'])
            x750s = self.transforms(sample['x750s'])

        return x647s,x750s,sample['locmat'],sample['bcmat']
