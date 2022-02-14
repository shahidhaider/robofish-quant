import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from quant_test import robofish_quant
from dataset import sim_ds
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import mlflow
from model import anglerFISH
import copy
           
def prob_loss(pred,target):

    return F.binary_cross_entropy_with_logits(pred,target,reduction='mean')


def valid_barcode_loss(pred:torch.tensor,target:torch.tensor,weight=1,reduction=torch.mean,target_device='cpu'):
    pos_weights = torch.tensor(16*[weight]).to(device=target_device)
    _t = F.binary_cross_entropy_with_logits(pred,target,pos_weight=pos_weights,reduction='none')
    # _t = weight*target * torch.maximum(torch.log(pred),limit) + (1-target)*torch.maximum(torch.log(1-pred),limit)
    return reduction(_t)

def barcode_loss(pred:torch.tensor,target:torch.tensor,weight=1,valid_weight=12/4,target_device='cpu'):

    target = target.permute(0,2,3,1)
    target = target.reshape(-1,16)
    

    pred = pred.permute(0,2,3,1)
    pred = pred.reshape(-1,16)

    flags = ((target.sum(dim=1)>0)*1).to(device=target_device)
    
    loss = torch.tensor(0.).to(device=target_device)
    for i in range(pred.shape[0]):
 
        loss += flags[i]*valid_barcode_loss(pred[i],target[i],valid_weight,target_device=target_device) + (1-flags[i])*F.binary_cross_entropy_with_logits(pred[i],target[i],reduction='mean')/weight
    loss /=pred.shape[0] 
    return loss



def train(model,dataset,train_dataloader,val_dataloader,optimizer_fn,epochs=2,print_rate=10,device='cpu',a=1,b=16,grad_clip=1e-2):
    
    if device=='gpu' and torch.cuda.is_available():
        target = torch.device('cuda')
        print('Using GPU')
    else:
        target = 'cpu'
        print('Using CPU')
        if torch.cuda.is_available():
            print('CUDA is available')

    model = model.to(device=target)
    val_lowest = 99999999
    for e in range(epochs):
        count = 0
        epoch_loss= 0
        epoch_pos_loss = 0
        epoch_bc_loss = 0
        
        model.train()
        for x647s,x750s,loc_mat,bc_mat in tqdm(train_dataloader):
            x647s,x750s,loc_mat,bc_mat = x647s.float().to(device=target),x750s.float().to(device=target),loc_mat.to(device=target),bc_mat.to(device=target)
            count+=1
            optimizer_fn.zero_grad()
            out_p,out_bc = model(x647s,x750s)
            
            train_bc_loss = barcode_loss(out_bc,bc_mat,dataset.non_emitters/dataset.emitters,12/4,target)
            
            train_prob_loss = prob_loss(out_p,loc_mat)
            
            loss = b*train_bc_loss + a*train_prob_loss
            epoch_pos_loss+=train_prob_loss.item()
            epoch_bc_loss+=train_bc_loss.item()
            #print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
            optimizer_fn.step()
            epoch_loss +=loss.item()
            if count % print_rate==0:
                denom = count
                print('Epoch: {} | Avg Running Loss per sample: {}'.format(e,epoch_loss/denom))

        denom = count
        mlflow.log_metric('train_loss',epoch_loss/denom)
        mlflow.log_metric('train_pos_loss',epoch_pos_loss/denom)
        mlflow.log_metric('train_barcode_loss',epoch_bc_loss/denom)

        model.eval()
        #model_int8 = torch.quantization.convert(model)
        val_count=0
        total_val_loss =0
        total_val_pos_loss = 0
        total_val_bc_loss = 0
        for x647s,x750s,loc_mat,bc_mat in tqdm(val_dataloader):
            x647s,x750s,loc_mat,bc_mat = x647s.float().to(device=target),x750s.float().to(device=target),loc_mat.to(device=target),bc_mat.to(device=target)
            val_count+=1
            out_p,out_bc = model(x647s,x750s)
            val_bc_loss = barcode_loss(out_bc,bc_mat,dataset.non_emitters/dataset.emitters,12/4,target)
            
            val_prob_loss = prob_loss(out_p,loc_mat)
            
            val_loss = b*val_bc_loss + a*val_prob_loss
            total_val_loss += val_loss.item()
            total_val_pos_loss += val_prob_loss.item()
            total_val_bc_loss +=val_bc_loss.item()
        denom = val_count
        mlflow.log_metric('val_loss',total_val_loss/denom)
        mlflow.log_metric('val_pos_loss',total_val_pos_loss/denom)
        mlflow.log_metric('val_barcode_loss',total_val_bc_loss/denom)

        if total_val_loss/denom < val_lowest:
            best_model = model
            mlflow.pytorch.log_state_dict(best_model.state_dict(), artifact_path=f"checkpoint/epoch{e}")
            val_lowest = total_val_loss/denom
    return best_model

if __name__=="__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of Epochs to train'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Size of batch'
    )
    

    parser.add_argument(
        '--gpu',
        type=int,
        help='GPU: 1 for gpu, 0 for cpu',
        default=0
    )

    parser.add_argument(
        '--recompile_ds',
        type=int,
        help='Recache the dataset. 1 to recache, 0 to use cached',
        default=0
    )

    parser.add_argument(
        '--print_rate',
        type=int,
        help='The rate of batch samples to print loss functions',
        default=10
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning Rate',
        default=1e-3
    )

    parser.add_argument(
        '--grad_clip',
        type=float,
        help='Gradient Clip',
        default=1e-2
    )

    parser.add_argument(
        '--qbackend',
        type=str,
        help='Quant Backend',
        default='qnnpack'
    )



    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")
    print("===== GPU =====")
    print(f'GPU: {args.gpu}')
    print("================")
    print("===== PARAMS =====")
    print(f'Batch Size: {args.batch_size}')
    print(f'Learning Rate: {args.lr}')
    print(f'Grad Clip: {args.grad_clip}')

    print("================")
    model = robofish_quant(anglerFISH())


    device = (args.gpu==1)*'gpu' + (args.gpu==0)*'cpu'
    print(device)
    

    transforms = transforms.Compose([
     transforms.Normalize((0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5),(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)),
     
     transforms.ConvertImageDtype(torch.float)])

    train_ds = sim_ds(root_data_dir = args.data_path, ds_type='train',ntiles=500,transforms=transforms,recompile=(args.recompile_ds==1))
    # test_ds = sim_ds(root_data_dir = args.data_path,ds_type='test',ntiles=50,transforms=transforms)
    val_ds = sim_ds(root_data_dir = args.data_path,ds_type='val',ntiles=50,transforms=transforms,recompile=(args.recompile_ds==1))

    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    # test_dl = DataLoader(test_ds,batch_size=1,shuffle=True)
    val_dl = DataLoader(val_ds,batch_size=1,shuffle=True)


    model.qconfig = torch.quantization.get_default_qat_qconfig(args.qbackend) #Specifically the QAT qconfig
    torch.backends.quantized.engine = args.qbackend #Weight packing backend

    model_fused = copy.deepcopy(model)


    for module_name, module in model_fused.main.twostage.named_children():
        for stage_name, stage_module in module.named_children():
            stage_module=torch.quantization.fuse_modules(stage_module,[['conv1', 'bn1'],['conv2', 'bn2'],['conv3', 'bn3']],inplace=True)
            
    model_prepped =torch.quantization.prepare_qat(model_fused)
    optimizer_fn = optim.AdamW(model_prepped.parameters(), lr=args.lr)
    local=False
    
    if local:
        model_prepped.train()
        bm = train(model=model_prepped,
                    dataset=train_ds,
                    train_dataloader=train_dl,
                    val_dataloader = val_dl,
                    optimizer_fn=optimizer_fn,
                    device=device,
                    print_rate=args.print_rate,
                    epochs=args.epochs,
                    grad_clip=args.grad_clip)
    else:

        with mlflow.start_run():
            model_prepped.train()
            mlflow.log_param('QAT','Yes')
            bm = train(model=model_prepped,
                    dataset=train_ds,
                    train_dataloader=train_dl,
                    val_dataloader = val_dl,
                    optimizer_fn=optimizer_fn,
                    device=device,
                    print_rate=args.print_rate,
                    epochs=args.epochs,
                    grad_clip=args.grad_clip)
            torch.quantization.convert(bm, inplace=True)

            mlflow.pytorch.log_model(bm,"model")

    
        