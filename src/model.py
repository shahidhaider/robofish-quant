import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = torch.cat((x,skip),dim=1)
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
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                padding_mode='reflect',
                                )

    def forward(self,x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
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

    def __init__(self,in_channels=8,first_stage_channels=48):
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

        #self.sigmoid = nn.Sigmoid()
    def forward(self,x647,x750):

        x = self.twostage(x647,x750)
        #out_p = self.sigmoid(self.head_p(x))
        #out_bc = self.sigmoid(self.head_bc(x))

        out_p = (self.head_p(x))
        out_bc =(self.head_bc(x))


        return out_p,out_bc

        