# Import dependencies
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import PIL
from PIL import Image



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


eps= 1e-12   #1e-4 
def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)



class SelfAttn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma*attn_g
        return out


class UNet(torch.nn.Module):

  def unet_conv(self , ch_in , ch_out , is_leaky):
    if is_leaky:
      return nn.Sequential(
          nn.Conv2d(ch_in , ch_out , 3 , padding=1),
          nn.BatchNorm2d(ch_out),
          nn.LeakyReLU(0.2 , inplace=True)
      )
    else:
      return nn.Sequential(
          nn.Conv2d(ch_in , ch_out , 3 , padding=1),
          nn.BatchNorm2d(ch_out),
          nn.ReLU(True)
      )

  def up(self,ch_in,ch_out):
    return nn.Sequential(
                          # nn.ConvTranspose2d(ch_in , ch_out , 3, 2 , 1 ,1),
                          # nn.BatchNorm2d(ch_out),
                          # nn.PReLU()
                          nn.Conv2d(ch_in, ch_out*4, kernel_size=3, padding=1),
                          nn.BatchNorm2d(ch_out*4),
                          nn.PixelShuffle(upscale_factor=2),
                          nn.PReLU()
                          ,nn.Conv2d(ch_out , ch_out , 3 , padding=1),
                          nn.BatchNorm2d(ch_out),
                          nn.PReLU()
                          
    )
    
    # nn.Sequential(
    #     nn.ConvTranspose2d(ch_in , ch_out , 3, 2 , 1 ,1),
    #     # nn.BatchNorm2d(ch_out),
    #     nn.PReLU()
    #     )
    
  def pooling(self,ch_in,ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out , kernel_size=2,stride=2, padding=0),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(True)
    )
  
  def __init__(self,is_leaky):
    super(UNet,self).__init__()

    #Residual connection in starting of u-net
    self.res1 = self.unet_conv(3 , 32 , is_leaky)
    self.res2 = self.unet_conv(32 , 32 , is_leaky)
    self.att_start = Attention_block(F_g=32,F_l=32,F_int=16)

    # First encoding layer
    self.conv1 = self.unet_conv(64,64, is_leaky)                    
    # Second encoding layer
    self.conv2 = self.unet_conv(64,128, is_leaky )                  
    # Third encoding layer
    self.conv3 = self.unet_conv(128,256, is_leaky)                 
    # Forth encoding layer
    self.conv4 = self.unet_conv(256,512, is_leaky)                 
    # Fifth encoding layer
    self.conv5 = self.unet_conv(512,1024, is_leaky)                 

    #Pooling layer
    self.pool1 =  self.pooling(64,64)  
    self.pool2 =  self.pooling(128,128)
    self.pool3 =  self.pooling(256,256)
    self.pool4 =  self.pooling(512,512)

    # First Upsampling layer  with attention blocks
    self.up1  = self.up(1024,512)                                  
    self.att1 = Attention_block(F_g=512,F_l=512,F_int=256)
   
    # Second Upsampling layer
    self.up2  = self.up(512,256)                                   
    self.att2 = Attention_block(F_g=256,F_l=256,F_int=128)
    # Third Upsampling layer
    self.up3  = self.up(256,128)                                     
    self.att3 = Attention_block(F_g=128,F_l=128,F_int=64)
    # Fourth Upsampling layer
    self.up4 = self.up(128,64)                                      
    self.att4 = Attention_block(F_g=64,F_l=64,F_int=32)
    self.self_att1 = SelfAttn(256)


    # First Decoding layer
    self.conv6 = self.unet_conv(1024, 512, False )
    # Second Decoding layer
    self.conv7 = self.unet_conv(512,  256, False )
    # Third Decoding layer 
    self.conv8 = self.unet_conv(256,  128, False  )
    # Fourth Decoding layer
    self.conv9 = self.unet_conv(128,  64, False  )

    # self.res_att = self.unet_conv(256, 64, False )

    # # Residual connections in the end of u-net
    # self.res3       = self.unet_conv(64 ,64, False )
    # self.self_att2  = SelfAttn(64)
    # self.res4       = self.unet_conv(64 ,64, False )
    # self.att_end    = Attention_block(F_g=64,F_l=64,F_int=32)
    # self.resEnd     = self.unet_conv(128 , 64, False  )


    #Last layer
    self.conv10 = nn.Conv2d(64,3, kernel_size=1 , padding=0)                            

   
  def forward(self, x):

    #Residual connections
    r1 = self.res1(x)
    r2 = self.res2(r1)

    r1 = self.att_start(g=r2 , x=r1)

    #Encoding Path
    x1 = self.conv1( torch.cat( (r1 , r2), 1) )
    r1=[]
    r2=[]
    x2 = self.conv2(self.pool1(x1))
    x3 = self.conv3(self.pool2(x2))
    x4 = self.conv4(self.pool3(x3))
    x5 = self.conv5(self.pool4(x4))                            # out 2 x 2 x 1024

    
    #Decoding Path with attention
    x  = self.up1(x5)
    x5=[]
    x4 = self.att1(g=x , x=x4)
    x  = self.conv6( torch.cat(( x ,  x4 ),1 ) )
    x4=[]

    x  = self.up2(x)
    x  = self.self_att1(x)      
    x3 = self.att2(g=x , x=x3)
    x  = self.conv7( torch.cat(( x ,  x3 ), 1 ) )
    x3=[]

    x  = self.up3(x)
    x2 = self.att3(g=x , x=x2)
    x  = self.conv8( torch.cat(( x , x2 ), 1 ) )
    x2=[]

    x  = self.up4(x)
    x1 = self.att4(g=x , x=x1)                     
    x1 = self.conv9( torch.cat(( x , x1 ), 1 ) )
    # x  = self.self_att1(x1)                             
    # x  = self.res_att( torch.cat((x , x1 ), 1 ) )
    # x1=[]

    # #Residual in End
    # r3 = self.res3(x)                  
    # r3 = self.self_att2(r3)                           
    # r4 = self.res4(r3)
    # x  = self.att_end(g=r4 , x=x)
    # x  = self.resEnd(torch.cat( (r4 , x), 1))          
    r4=[]
    r3=[]

    # x = self.conv10(x)
    x = self.conv10(x1)
    m = nn.Tanh()
    x = m(x)

    return x