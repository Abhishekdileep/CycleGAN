import torch 
import torch.nn as nn 

'''
    Class Block 
        Holds the Intial discriminator Block with all the Conv layer 
'''

class Block(nn.Module) : 
    def __init__(self , in_channels , out_channels , stride ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 4 , stride , 1 , bais=True , padding_mode="reflect")
        )