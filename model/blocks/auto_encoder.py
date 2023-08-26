import torch
import torch.nn as nn

from model import DC, WA
try:
    from model import Transformer
except:
    from .transformer import Transformer
    print("Warning using single file debug mode!")

class AVSNET(nn.Module):
    def __init__(
            self, 
            hidden_dim = 2048,
            ff_dim = 4096, 
            p_size = (40, 23),
            i_size = (1, 640, 368),
            in_c = 2,
            n_head = 16,
            drop_r = .1,
            dc_para = .1,
            wa_para = .1, 
            n_block = 1
        ):
        super(AVSNET, self).__init__()
        
        self.n_block = n_block 
        transformer_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(n_block):
            transformer_blocks.append(
                Transformer(hidden_dim = hidden_dim,
                    ff_dim = ff_dim, 
                    p_size = p_size,
                    i_size = i_size,
                    in_c = in_c,
                    n_head = n_head,
                    drop_r = drop_r)
                ) 
            dc_blocks.append(DC(dc_para)) 
            wa_blocks.append(WA(wa_para)) 

        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
    def forward(self, x, ksp, mask, sen):
        for i in range(self.n_block):
            Tx = self.transformer_blocks[i](x)
            Sx = self.dc_blocks[i].apply(x, ksp, mask, sen)
            x = self.wa_blocks[i].apply(x + Tx, Sx)
        return x