import torch
import torch.nn as nn

try:
    from model import Encoder, Decoder
except:
    print("Warning using single file debug mode!!!")

class Transformer(nn.Module):
    def __init__(
            self, 
            hidden_dim = 2048,
            ff_dim = 4096, 
            p_size = (40, 23),
            i_size = (1, 640, 368),
            in_c = 2,
            n_head = 16,
            drop_r=.1
        ):
        super().__init__()
        
        self.encoder = Encoder(
            hidden_dim = hidden_dim,
            ff_dim = ff_dim,
            i_size = i_size, 
            in_c = in_c,
            n_head = n_head,
            drop_r = drop_r)
        
        self.decoder = Decoder(
            hidden_dim = hidden_dim, 
            i_size = i_size, 
            in_c = in_c,
            n_head = n_head
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.view_as_complex(x)