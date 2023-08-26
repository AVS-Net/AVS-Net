import torch
import torch.nn as nn

class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, hidden_dim, i_size, in_c, n_head=16):
        super(Decoder, self).__init__()
        _dim = hidden_dim // n_head
        
        self.i_size = i_size
        self.fc1 = nn.Linear(n_head + 1, n_head)
        self.fc2 = nn.Linear(_dim, i_size[1] * i_size[2] //n_head)
    
    def forward(self, x):
        b = x.shape[0]//2
        x = x.permute(0, 2, 1)
        x = self.fc1(x).permute(0, 2, 1)
        x = self.fc2(x).reshape(2, self.i_size[0], self.i_size[1], self.i_size[2])
        x = x.permute(1, 2, 3, 0).contiguous()
        return x