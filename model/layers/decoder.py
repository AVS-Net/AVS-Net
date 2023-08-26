import torch
import torch.nn as nn

class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, hidden_dim, i_size, in_c, n_head=16):
        super(Decoder, self).__init__()
        
        self.i_size = i_size
        self.fc = nn.Linear(n_head * n_head + 1, i_size[0] * i_size[1])
        self.recon = nn.Conv2d(hidden_dim, in_c, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 2, 1)
        x = self.fc(x).reshape(b, -1, self.i_size[0], self.i_size[1])
        x = self.recon(x).permute(0, 2, 3, 1).contiguous()
        return x