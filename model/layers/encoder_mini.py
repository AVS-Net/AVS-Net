import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, h_dim, ff_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(h_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, h_dim)

    def forward(self, x):
        x = self.fc2(F.gelu(self.fc1(x)))
        return x
    
class Embed(nn.Module):
    # Pathes and Position Embedding Layer
    ## For example:
    #### Pathes Embedding (b, 2, 640, 368); -> (b, n_head, h_dim)
    #### Position Embedding (b, , 16 * 16 , h_dim); -> (b, 16 * 16 + 1, h_dim)
    def __init__(self, h_dim=1024, n_head=16, m_dim=512, i_size=(1, 640, 368)):
        super(Embed, self).__init__()
        
        self.h_dim = h_dim        
        self.n_head = n_head        
        self.drop_rate = 0.1
        self.token = nn.Parameter(torch.zeros(1, 1, h_dim))
        self.fc = nn.Linear(m_dim,h_dim)
        # self.embed_0 = nn.Conv2d(in_c, h_dim, kernel_size=p_size, stride=p_size)
        self.embed_1 = nn.Parameter(torch.zeros(2*i_size[0], n_head + 1, h_dim))
        
    def forward(self, x):
        # p = self.embed_0(x)
        p = self.apply_patch(x)
        token = self.token.expand(p.size(0),-1,-1)
        x = torch.cat((token,p), dim=-2)
        return x + self.embed_1

    def apply_patch(self, x):
        x = torch.view_as_real(x).permute(3, 0, 1, 2)
        b, c, _, _ = x.shape
        x = x.reshape((b*c, self.n_head , -1))
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, hidden_dim, ff_dim, i_size, in_c, n_head=16, drop_r=.1):
        super(Encoder, self).__init__()
        _dim = hidden_dim // n_head
        self.embed = Embed(_dim, n_head, i_size[1] * i_size[2] // n_head , i_size)
        self.attn = nn.MultiheadAttention(_dim, n_head, drop_r)
        self.lin_proj = nn.Linear(_dim, _dim)
        self.norm1 = nn.LayerNorm(_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(_dim, eps=1e-6)
        self.mlp = MLP(_dim, ff_dim)
        self.drop = nn.Dropout(drop_r)
    
    def forward(self, x):
        x_in = self.embed(x)
        x = self.norm1(self.lin_proj(x_in))
        attn_out, _ = self.attn(x, x, x)
        x = x_in + self.drop(attn_out)
        ff_out = self.mlp(self.norm2(x))
        x = x + self.drop(ff_out)
        return x