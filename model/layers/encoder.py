import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        x = self.fc2(F.gelu(self.fc1(x)))
        return x
    
class Embed(nn.Module):
    # Pathes and Position Embedding Layer
    ## For example:
    #### Pathes Embedding (b, c, 640, 368); -> (b, 16 * 16 , hidden_dim)
    #### Position Embedding (b, , 16 * 16 , hidden_dim); -> (b, 16 * 16 + 1, hidden_dim)
    def __init__(self, hidden_dim=1024, n_head=16, p_size=(40,23), in_c=2):
        super(Embed, self).__init__()
                
        self.drop_rate = 0.1
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.embed_0 = nn.Conv2d(in_c, hidden_dim, kernel_size=p_size, stride=p_size)
        self.embed_1 = nn.Parameter(torch.zeros(1, n_head*n_head+1, hidden_dim))
        
    def forward(self, x):
        x = torch.view_as_real(x).permute(0, 3, 1, 2)
        token = self.token.expand(x.shape[0],-1,-1)
        p = self.embed_0(x)
        p = p.flatten(2).permute(0, 2, 1)
        x = torch.cat((token,p), dim=-2)
        return x + self.embed_1

class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, hidden_dim, ff_dim, p_size, in_c, n_head=16, drop_r=.1):
        super(Encoder, self).__init__()
        self.embed = Embed(hidden_dim, n_head, p_size, in_c)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, drop_r)
        self.lin_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(hidden_dim, ff_dim)
        self.drop = nn.Dropout(drop_r)
    
    def forward(self, x):
        x_in = self.embed(x)
        x = self.norm1(self.lin_proj(x_in))
        attn_out, _ = self.attn(x, x, x)
        x = x_in + self.drop(attn_out)
        ff_out = self.mlp(self.norm2(x))
        x = x + self.drop(ff_out)
        return x