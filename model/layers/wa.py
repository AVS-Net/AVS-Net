import torch
import torch.nn as nn

class WA(nn.Module):
    def __init__(self, para=0):
        super(WA, self).__init__()
        self.para = nn.Parameter(torch.Tensor([para]))

    def apply(self, x, Sx):
        x = self.para * x + (1 - self.para) * Sx
        return x
    
class WA_AMP(nn.Module):
    def __init__(self, para=0):
        super(WA_AMP, self).__init__()
        self.para = nn.Parameter(torch.Tensor([para]))

    def apply(self, x, Cx, Sx):
        x = self.para * x + self.para * torch.view_as_complex(Cx.float()) + (1 - self.para) * Sx
        return x