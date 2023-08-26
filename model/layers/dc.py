from torch.fft import fftn, ifftn
import torch
import torch.nn as nn


class DC(nn.Module):
    def __init__(self, noise_lvl=0):
        super(DC, self).__init__()
        self.noise_lvl = nn.Parameter(torch.Tensor([noise_lvl]))

    def apply(self, x, ksp_acc, mask, sen):
        """
        k    - input in k-space
        ksp_acc   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = x.unsqueeze(1) * sen
        try:
            k = torch.fft.fft2(x, norm="ortho")
        except:
            k = fftn(x, dim=(-2, -1), norm="ortho")


        v = self.noise_lvl
        
        out =  k * (1 - mask) +  (v * k + (1 - v) * ksp_acc) * mask 

        try:
            x = torch.fft.ifft2(out, norm="ortho")
        except:
            x = ifftn(out, dim=(-2, -1), norm="ortho")
        ss = torch.complex(sen.real, -sen.imag)
        Sx = (x * ss).sum(dim=1)
        return Sx
    
class DC_AMP(nn.Module):
    def __init__(self, noise_lvl=0):
        super(DC_AMP, self).__init__()
        self.noise_lvl = nn.Parameter(torch.Tensor([noise_lvl]))

    def apply(self, x, ksp_acc, mask, sen):
        """
        k    - input in k-space
        ksp_acc   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = x.unsqueeze(1) * sen
        k = fftn(x, dim=(-2, -1), norm="ortho")

        v = self.noise_lvl
        
        out =  k * (1 - mask) +  (v * k + (1 - v) * ksp_acc) * mask 

        x = ifftn(out, dim=(-2, -1), norm="ortho")
        ss = torch.complex(sen.real, -sen.imag)
        Sx = (x * ss).sum(dim=1)
        return Sx
