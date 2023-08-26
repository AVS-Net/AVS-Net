import torch
import numpy as np
from scipy import fft

# It was written to compat with Pytorch <1.9

class Meta(object):
    '''
    Meta Class for MRI raw and sen data loaded with CHW format.
    #### Member: raw, and sen; Both with CHW(15, 640, 368).
    ##### raw: kspace data with 15 channels.
    ##### sen: Coil sensitivities precomputed using BART.
    '''
    
    # CHW (15,640, 368)
    def __init__(self, raw, sen, cf=.08, acc=4):
        self.raw = raw
        self.sen = sen
        self.gt  = None
        self.ud  = None
        self.ksp_acc = None # masked or accelerated kspace data
        self.masks = None
        self.cf = cf
        self.acc = acc
    
    def excite(self):
        '''
        Excite input data: (gt, ud, ksp_acc, masks, sen) from meta.
        ### Where, gt, ud with HW(640, 368).
        #### ksp_acc, masks, sen with  CHW(15, 640, 368).
        '''
        self.gt, self.ud, self.ksp_acc, self.masks, self.sen = undersample(self.raw, self.sen, self.cf, self.acc)
        return torch.from_numpy(self.gt), torch.from_numpy(self.ud), torch.from_numpy(self.ksp_acc), torch.from_numpy(self.masks), torch.from_numpy(self.sen)
    
def undersample(raw, sen, cf, acc):
    ksp = kspace(raw)
    masks = mask(raw, cf, acc)
    ksp_acc = ksp * masks
    
    raw_gt, raw_ud = fft.ifft2(ksp, norm="ortho"), fft.ifft2(ksp_acc, norm="ortho")
    norm = max(1e-6, abs(raw_ud).max())
    ksp_acc, raw_gt, raw_ud = tuple(map(lambda _: _/norm, (ksp_acc, raw_gt, raw_ud)))
    
    ss = sen.real + sen.imag * (-1j)
    gt, ud = (raw_gt * ss).sum(axis=0), (raw_ud * ss).sum(axis=0)
    
    return gt, ud, ksp_acc, masks, sen

def kspace(raw):
    # shift and adjust raw -> ksp
    c, h, w = raw.shape
    x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    adjust = ((-1)**(x+y)[None, ...]).astype(np.float32)

    ksp = fft.ifftshift(raw, axes=(-2,-1)) * adjust
    return ksp

def mask(raw, cf=.08, acc=4):
    '''
        input: 
            raw; shape -> CHW(16, 640, 368) 
            
        output:
            mask; shape -> W(368,) -> ifftshift -> CHW(16, 640, 368) 
    '''
    c, h, w = raw.shape
    rng = np.random.RandomState(seed=1)
    lf = int(round(w * cf))
    prob = (w / acc - lf) / (w - lf)
    mask = rng.uniform(size=w) < prob
    pad = (w - lf + 1) // 2
    mask[pad:pad + lf] = True
    mask = fft.ifftshift(mask.astype(np.float32))
    return np.tile(mask, (c, h, 1))
