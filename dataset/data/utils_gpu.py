import torch
from torch.fft import fftn, ifftn
import numpy as np
# It only works with Pytorch >= 1.7 which has fftn, ifftn support.

class Meta_GPU(object):
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
        return undersample(self.raw, self.sen, self.cf, self.acc)

def undersample(raw, sen, cf, acc):
    sen = torch.from_numpy(sen).cuda()
    ksp = torch.from_numpy(kspace(raw)).cuda()
    masks = torch.from_numpy(mask(raw, cf, acc)).cuda()
    ksp_acc = ksp * masks
    
    raw_gt, raw_ud = ifftn(ksp, dim=(-2, -1), norm="ortho"), ifftn(ksp_acc, dim=(-2, -1), norm="ortho")
    norm = max(1e-6, abs(raw_ud).max())
    ksp_acc, raw_gt, raw_ud = tuple(map(lambda _: _/norm, (ksp_acc, raw_gt, raw_ud)))
    
    ss = torch.complex(sen.real, -sen.imag)
    gt, ud = (raw_gt * ss).sum(dim=0), (raw_ud * ss).sum(dim=0)

    return gt, ud, ksp_acc, masks, sen

def kspace(raw):
    # shift and adjust raw -> ksp
    c, h, w = raw.shape
    x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    adjust = ((-1)**(x+y)[None, ...]).astype(np.float32)
    ksp = np.fft.ifftshift(raw, axes=(-2,-1)) * adjust
    return ksp

def mask(raw, cf=.08, acc=4):
    '''
        input: 
            raw; shape -> CHW(15, 640, 368) 
            
        output:
            mask; shape -> W(368,) -> ifftshift -> CHW(15, 640, 368) 
    '''
    c, h, w = raw.shape
    rng = np.random.RandomState(seed=1)
    lf = int(round(w * cf))
    prob = (w / acc - lf) / (w - lf)
    mask = rng.uniform(size=w) < prob
    pad = (w - lf + 1) // 2
    mask[pad:pad + lf] = True
    mask = np.fft.ifftshift(mask.astype(np.float32))
    return np.tile(mask,(c,h,1))

