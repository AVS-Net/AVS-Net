import os
import matplotlib.pyplot as plt
import numpy as np
import torch

save_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../example"))

def plot_inplace(gt, ud, recon, masks, info, save_path="temp.png"):
    gt, ud, recon, masks = tuple(map(compat, [gt, ud, recon, masks]))
    h, w = gt.shape
    pn, pd  = .088*h, .18*w
    f, ax = plt.subplots(1,4, gridspec_kw={'height_ratios':[1]})

    ax[0].imshow(gt, cmap="gray")
    ax[0].text(pd, pn, "ground_truth", color="gray")
    ax[0].text(5, h-pn-20, "Loss: ", color="firebrick", fontsize=10)
    ax[0].text(88, h-pn+5, info[0][0], color="darkred", fontsize=6, weight="bold")
    ax[0].text(88, h-pn+30, info[0][1], color="red", fontsize=6, weight="bold")    
    ax[0].axis('off')

    ax[1].imshow(ud, cmap="gray")
    ax[1].text(pd, pn, "under_sample", color="gray")
    ax[1].text(5, h-pn-20, "SSIM: ", color="firebrick", fontsize=10)
    ax[1].text(88, h-pn+5, info[1][0], color="darkred", fontsize=6, weight="bold")
    ax[1].text(88, h-pn+30, info[1][1], color="red", fontsize=6, weight="bold")
    ax[1].axis('off')
    
    ax[2].imshow(recon, cmap="gray")
    ax[2].text(pd, pn, "reconstruction", color="gray")
    ax[2].text(5, h-pn-20, "PSNR: ", color="firebrick", fontsize=10)
    ax[2].text(88, h-pn+5, info[2][0], color="darkred", fontsize=6, weight="bold")
    ax[2].text(88, h-pn+30, info[2][1], color="red", fontsize=6, weight="bold")
    ax[2].axis('off')
    
    ax[3].imshow(masks.sum(dim=0), cmap="gray")
    ax[3].axis('off')
    
    plt.margins(0,0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    # plt.clf() 
    return f

def plot(gt, ud, ksp, mask, save_path=save_dir):
    plot_gt(gt,os.path.join(save_path,"gt.svg"))
    plot_ud(ud,os.path.join(save_path,"ud.svg"))
    plot_ksp(ksp,os.path.join(save_path,"ksp.svg"))
    plot_mask(mask,os.path.join(save_path,"mask.svg"))
    
def _plot(data, save_path=os.path.join(save_dir,"default.svg")):
    
    plt.imshow(data, cmap="gray")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    plt.clf()    

def plot_mask(mask, save_path=os.path.join(save_dir,"mask.svg")):
    _plot(abs(mask.sum(dim=0)), save_path)
    
def plot_ksp(ksp, save_path=os.path.join(save_dir,"ksp.svg")):
    _plot(abs(ksp.sum(dim=0)), save_path)

def plot_ksp_ud(ksp_ud, save_path=os.path.join(save_dir,"ksp_ud.svg")):    
    _plot(abs(ksp_ud.sum(dim=0)), save_path)
    
def plot_raw_gt(raw_gt, save_path=os.path.join(save_dir,"raw_gt.svg")):    
    _plot(abs(raw_gt.sum(dim=0)), save_path)
    
def plot_raw_ud(raw_ud, save_path=os.path.join(save_dir,"raw_ud.svg")):    
    _plot(abs(raw_ud.sum(dim=0)), save_path)

def plot_sen(sen, save_path=os.path.join(save_dir,"sen.svg")):    
    
    for i in sen:
        _plot(abs(i), save_path)
    plt.show()
    plt.clf()     
def plot_gt(gt, save_path=os.path.join(save_dir,"gt.svg")):    
    _plot(abs(gt), save_path)
    
def plot_ud(ud, save_path=os.path.join(save_dir,"ud.svg")):    
    _plot(abs(ud), save_path)
    
def plot_ksp_ifft(ksp, save_path=os.path.join(save_dir,"ksp_ifft.svg")):

    c, h, w = ksp.shape
    x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    adjust = ((-1)**(x+y)[None, ...]).astype(np.float32)
    ksp =  np.fft.ifftshift(ksp,axes=(-2,-1)) * adjust
    
    _plot(abs(ksp[0], save_path))

def compat(data):
    try:
        data = data.detach().cpu()
    except:
        data = torch.from_numpy(data)
        
    if data.shape[-1]==2:
        data = torch.view_as_complex(data)
    return abs(data)
    
            