import torch
import os
from .loss import NMSE, PSNR, SSIM
from dataset import plot_inplace

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/rds/projects/d/duanj-ai-in-medical-imaging/fast_mri/experiments/transformer')

def log(gt, ud, rec, masks, loss, writer, save_dir=None, global_step=None, epoch=None, mode="train"):
    data = show_batch_info(gt, ud, rec, masks, save_dir, epoch)
    fn = tuple(zip(["NMSE","SSIM","PSNR"],["NMSE","SSIM","PSNR"]))
    for fig, info in data:
        writer.add_figure(mode, figure=fig, global_step=global_step)
        writer.add_scalar("loss/{}".format(mode), loss, global_step=global_step)
        for j in range(3):
            writer.add_scalar("{}/base_{}".format(fn[j][0], mode), info[j][0], global_step=global_step)
            writer.add_scalar("{}/{}".format(fn[j][1], mode), info[j][1], global_step=global_step)
        fig.clf()
        break
                
def show_batch_info(gt, ud, rec, masks, save_dir=None, epoch=None):
    # Plot a batch of data.
    data = []
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if gt.shape[0] == 1:
        fig, info = show_info(gt[0], ud[0], rec[0], masks[0], "{}/{}.png".format(save_dir, epoch))
        data.append([fig, info])
    
    for b, (i, j, k, m) in enumerate(zip(gt, ud, rec, masks)):
        fig, info = show_info(i, j, k, m, "{}/{}-b{}.png".format(save_dir, epoch, b))
        data.append([fig, info])
        break
    return data

def show_info(gt, ud, rec, masks, save_path):

    gt = abs(gt.detach().cpu())
    ud = abs(ud.detach().cpu())
    rec = abs(rec.detach().cpu())
    masks = abs(masks.detach().cpu())

    base_info = [NMSE(gt, ud), SSIM(gt, ud), PSNR(gt, ud)]
    proc_info = [NMSE(gt, rec), SSIM(gt, rec), PSNR(gt, rec)]

    info = tuple(zip(base_info, proc_info))

    fig = plot_inplace(gt, ud, rec, masks, info, save_path)
    
    return fig, info