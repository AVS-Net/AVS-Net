
from figure import plot_inplace
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

def excite(meta):
    gt, ud, ksp_acc, masks, sen  = meta.excite()
    
    gt = abs(gt.detach().cpu().numpy())
    ud = abs(ud.detach().cpu().numpy())
    ksp_acc = abs(ksp_acc.detach().cpu().numpy())
    masks = abs(masks.detach().cpu().numpy())
    nsm = normalized_root_mse(gt, ud)
    psnr = peak_signal_noise_ratio(gt, ud, data_range=gt.max())
    ssim = structural_similarity(gt, ud, data_range=gt.max())
    info = [[nsm,nsm], [ssim,ssim], [psnr,psnr]]
    
    return gt, ud, ksp_acc, masks, info

# if __name__ == "__main__":
#     import os
#     from data.utils_np import Meta_np
#     from data.utils_sci import Meta_sci
#     from data.utils_torch import Meta_torch
#     from data.utils import Meta
#     from figure import plot_inplace
#     from dataset.loader import get_pth, load_item
#     from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
    
#     data_path = os.path.abspath(os.path.join(os.path.abspath(__file__),"../knee/coronal_pd"))
#     path_dict = get_pth(data_path)
#     data = path_dict['train'][3]
#     meta_np = Meta_np(*load_item(*data))
#     meta_sci = Meta_sci(*load_item(*data))
#     meta_torch = Meta_torch(*load_item(*data))
#     meta = Meta(*load_item(*data))
    
#     gt, ud, ksp_acc, masks, sen  = meta.excite()
    
#     gt = abs(gt.data.numpy())
#     ud = abs(ud.data.numpy())
#     ksp_acc = abs(ksp_acc.data.numpy())
#     masks = abs(masks.data.numpy())

#     nsm = normalized_root_mse(gt, ud)
#     psnr = peak_signal_noise_ratio(gt, ud, data_range=gt.max())
#     ssim = structural_similarity(gt, ud, data_range=gt.max())
#     info = [[nsm,nsm], [ssim,ssim], [psnr,psnr]]
    
#     print(gt.shape)
#     print(ud.shape)
#     plot_inplace(gt, ud, ud, masks, info)