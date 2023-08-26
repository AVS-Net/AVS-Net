from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

def NMSE(gt, rec):
    try:
        gt, rec = gt.numpy(), rec.numpy()
    except:
        pass
    return normalized_root_mse(gt, rec)

def PSNR(gt, rec):
    try:
        gt, rec = gt.numpy(), rec.numpy()
    except:
        pass
    return peak_signal_noise_ratio(gt, rec, data_range=gt.max())

def SSIM(gt, rec):
    try:
        gt, rec = gt.numpy(), rec.numpy()
    except:
        pass
    return structural_similarity(gt, rec, data_range=gt.max())