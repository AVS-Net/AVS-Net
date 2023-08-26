import os
import numpy as np
import torch
from pprint import pprint
from scipy.io import loadmat
from torch.utils.data import DataLoader

try:
    from dataset import Meta
except:
    print("Warning using single file debug mode!")

class Loader_knee(DataLoader):
    '''
    NYU knee Meta dataset loader class embedded with Pytorch DataLoader.
    #### Member: path, and stage
    ##### path: full dataset folder path.
    ##### stage: current model stage, in ['train', 'val'].
    '''
    stage = "train" # Or "val"
    
    def __init__(self, path):
        self.path = get_pth(path)
        
    def __len__(self):
        return len(self.path['train'])

    def __getitem__(self, i):
        meta = Meta(*load_item(*self.path[self.stage][i]))
        return meta.excite()

def load_item(rpth, spth):
    '''
    Load raw and sen data from MatLab data file format. (Required Pytorch 1.9+ to support complex number)
    #### Original data shape: Height, Width, Channels (640, 368, 15) both for raw, and sen.
    ##### Produced data shape, CHW(15, 640, 368) for Meta(raw,sen).
    '''
    raw = np.complex64(loadmat(rpth)['rawdata']).transpose(2,0,1)
    sen = np.complex64(loadmat(spth)['sensitivities']).transpose(2,0,1)
    # raw = loadmat(rpth)['rawdata'].transpose(2,0,1)
    # sen = loadmat(spth)['sensitivities'].transpose(2,0,1)
    return raw, sen

def get_pth(path):
    '''
    ### Dataset: /rds/projects/d/duanj-ai-in-medical-imaging/knee_nyu
    #### Protocol: coronal_pd
    Dataset architecture:
    approximately 40 slices, each slice 15 channels, and HW (640 , 368)
    ```
    knee_nyu
        - axial_t2  coronal_pd(X)  coronal_pd_fs  sagittal_pd  sagittal_t2
            |           |           |               |           |
        - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] masks
                        |                                       |
        -            [train]                                  [val]
            |                                                            |
        - espirit*.mat(1-40), rawdata*.mat(1-40)                       *_masks.mat
    ```
    '''
    if not os.path.isdir(path):
        raise FileNotFoundError("Dataset not found!")

    path_list = [[[
        "{0}/{1}{2}.mat".format(os.path.join(path, str(f)), i, j) for i in ['rawdata', 'espirit'] ]
            for f in range(t+1,t+11) for j in range(11, 30) ] 
                for t in [0,10] ]
    
    return dict(zip(['train', 'val'], path_list))

# if __name__ == "__main__":
#     from data.utils import Meta
#     from figure import plot_inplace, plot_sen
#     data_path = os.path.abspath(os.path.join(os.path.abspath(__file__),"../knee/coronal_pd"))
#     path_dict = get_pth(data_path)
#     data = path_dict['train'][-1]
#     meta = Meta(*load_item(*data))
    
#     from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
    
#     gt, ud, ksp_acc, masks, sen  = meta.excite()
#     # print(gt)
#     gt = abs(gt.data.numpy())
#     ud = abs(ud.data.numpy())
#     ksp_acc = abs(ksp_acc.data.numpy())
#     masks = abs(masks.data.numpy())

#     # nsm = normalized_root_mse(gt, ud)
#     # psnr = peak_signal_noise_ratio(gt, ud, data_range=gt.max())
#     # ssim = structural_similarity(gt, ud, data_range=gt.max())
#     # info = [[nsm,nsm], [ssim,ssim], [psnr,psnr]]
    
#     print(gt.shape)
#     print(ud.shape)
#     print(masks.shape)
#     # plot_inplace(gt, ud, ud, masks, info)
#     # print(gt.shape)
#     # print(gt.shape, ud.shape, ksp_acc.shape, masks.shape, sen.shape)

#     # from figure import plot_mask, plot_kspace, plot_kspace_ifft, plot_kspace_ud, plot_gt, plot_ud, plot_raw_gt, plot_raw_ud, plot_sen
#     # plot_mask(masks)
#     # plot_kspace(meta.raw)
#     # plot_gt(gt)
#     # plot_ud(ud)
#     # plot_kspace_ud(ksp_acc)
#     plot_sen(sen)

