import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import NMSE, PSNR, SSIM, AVSNET, log
from dataset import Loader_knee
import os

rds_path = '/rds/projects/d/duanj-ai-in-medical-imaging/knee_nyu/coronal_pd'

log_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../results/experiments/avs/avs-15-dc"))
local_path = os.path.abspath(os.path.join(os.path.abspath(__file__),"../dataset/knee_nyu/coronal_pd"))
hugging_path = os.path.abspath(os.path.join(os.path.abspath(__file__),"../dataset/knee_fast_mri/knee/coronal_pd"))

writer = SummaryWriter(log_dir)
init_lr = 4e-3
lr_decay = .966
train_step = 0
test_step = 0
n_block = 15
hidden_dim = 512
ff_dim = 2048

try:
    data = Loader_knee(local_path)
    device = 'cpu'
except:
    try:
        data = Loader_knee(hugging_path)
        device = 'cuda'
    except:
        data = Loader_knee(rds_path)
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

loader = DataLoader(data, shuffle=True, batch_size=1, num_workers=0)
model = AVSNET(hidden_dim=hidden_dim, ff_dim=ff_dim, n_block=n_block).to(device)
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
mse = nn.MSELoss().to(device)

for t in range(300):
    loader.stage = 'train'
    model.train()
    for i, sample in enumerate(loader):
        gt, ud, ksp_acc, masks, sen = sample

        x = model(ud, ksp_acc, masks, sen)
        optimizer.zero_grad(set_to_none=True)
        loss = mse(torch.view_as_real(x), torch.view_as_real(gt))
        loss.backward()
        optimizer.step()
        
        log(gt, ud, x, masks, loss, writer,save_dir=log_dir, epoch=i, global_step=train_step, mode="train")
        train_step+=1
        print(t, i, loss.detach().cpu().numpy())
        
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    
    with torch.no_grad():
        loader.stage = 'val'
        model.eval()
        for i, sample in enumerate(loader):
            gt, ud, ksp_acc, masks, sen = sample
            x = model(ud, ksp_acc, masks, sen)
            loss = mse(torch.view_as_real(x), torch.view_as_real(gt))
            log(gt, ud, x, masks, loss, writer,save_dir=log_dir, epoch=i, global_step=test_step, mode="test")
            test_step+=1

    if t % 50 == 0 and t > 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        print('save the model at epoch {}'.format(t))
        model_dir = './saved/{}'.format("avs-tsboard")
        if not (os.path.exists(model_dir)):
            os.makedirs(model_dir)
        torch.save(
            checkpoint, "{0}/avs_{1:03d}.pth".format(model_dir, t))

    scheduler.step()