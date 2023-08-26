
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        cx = x.permute(0, 2, 3, 1)
        return cx

class Conv_AMP(nn.Module):
    def __init__(self):
        super(Conv_AMP, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  kernel_size=3, padding=1, bias=True)
        )
        
    def forward(self, x):
        x = torch.view_as_real(x).permute(0, 3, 1, 2)
        x = self.conv(x)
        cx = x.permute(0, 2, 3, 1)
        return cx