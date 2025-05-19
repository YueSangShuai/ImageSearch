from torch import nn
from torch.nn import functional as F
from .utils import FeatureInfo

class N0(nn.Module):
    def __init__(self, *args, channel=3, **kwargs) -> None:
        super().__init__()
        self.feature_info = FeatureInfo([16,16,16], [8, 16, 32])
        self.x1 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU())
        self.x2 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.x3 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=4, stride=4))
    def forward(self, x, return_features=True):
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        return x1,x2,x3

class M0(nn.Module):
    def __init__(self, *args, channel=3, **kwargs) -> None:
        super().__init__()
        self.feature_info = FeatureInfo([8,16,16,16], [4, 8, 16, 32])
        self.x0 = nn.Sequential(nn.Conv2d(channel, 8, kernel_size=4, stride=4), nn.BatchNorm2d(8), nn.ReLU())
        self.x1 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU())
        self.x2 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.x3 = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=8, stride=8), nn.BatchNorm2d(16), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=4, stride=4))
    def forward(self, x, return_features=True):
        x0 = self.x0(x)
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        return x0,x1,x2,x3
    
class N1(nn.Module):
    def __init__(self, *args, channel=3, **kwargs) -> None:
        super().__init__()
        self.feature_info = FeatureInfo([16,16,16], [8, 16, 32])
        self.x1 = nn.Conv2d(channel, 16, kernel_size=8, stride=8)
        self.x2 = nn.BatchNorm2d(16)
        self.x3 = nn.ReLU()
        self.x4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, groups=16, padding=1)
        self.x5 = nn.BatchNorm2d(16)
        self.x6 = nn.ReLU()
        self.x7 = nn.Conv2d(16, 16, kernel_size=3, stride=2, groups=16, padding=1)
        self.x8 = nn.BatchNorm2d(16)
        self.x9 = nn.ReLU()
    def forward(self, x, return_featuremaps=True):
        x = self.x1(x)
        x = self.x2(x)
        x1 = self.x3(x)
        x = self.x4(x1)
        x = self.x5(x)
        x2 = self.x6(x)
        x = self.x7(x2)
        x = self.x8(x)
        x3 = self.x9(x)
        return x1,x2,x3
