from . import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import smallvit

class NetA8C2(nn.Module):
    def __init__(self, out_features, channel=3):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channel, 20,7,1,2), nn.BatchNorm2d(20),  utils.Max2(), nn.MaxPool2d(2),
            nn.Conv2d(10, 40,2), nn.BatchNorm2d(40),  utils.Max2(), nn.MaxPool2d(2),
            nn.Conv2d(20, 60,2), nn.BatchNorm2d(60), utils.Max2(), utils.Interpolate((12,10))
            )
        self.p = nn.Sequential(
            nn.Conv2d( 30, 80,5,1,1), nn.BatchNorm2d(80),  utils.Max2(), nn.MaxPool2d(2), 
            nn.Conv2d( 40,120,2), nn.BatchNorm2d(120),  utils.Max2(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Linear(30*12*10, 200), nn.BatchNorm1d(200), utils.Max2())
        self.out = nn.Sequential(nn.Linear(100+60*2*1, out_features), nn.BatchNorm1d(out_features))
        self.feature_info = utils.FeatureInfo([30, 60, out_features], [4, 6, 1024])
    def forward(self, x, return_features=False):
        # x: 56x48
        x5 = self.net1(x)
        #print(x5.size()) #torch.Size([1, 30, 12, 10])
        y1 = self.p(x5)
        #print(y1.size()) #torch.Size([1, 30, 2, 1])
        y2 = self.c(x5.view(x.size(0), -1))
        y = self.out(torch.cat([y2,y1.view(x.size(0), -1)],1))
        if return_features:
            return [x5, y1, y]
        return y

class NetA8C2T(nn.Module):
    def __init__(self, out_features, channel=3):
        super().__init__()
        self.gnet = smallvit.Nano(128, channel=channel)
        self.net1 = nn.Sequential(
            nn.Conv2d(channel, 24,7,1,2), nn.BatchNorm2d(24),  utils.Max2(), nn.MaxPool2d(2),
            nn.Conv2d(12, 40,2), nn.BatchNorm2d(40),  utils.Max2(), nn.MaxPool2d(2),
            nn.Conv2d(20, 64,2), nn.BatchNorm2d(64), utils.Max2(), utils.Interpolate((12,10))
            )
        self.p = nn.Sequential(
            nn.Conv2d( 32, 80,5,1,1), nn.BatchNorm2d(80),  utils.Max2(), nn.MaxPool2d(2), 
            nn.Conv2d( 40,128,2), nn.BatchNorm2d(128),  utils.Max2(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Linear(32*12*10, 256), nn.BatchNorm1d(256), utils.Max2())
        self.out = nn.Sequential(nn.Linear(128+64*2*1+128, out_features), nn.BatchNorm1d(out_features))
        self.feature_info = utils.FeatureInfo([32, 64, out_features], [4, 6, 1024])
    def forward(self, x, return_features=False):
        # x: 56x48
        x5 = self.net1(x)
        # print(x5.size()) #torch.Size([1, 30, 12, 10])
        y1 = self.p(x5)
        #print(y1.size()) #torch.Size([1, 30, 2, 1])
        y2 = self.c(x5.view(x.size(0), -1))
        y3 = self.gnet(x)
        y = self.out(torch.cat([y2,y1.view(x.size(0), -1),y3],1))
        if return_features:
            return [x5, y1, y]
        return y

if __name__ == '__main__':
    net = utils.test_model(NetA8C2T, (3, 224, 192))
    torch.save(net.state_dict(), f"{net.__class__.__name__}.bin")
