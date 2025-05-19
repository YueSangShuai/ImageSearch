import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .utils import FeatureInfo
import os

# 从环境变量中读取模型明和预训练模型路径

class A(nn.Module):
    def __init__(self, out_features, channel=3, **kwargs):
        super().__init__()
        model_name = os.environ.get('MODEL_NAME', 'mixnet_s')
        pretrained_path = os.environ.get('PRETRAINED_PATH', None)
        pretrained = os.environ.get('PRETRAINED', None)
        model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=pretrained_path, 
                num_classes=out_features, in_chans=channel, **kwargs)
        self.net = model
        self.feature_info = FeatureInfo([model.num_features, out_features], [32, 1024])
    def forward(self, x, return_features=False):
        x = self.net.forward_features(x)
#        print(x.shape)  # torch.Size([1, 1536, 7, 7])
        y = self.net.global_pool(x)
        y = torch.flatten(y, 1)
        y = self.net.classifier(y)
        if return_features:
            return [x, y]
        return y
    
class B(nn.Module):
    def __init__(self, out_features, channel=3, **kwargs):
        super().__init__()
        model_name = os.environ.get('MODEL_NAME', 'mixnet_s')
        pretrained_path = os.environ.get('PRETRAINED_PATH', None)
        pretrained = os.environ.get('PRETRAINED', None)
        model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=pretrained_path, 
                    features_only=True, in_chans=channel, **kwargs)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(model.feature_info.channels()[-1], out_features)
        self.net = model
        self.feature_info = FeatureInfo(model.feature_info.channels()+[out_features], 
                                        model.feature_info.reduction()+[1024])
    def forward(self, x, return_features=False):
        y = self.net(x)
        x = self.global_pool(y[-1])
        x = self.classifier(x)
        y.append(x)
        if return_features: 
            return y
        return x
    
if __name__ == '__main__':
    model = B(100)
    x = torch.randn(1, 3, 224, 224)
    y = model(x, return_features=True)
    for y_ in y: 
        print(y_.shape)
    print(model.feature_info)