import torch
from torch import nn
from .utils import FeatureInfo, FReLU
import torch.nn.functional as F


def at(x):
    return F.normalize(x.pow(2).mean(1))

class target(nn.Module): 
    def __init__(self, feat_type='attention'):
        super(target, self).__init__()
        self.feat_type = feat_type
        
    def forward(self, CA, SA=None):
        if self.feat_type == 'attention':
            assert SA is not None
            return [CA, SA]
        elif self.feat_type == 'self_attention':
            assert SA is None
            return at(CA)
        elif self.feat_type == 'feature':
            assert SA is None
            return CA
        else:
            raise('Select Proper Target')
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CAModule(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        c = channels // reduction
        if c<1: c = 1
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, c, kernel_size=1, padding=0, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(c, channels, kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out

class SAModule(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out


class BottleNeck_IR_CBAM(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, frelu=False):
        super(BottleNeck_IR_CBAM, self).__init__()
        dim_match = (in_channel == out_channel) and (stride == 1)
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel) if not frelu else FReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))

        self.channel_layer = CAModule(out_channel, 16)
        self.spatial_layer = SAModule()
        self.attention_target = target(feat_type='attention')

        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )


    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        # Target for A-SKD
        res, att_c = self.channel_layer(res)
        res, att_s = self.spatial_layer(res)
        _ = self.attention_target(att_c, att_s)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
            
        return shortcut + res


filter_list = [64, 64, 128, 256, 512]
filter_list2 = [64, 64, 128, 256, 768]
def get_layers(num_layers):
    if num_layers == 34:
        return [1, 2, 8, 3]
    elif num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 101:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]

class CBAMResNet(nn.Module):
    def __init__(self, out_features=512, layers=[3, 4, 14, 3], drop_ratio=0.4, steam_stride=1, 
        filter_list=filter_list, fmap_size=7, frelu=False, channel=3, fc_pool='linear'):
        super().__init__()
        self.filter_list = filter_list
        block = BottleNeck_IR_CBAM
        self.channel = channel
        k = max(3, steam_stride)
        self.input_layer = nn.Sequential(nn.Conv2d(channel, filter_list[0], k, stride=steam_stride, padding=1 if k!=steam_stride else 0, bias=False),
                                         nn.BatchNorm2d(filter_list[0]),
                                         nn.PReLU(filter_list[0]) if not frelu else FReLU(filter_list[0]))

        self.feature_target = target(feat_type='feature')
        self.self_attention_target = target(feat_type='self_attention')
        
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2, frelu=frelu)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2, frelu=frelu)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2, frelu=frelu)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2, frelu=frelu)
        self.addition = None
        self.fmap_size = fmap_size
        self.output_layer = nn.Identity()
        if out_features>0:
            if fc_pool == 'linear':
                self.fmap_size = fmap_size
                self.output_layer = nn.Sequential(
                    nn.BatchNorm2d(filter_list[4]),
                    nn.Dropout(drop_ratio),
                    Flatten(),
                    nn.Linear(filter_list[4] * fmap_size * fmap_size, out_features),
                    nn.BatchNorm1d(out_features))
            elif fc_pool == 'none':
                self.fmap_size = 0
                self.output_layer = nn.Sequential(
                    nn.BatchNorm2d(filter_list[4]),
                    nn.PReLU(filter_list[4]),
                    nn.Conv2d(filter_list[4], out_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_features),
                )
                self.addition = nn.Flatten()
            else:
                self.fmap_size = 0
                self.output_layer = nn.Sequential(
                    nn.BatchNorm2d(filter_list[4]),
                    nn.AdaptiveMaxPool2d(1) if fc_pool=='max' else nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    nn.Linear(filter_list[4], out_features),
                    nn.BatchNorm1d(out_features)
                )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.feature_info  = FeatureInfo(filter_list[1:]+[out_features], [c*steam_stride for c in [2,4,8,16,1024]])
    def _make_layer(self, block, in_channel, out_channel, blocks, stride, frelu=False):
        layers = []
        layers.append(block(in_channel, out_channel, stride, frelu))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, frelu))
        return nn.Sequential(*layers)

    def forward(self, x, return_featuremaps=False):
        b,c,h,w = x.shape
        if self.channel == 1 and c == 3:
            x = x.mean(dim=1, keepdim=True)
        elif self.channel == 3 and c == 1:
            x = x.repeat(1,3,1,1)
        x = self.input_layer(x)
        x1 = self.layer1(x)
        _ = self.self_attention_target(x1)

        x2 = self.layer2(x1)
        _ = self.self_attention_target(x2)
        
        x3 = self.layer3(x2)
        _ = self.self_attention_target(x3)
        
        x4 = self.layer4(x3)
        _ = self.self_attention_target(x4)

        y = [x1, x2, x3, x4]
        if self.fmap_size>0: x4 = F.interpolate(x4, self.fmap_size)
        out = self.output_layer(x4)
        _ = self.feature_target(out)
        y.append(out)
        if self.addition is not None:
            out = self.addition(out)
            y.append(out)
        if return_featuremaps:
            return y
        return out

class CBAMResNet34O1(CBAMResNet):
    # size: 2.6M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), steam_stride=2, drop_ratio=0, 
            filter_list=[x//8 for x in filter_list], frelu=False, fmap_size=4, **kwargs)

class CBAMResNet34Q1(CBAMResNet):
    # param: 1.654M flops: 256.719M, size: 6.5M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), steam_stride=2, drop_ratio=0, 
            filter_list=[x//4 for x in filter_list], frelu=False, fmap_size=4, **kwargs)

class CBAMResNet34Q2(CBAMResNet):
    # param: 1.654M flops: 126.008M, size: 6.5M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), steam_stride=3, drop_ratio=0, 
            filter_list=[x//4 for x in filter_list], frelu=False, fmap_size=4, **kwargs)

class GrayCBAMResNet34Q2(CBAMResNet34Q2):
    # param: 1.654M flops: 126.008M, size: 6.5M
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, channel=c, **kwargs)

class CBAMResNet34Q(CBAMResNet):
    # param: 2.076M flops: 1.026G, size: 6.5M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), drop_ratio=0.1, 
            filter_list=[x//4 for x in filter_list], frelu=False, **kwargs)

class GrayCBAMResNet34Q(CBAMResNet):
    # param: 2.076M flops: 1.026G, size: 6.5M
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, layers=get_layers(34), drop_ratio=0.1, 
            filter_list=[x//4 for x in filter_list], frelu=False, channel=c, **kwargs)
        
class CBAMResNet34QF(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), drop_ratio=0.1, 
            filter_list=[x//4 for x in filter_list], frelu=True, **kwargs)

class CBAMResNet34(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(34), drop_ratio=0.1, 
            filter_list=filter_list, frelu=True, **kwargs)

class GrayCBAMResNet34X(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, layers=get_layers(34),  
            filter_list=filter_list, frelu=False, fmap_size=0, channel=1, **kwargs)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.filter_list[4]),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(self.filter_list[4], out_features),
            nn.BatchNorm1d(out_features)
        )

class GrayCBAMResNet34Q2X(GrayCBAMResNet34Q2):
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, fc_pool='max', channel=c, **kwargs)

class GrayCBAMResNet34A(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, layers=get_layers(34),  
            filter_list=filter_list, frelu=False, fc_pool='avg', channel=c, **kwargs)

class GrayCBAMResNet34Q2A(GrayCBAMResNet34Q2):
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, fc_pool='avg', channel=c, **kwargs)

class GrayCBAMResNet34QA(GrayCBAMResNet34Q):
    def __init__(self, out_features=512, **kwargs):
        c = kwargs.pop('channel', 1)
        super().__init__(out_features=out_features, fc_pool='avg', channel=c, **kwargs)

class GrayCBAMResNet34Q2X(CBAMResNet34Q2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, channel=1, **kwargs)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.filter_list[4]),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(self.filter_list[4], out_features),
            nn.BatchNorm1d(out_features)
        )
class CBAMResNet50Q(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(50), drop_ratio=0.1, 
            filter_list=[x//4 for x in filter_list], **kwargs)

class CBAMResNet50QF(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(50), drop_ratio=0.1, 
            filter_list=[x//4 for x in filter_list], frelu=True, **kwargs)

class CBAMResNet50H(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(50), drop_ratio=0.2, 
            filter_list=[x//2 for x in filter_list], **kwargs)

class CBAMResNet50H1(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(50), drop_ratio=0.2, 
            filter_list=[x//2 for x in filter_list], steam_stride=2, **kwargs)

class CBAMResNet101(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(101), drop_ratio=0.4, filter_list=filter_list, **kwargs)

class CBAMResNet152(CBAMResNet):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=get_layers(152), drop_ratio=0.4, filter_list=filter_list, **kwargs)

class CBAMResNetS(CBAMResNet):
    # 122M
    def __init__(self, out_features=512, layers=get_layers(50), filter_list=filter_list, drop_ratio = 0.2, **kwargs):
        filters = filter_list
        super().__init__(out_features=out_features, layers=layers, drop_ratio=drop_ratio, filter_list=filters, **kwargs)
        f = filters[4]
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(f),
            nn.Conv2d(f, f, 3, 2, 1, groups=f), 
            nn.BatchNorm2d(f), 
            nn.PReLU(f),
            nn.Conv2d(f, f, 2, 2, 0, groups=f), 
            nn.BatchNorm2d(f), 
            nn.PReLU(f),
            nn.Dropout(drop_ratio),
            nn.Conv2d(f, out_features, 2, 2), 
            nn.BatchNorm2d(out_features), 
            Flatten(),
            )

class CBAMResNetSA(CBAMResNetS):
    # 36M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=[14, 7, 2, 1], filter_list=[64, 64, 128, 256, 512], **kwargs)

class CBAMResNetS2(CBAMResNetS):
    # 32M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, layers=[8, 14, 5, 1], filter_list=[32, 48, 96, 192, 384], **kwargs)

if __name__=="__main__":
    from . import utils
    net = utils.test_model(CBAMResNet34Q, [3,112,112])
    torch.save(net.state_dict(), f"{net.__class__.__name__}.bin")
