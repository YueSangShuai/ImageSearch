import torch
from torch import nn
from torch.nn import functional as F
from .utils import FeatureInfo

#from .utils import FeatureInfo, FReLU, GELU, ReLU, test_model

class HTBase(nn.Module):
    def __init__(self, out_features=512, model="hiera_base_224", pretrained=True, checkpoint="mae_in1k", source='local') -> None:
        super().__init__()
        self.model = torch.hub.load("facebookresearch/hiera", model=model, pretrained=pretrained, checkpoint=checkpoint, source=source)
        self.model.head.projection = nn.Linear(768, out_features)
        self.model.head.act_func = nn.Identity()
        self.feature_info  = FeatureInfo([96, 192, 384, 768, out_features], [4, 8, 16, 32, 1024])
    def forward(self, x, return_features=False):
        ret, intermediates = self.model(x, return_intermediates=True)
        if return_features:
            return [i.permute(0, 3, 1, 2) for i in intermediates]+[ret]
        return ret

class HTTiny(HTBase):
    def __init__(self, out_features=512, model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k", source='local') -> None:
        super().__init__(out_features=out_features, model=model, pretrained=pretrained, checkpoint=checkpoint, source=source)
    
