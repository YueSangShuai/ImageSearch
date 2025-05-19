import torch

dinov2_dims = {
    'vits14': 384,
    'vitb14': 768,
    'vitl14': 1024,
    'vitg14': 1536,
}

class DinoV2(torch.nn.Module):
    def __init__(self, out_features=512, name="vits14", freeze=False, **kwargs):
        super().__init__()
        from .utils import FeatureInfo
        self.net = torch.hub.load('facebookresearch/dinov2', 'dinov2_'+name, source='local')
        if freeze:
            for p in self.net.parameters(): p.requires_grad=False
        self.freeze = freeze
        self.head = torch.nn.Linear(dinov2_dims[name], out_features)
        self.feature_info  = FeatureInfo([dinov2_dims[name], out_features], [1024, 1024])
    def train(self, mode=True):
        self.net.eval() if self.freeze else self.net.train(mode)
        self.head.train(mode)
    def forward(self, x, return_featuremaps=False):
        x = self.net(x)
        y = self.head(x)
        if return_featuremaps: return [x, y]
        return y
    
#dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

class S14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vits14", **kwargs)

class FS14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vits14", freeze=True, **kwargs)

class B14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vitb14", **kwargs)
    def forward(self, x, return_featuremaps=False):
        ret1 = super().forward(x, return_featuremaps)
        return ret1

class FB14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vitb14", freeze=True, **kwargs)

class L14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vitl14", **kwargs)        

class FL14(DinoV2):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, name="vitl14", freeze=True, **kwargs)        


def load_from_onnx(net, onnx_file, fc_convert={}):
    import onnx
    o = onnx.load(onnx_file)
    state = {}
    for i in range(len(o.graph.initializer)):
        name = o.graph.initializer[i].name
        state[name] = torch.from_numpy(onnx.numpy_helper.to_array(o.graph.initializer[i])[:])
#        print(name, state[name].shape)
    used = []
    m_state = net.state_dict()
    for k in m_state:
        if k in fc_convert:
            k0 = fc_convert[k]
            if k0:
                m_state[k] = state[k0].t()
#                print(f"loaded {k0} -> {k}")
                used.append(k0)
        elif k in state:
            m_state[k][:] = state[k][:]
#            print(f"loaded {k}")
            used.append(k)
        else:
            k0 = k.replace('net.', '')
            if k0 in state:
                if m_state[k].shape != state[k0].shape:
                    print(f"mismatch {k}: {m_state[k].shape} != {state[k0].shape}")
                else:
                    m_state[k][:] = state[k0][:]
#                print(f"loaded {k0} -> {k}")
                used.append(k0)
            else:
                print(f"missing {k}: {m_state[k].shape}")
    for k in used: del state[k]
    print(f"unused: {list(state.keys())}")
    net.load_state_dict(m_state, strict=False)

if __name__=="__main__":
    net = B14(2)
    net.eval()
    load_from_onnx(net, 'fas-dinov2.onnx', {
        "net.mask_token": "", #torch.Size([1 768])
        "net.blocks.0.attn.qkv.weight": "1187", #torch.Size([2304, 768])
        "net.blocks.0.attn.proj.weight": "1198", #torch.Size([768, 768])
        "net.blocks.0.mlp.fc1.weight": "1199", #torch.Size([3072, 768])
        "net.blocks.0.mlp.fc2.weight": "1200", #torch.Size([768, 3072])
        "net.blocks.1.attn.qkv.weight": "1201", #torch.Size([2304, 768])
        "net.blocks.1.attn.proj.weight": "1212", #torch.Size([768, 768])
        "net.blocks.1.mlp.fc1.weight": "1213", #torch.Size([3072, 768])
        "net.blocks.1.mlp.fc2.weight": "1214", #torch.Size([768, 3072])
        "net.blocks.2.attn.qkv.weight": "1215", #torch.Size([2304, 768])
        "net.blocks.2.attn.proj.weight": "1226", #torch.Size([768, 768])
        "net.blocks.2.mlp.fc1.weight": "1227", #torch.Size([3072, 768])
        "net.blocks.2.mlp.fc2.weight": "1228", #torch.Size([768, 3072])
        "net.blocks.3.attn.qkv.weight": "1229", #torch.Size([2304, 768])
        "net.blocks.3.attn.proj.weight": "1240", #torch.Size([768, 768])
        "net.blocks.3.mlp.fc1.weight": "1241", #torch.Size([3072, 768])
        "net.blocks.3.mlp.fc2.weight": "1242", #torch.Size([768, 3072])
        "net.blocks.4.attn.qkv.weight": "1243", #torch.Size([2304, 768])
        "net.blocks.4.attn.proj.weight": "1254", #torch.Size([768, 768])
        "net.blocks.4.mlp.fc1.weight": "1255", #torch.Size([3072, 768])
        "net.blocks.4.mlp.fc2.weight": "1256", #torch.Size([768, 3072])
        "net.blocks.5.attn.qkv.weight": "1257", #torch.Size([2304, 768])
        "net.blocks.5.attn.proj.weight": "1268", #torch.Size([768, 768])
        "net.blocks.5.mlp.fc1.weight": "1269", #torch.Size([3072, 768])
        "net.blocks.5.mlp.fc2.weight": "1270", #torch.Size([768, 3072])
        "net.blocks.6.attn.qkv.weight": "1271", #torch.Size([2304, 768])
        "net.blocks.6.attn.proj.weight": "1282", #torch.Size([768, 768])
        "net.blocks.6.mlp.fc1.weight": "1283", #torch.Size([3072, 768])
        "net.blocks.6.mlp.fc2.weight": "1284", #torch.Size([768, 3072])
        "net.blocks.7.attn.qkv.weight": "1285", #torch.Size([2304, 768])
        "net.blocks.7.attn.proj.weight": "1296", #torch.Size([768, 768])
        "net.blocks.7.mlp.fc1.weight": "1297", #torch.Size([3072, 768])
        "net.blocks.7.mlp.fc2.weight": "1298", #torch.Size([768, 3072])
        "net.blocks.8.attn.qkv.weight": "1299", #torch.Size([2304, 768])
        "net.blocks.8.attn.proj.weight": "1310", #torch.Size([768, 768])
        "net.blocks.8.mlp.fc1.weight": "1311", #torch.Size([3072, 768])
        "net.blocks.8.mlp.fc2.weight": "1312", #torch.Size([768, 3072])
        "net.blocks.9.attn.qkv.weight": "1313", #torch.Size([2304, 768])
        "net.blocks.9.attn.proj.weight": "1324", #torch.Size([768, 768])
        "net.blocks.9.mlp.fc1.weight": "1325", #torch.Size([3072, 768])
        "net.blocks.9.mlp.fc2.weight": "1326", #torch.Size([768, 3072])
        "net.blocks.10.attn.qkv.weight": "1327", #torch.Size([2304, 768])
        "net.blocks.10.attn.proj.weight": "1338", #torch.Size([768, 768])
        "net.blocks.10.mlp.fc1.weight": "1339", #torch.Size([3072, 768])
        "net.blocks.10.mlp.fc2.weight": "1340", #torch.Size([768, 3072])
        "net.blocks.11.attn.qkv.weight": "1341", #torch.Size([2304, 768])
        "net.blocks.11.attn.proj.weight": "1352", #torch.Size([768, 768])
        "net.blocks.11.mlp.fc1.weight": "1353", #torch.Size([3072, 768])
        "net.blocks.11.mlp.fc2.weight": "1354", #torch.Size([768, 3072])        
    })
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    mb = round(sum([p.nelement() for p in net.parameters()])*4/(1024*1024))
    print(f'output: {y.shape}, parameters: {mb} MB')
    torch.save(net.state_dict(), 'fas-dinov2.pth')
    
