import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import re
from collections import abc
import numpy as np
import functools
from inspect import getfullargspec
from random import random
from itertools import repeat
import collections.abc
import math, os
import torch.distributed as dist
from einops import rearrange, pack, unpack
from torch.autograd import Function
import warnings
import requests

class RIConv2d(nn.Conv2d):
    """
    旋转不变卷积
    """
    def __init__(self, *args, **kwargs):
        super(RIConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.kernel_size == (1, 1):
            # print(self.kernel_size)
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        weight = torch.stack([
            torch.rot90(self.weight, 0, [2, 3]),
            torch.rot90(self.weight, 1, [2, 3]),
            torch.rot90(self.weight, 2, [2, 3]),
            torch.rot90(self.weight, 3, [2, 3]),
        ], dim=1)
        weight = weight.view(weight.size(0)*weight.size(1),
                             weight.size(2), weight.size(3), weight.size(4))
        bias = self.bias
        if bias is not None:
            bias = torch.stack([
                bias, bias, bias, bias
            ], dim=1)
            bias = bias.view(bias.size(0)*bias.size(1))

        out = F.conv2d(input, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)
        out, _ = out.view(out.size(0), out.size(
            1)//4, 4, out.size(2), out.size(3)).max(dim=2)
        return out

class CSymConv2d(nn.Conv2d):
    """
    中心对称卷积, 180度旋转不变
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        if self.kernel_size == (1, 1):
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        weight = (self.weight + torch.rot90(self.weight, 2, [2, 3]))/2
        bias = self.bias
        out = F.conv2d(input, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out
    def __repr__(self):
        return f'{self.__class__.__name__}({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups})'

class RICSymConv2d(nn.Conv2d):
    """
    中心对称卷积, 旋转不变
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        if self.kernel_size == (1, 1):
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        weight = (self.weight + torch.rot90(self.weight, 2, [2, 3]))/2
        weight2 = torch.rot90(weight, 1, [2, 3])
        weight = torch.cat([weight, weight2], dim=0)
        bias = self.bias
        if bias is not None:
            bias = torch.cat([bias, bias], dim=0)
        out = F.conv2d(input, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)
        shape = out.shape
        shape2 = list(shape)
        shape2[1]=shape2[1]//2
        out = out.view(shape[0], 2, -1)
        out = out.max(dim=1)[0].view(shape2)
        return out
    def __repr__(self):
        return f'{self.__class__.__name__}({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups})'

class SymConv2d(nn.Conv2d):
    """
    中心对称卷积, 旋转不变
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        if self.kernel_size == (1, 1):
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        weight = (self.weight + torch.rot90(self.weight, 2, [2, 3]))/2
        weight2 = torch.rot90(weight, 1, [2, 3])
        weight = torch.cat([weight, weight2], dim=0)
        bias = self.bias
        if bias is not None:
            bias = torch.cat([bias, bias], dim=0)
        out = F.conv2d(input, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)
        shape = out.shape
        shape2 = list(shape)
        shape2[1]=shape2[1]//2
        out = out.view(shape[0], 2, -1)
        out = out.max(dim=1)[0].view(shape2)
        return out
    def __repr__(self):
        return f'{self.__class__.__name__}({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups})'


class MaxMin(nn.Module):
    def __init__(self, *args):
        super(MaxMin, self).__init__()

    def forward(self, input):
        c = input.size(1)//2
        i1 = input[:, :c]
        i2 = input[:, c:c*2]
        x1 = torch.max(i1, i2)
        x2 = torch.min(i1, i2)
        if c*2 < input.size(1):
            x = torch.cat([x1, x2, F.relu(input[:, -1:])], dim=1)
        else:
            x = torch.cat([x1, x2], dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__+"()"


class Max2(nn.Module):
    def forward(self, input):
        c = input.size(1)//2
        i1 = input[:, :c]
        i2 = input[:, c:c*2]
        x1 = torch.max(i1, i2)
        return x1
    def __repr__(self):
        return self.__class__.__name__+"()"


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(
            in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x

class ActIdentity(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
    def forward(self, x):
        return x

class GELU(nn.GELU):
    def __init__(self, channel=1):
        super().__init__()

class SiLU(nn.SiLU):
    def __init__(self, channel=1):
        super().__init__()

class Ident(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args 
        self.kwargs = kwargs
    def forward(self, x):
        print(self.args, self.kwargs, x.shape)
        return x
        
class ReLU(nn.ReLU):
    def __init__(self, channel):
        super().__init__()

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, channel=None, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, channel=None, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

def try_load_state(model, state_fn, must_exist=False):
    if not must_exist and (type(state_fn)==str) and not os.path.exists(state_fn):
        print('File %s does not exist.' % state_fn)
        return
    if type(state_fn)==str:
        checkpoint = torch.load(state_fn, map_location=lambda storage, loc: storage, weights_only=True)
    else:
        checkpoint = state_fn
    if 'net_state_dict' in checkpoint: checkpoint=checkpoint['net_state_dict']
    if 'state_dict' in checkpoint: checkpoint=checkpoint['state_dict']
    if 'state_dicts' in checkpoint: checkpoint=checkpoint['state_dicts']
    if 'model' in checkpoint and (len(checkpoint.keys())==1): checkpoint=checkpoint['model']
    try:
        model.load_state_dict(checkpoint)
        return 1
    except Exception as e:
        print(e)
        
    state = model.state_dict()
#    print(state.keys(), checkpoint)
    i = 0
    checkpoint2 = {}
    for k in checkpoint:
        k2 = re.sub(r"_(\d)", r".\1", k)
        if k2==k:
            if not k.startswith('backbone.'): continue 
            k2=k[9:]
        checkpoint2[k2]=checkpoint[k]
    for k in state.keys():
        k0 = k
        if k not in checkpoint and k not in checkpoint2:
            k2 = '_orig_mod.'+k
            if k2 in checkpoint: k=k2
            elif k.startswith('module.'): k=k[7:]
            elif k.startswith('feature.'): k=k[8:]
        v = None
        if k in checkpoint: v = checkpoint[k]
        elif k in checkpoint2: v = checkpoint2[k]
        elif 'module.'+k in checkpoint: v = checkpoint['module.'+k]
        if v is not None:
            if state[k0].shape: 
                s = v.shape[0]
                try:
                    if state[k0].shape==v.shape:
                        state[k0]=v
                    elif s>state[k0].shape[0]:
                        state[k0][:]=v[:state[k0].shape[0]]
                        print(f'Warning {k0}: {s} is truncated as {state[k0].shape[0]}')
                    elif s==state[k0].shape[0] and len(state[k0].shape)==len(v.shape)>2:  #具有相同输出通道但不同输入通道的情况
                        if state[k0].shape[1]==1: # 多通道累加到单通道
                            state[k0]=v.sum(1, keepdim=True)
                            print(f'Warning {k0}: {v.shape} is averaged as {state[k0].shape}')
                        elif v.shape[1]==1: # 单通道扩展到多通道
                            state[k0][:]=v[:,[0,]*state[k0].shape[1]]
                            print(f'Warning {k0}: {v.shape} is expanded as {state[k0].shape}')
                        else:
                            state[k0][:s]=v
                            print(f'Warning {k0}: {v.shape} is full as {s}')
                    else:
                        state[k0][:s]=v
                        print(f'Warning {k0}: {v.shape} is full as {s}')
                except:
                    if v.nelement()==state[k0].nelement():
                        if s==state[k0].shape[1] and state[k0].shape[0]==v.shape[1]:
                            state[k0]=v.transpose(0,1)
                            print(f"Warning {k0}: transpose: {v.shape} as {state[k0].shape}")
                        else:
                            state[k0]=v.view(state[k0].shape)
#                            print(f"Warning {k0}: reshape {v.shape} as {state[k0].shape}")
                    else:
                        print(f"Warning: ignore parameters {k0}. because {v.shape}->{state[k0].shape} failed.")
            else:
                state[k0] = v
            i+=1
        else:
            print("ignore parameter:", k)
    model.load_state_dict(state)
    return i


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    world_size = dist.get_world_size()
    t = t.contiguous()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

def has_only_one_value(t):
    return (t == t[0]).all()

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

class SplitByRank(Function):
    @staticmethod
    def forward(ctx, x):
        rank = dist.get_rank()
        return x[rank]

    @staticmethod
    def backward(ctx, grads):
        grads = rearrange(grads, '... -> 1 ...')
        grads = all_gather_variable_dim(grads)
        return grads

split_by_rank = SplitByRank.apply

def test_model(net, sizes=[3,224,224], fd=100):    
    import time
    model = net(fd)
    model.eval()
    print(model)
    if hasattr(model, 'channel'): sizes[0]=model.channel
    if hasattr(model, 'size'): 
        if type(model.size)==int: 
            sizes[1], sizes[2]=model.size, model.size
        else: sizes[1:]=model.size
    B = 3
    test_data = torch.rand(B, *sizes)
    print('input size:', test_data.size())
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(test_data, ), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print("param:", params, "flops:", flops/B) #  param: 2.542M flops: 102.048M
    except: pass
    if hasattr(model, "feature_info"): print("channels:", model.feature_info.channels(), "strides:", model.feature_info.reduction())
    for i in range(1):
        t = time.time()
        test_outputs = model(test_data, True)  # , test_data_2]
        t2 = time.time()
        print('time(ms):', 100*(t2 - t))
        if hasattr(test_outputs, "items"):
            for o, v in test_outputs.items():
                print('output size:', o, v.shape)
        elif type(test_outputs) in (list, tuple):
            for i, v in enumerate(test_outputs):
                print('output size:', i, v.shape)
        else:
            print('output size:', test_outputs.shape)
    return model

class FeatureInfo(object):
    def __init__(self, channels, reductions) -> None:
        super().__init__()
        self.channels_ = channels
        self.reductions_ = reductions
    def channels(self):
        return self.channels_
    def reduction(self):
        return self.reductions_
    def __str__(self) -> str:
        return f"channels: {self.channels()}, strides: {self.reduction()}"



def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs

def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.
    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored.
    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.
    :Example:
        class MyModule1(nn.Module)
            # Convert x and y to fp32
            @force_fp32()
            def loss(self, x, y):
                pass
        class MyModule2(nn.Module):
            # convert pred to fp32
            @force_fp32(apply_to=('pred', ))
            def post_process(self, pred, others):
                pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper

class DropPath(nn.Module):
    def __init__(self, module, keep_prob=0.6):
        super(DropPath, self).__init__()
        self.module = module
        self.keep_prob = keep_prob
        self.shape = None
        self.training = True
        self.dtype = torch.FloatTensor

    def forward(self, *input):
        if self.training:
            # If we don't now the shape we run the forward path once and store the output shape
            if self.shape is None:
                temp = self.module(*input)
                self.shape = temp.size()
                if temp.data.is_cuda:
                    self.dtype = torch.cuda.FloatTensor
                del temp
            p = random()
            if p <= self.keep_prob:
                return self.dtype(self.shape).zero_()
            else:
                return self.module(*input)/self.keep_prob # Inverted scaling
        else:
            return self.module(*input)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath_new(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = nn.AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = norm_cfg(embed_dims)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.size = size
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=nn.ReLU, norm_layer=None):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

SqueezeExcite = SEModule  # alias



def download_from_url(url, path=None, root='.data', overwrite=False):


    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            print('File %s already exists.' % path)
            if not overwrite:
                return path
            print('Overwriting file %s.' % path)
        print('Downloading file {} to {} ...'.format(filename, path))

        with open(path, "wb") as file:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)
        print('File {} downloaded.'.format(path))
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        raise RuntimeError(
            "Download directory {} does not exist. "
            "Did you create it?".format(root))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Backbone Checkor')
    parser.add_argument('backbone', help="backbone name")
    parser.add_argument('--image-size', type=int, nargs="+",
                        default=[224, 192], help='切图后的缩放大小。通常是网络的输入尺寸, default=224')
    parser.add_argument('--feature-dim', default=256, type=int, help="特征维度")

    args = parser.parse_args()
    if len(args.image_size)==1: args.image_size=[args.image_size[0], args.image_size[0]]
    if len(args.image_size)==2: args.image_size=[3, args.image_size[0], args.image_size[1]]
    return args

if __name__=="__main__":
    args = get_args()
    import utils
    import os
    m=args.backbone
    print("test model: ", m)
    try:
        net = utils.import_var(m)
    except:
        net = utils.import_var("backbone."+m)
    model = test_model(net, args.image_size, args.feature_dim)
    if hasattr(model, "export"): model.export()
    fn = f"ftr100model/{m}.pth"
    torch.save(model.state_dict(), fn)
    print(f"Saved to {fn}")
    os.system(f"ls -lh {fn}")
