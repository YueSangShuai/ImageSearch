import logging
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
import os
import torch.nn as nn
import torch.optim as torch_optim
import torch
from utils import logging_info

def prepare_optimizer(args, model):
    if args.train_modules:
        for n, x in model.named_parameters():    
            requires_grad = False
            for k in args.train_modules:
                if n.startswith(k): 
                    requires_grad = True
                    break
            if not requires_grad:
                x.requires_grad = False
                logging_info('Freeze module: ' + n)
    optimizing_parameters=[p for p in model.parameters() if p.requires_grad]

    n = 0
    for p in optimizing_parameters: n+=p.numel()
    logging_info('    Total trainable params: %.4fM' % (n / 1024./1024.))
    schedule_free = False
    if args.optim.lower() == 'adam':
        optimizer = torch_optim.Adam(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adamw':
        optimizer = torch_optim.AdamW(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'cadamw':
        from optim.cadamw import AdamW
        optimizer = AdamW(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'clion':
        from optim.clion import Lion
        optimizer = Lion(optimizing_parameters,
                        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adan':
        from optim.adan import Adan
        optimizer = Adan(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay, betas=(0.98, 0.92, 0.99), eps=1e-8,
                            max_grad_norm=args.gradient_clip_max, no_prox=False, foreach=True)
    elif args.optim.lower() == 'lion':
        from optim.lion import Lion
        optimizer = Lion(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'sgd':
        optimizer = torch_optim.SGD(optimizing_parameters, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'schedulefree_adamw':
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(optimizing_parameters,
                            lr=args.lr, weight_decay=args.weight_decay)
        schedule_free = True
    elif args.optim.lower() == 'schedulefree_sgd':
        from schedulefree import SGDScheduleFree
        optimizer = SGDScheduleFree(optimizing_parameters, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
        schedule_free = True
    else:
        raise NotImplementedError()
    logging_info(f'optimizer: {optimizer}')
    div_factor = 1e4
    if schedule_free:
        scheduler_warmup = ConstantLR(optimizer)
    else:
        scheduler_warmup = OneCycleLR(optimizer, max_lr=args.lr,
                                  pct_start=args.lr_warmup_step/args.lr_cosine_annealing_step, 
                                  final_div_factor=args.lr/(args.lr_min*div_factor),
                                  div_factor=div_factor,
                                  total_steps=args.lr_cosine_annealing_step, anneal_strategy='cos')
    return scheduler_warmup, optimizer, optimizing_parameters
