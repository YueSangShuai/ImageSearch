import argparse
import torch
import os
import sys
import logging
from utils import is_main_process

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--arch', type=str, default='Model', help='Model architecture')
    parser.add_argument("--backbone", type=str, default="backbone.mf3.ANet2")
    parser.add_argument("--max-seq-length", type=int, default=200)
    parser.add_argument('--feature-dim', type=int, default=1152, help='Feature dimension')
    parser.add_argument("--embedding_path", type=str, default="embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=str, default="SYNTH-PEDES/synthpedes-dataset.json")
    parser.add_argument("--text-backbone", type=str, default="tbackbone.minimind.L1")

    # training
    parser.add_argument('-j', '--workers', type=int,
                        help='加载训练数据的进程数量', default=4)
    parser.add_argument('-b', '--batch-size', type=int,
                        default=64, help='训练数据的批大小')
    parser.add_argument('--compile', action='store_true', help='是否编译模型')
    parser.add_argument('--data-count', type=int, default=0, help='训练和测试数据的最大数量')
    parser.add_argument('--image-size', type=int, nargs="+",
                        default=[224, 112], help='切图后的缩放大小。通常是网络的输入尺寸, default=224')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率, default=0.001')
    parser.add_argument('--optim', type=str, default="adan",
                        help='优化方法，支持 Adam, SGD, AdamW, default="AdamW"')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD default=0.9')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay for SGD, default=0.0005')
    parser.add_argument('--lr-warmup-step', default=10000, type=int, help="初始学习率从0到warmup到最大值的步数")
    parser.add_argument('--lr-cosine-annealing-step', default=250000, type=int, help="学习率衰减期：从最高学习率衰减到0的步数，默认140000")
    parser.add_argument('--lr-min', default=2e-5, type=float, help="最低学习率")
    parser.add_argument('--ema-step', default=0, type=int)
    parser.add_argument('--ema-decay', default=0.9999, type=float)
    parser.add_argument('--mixed_precision', default='fp16')
    parser.add_argument('--start-iterations', default=0, type=int, metavar='N',
                        help='manual iterations number (useful on restarts)')
    parser.add_argument('--iterations', default=300000, type=int,
                        help='Number of training iterations')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--train-modules', default=[], type=str, nargs='+', help='可训练层名称的前缀列表。用于仅仅训练网络的指定层。最后的分类全连接层名称为metrics')
    parser.add_argument('--gradient-clip-max', type=float, default=1.0, help='Gradient clipping threshold')

    # loss weights
    parser.add_argument('--dist-rel-weight', type=float, default=100.0, help='权重系数 for dist_rel loss')
    parser.add_argument('--id-image-weight', type=float, default=0.01, help='权重系数 for id_image loss')
    parser.add_argument('--id-cap-weight', type=float, default=0.01, help='权重系数 for id_cap loss')
    parser.add_argument('--id-img-cap-weight', type=float, default=0.001, help='权重系数 for id_img_cap loss')
    parser.add_argument('--rkda-weight', type=float, default=0.0, help='权重系数 for rkda loss')
    parser.add_argument('--cap-distall-weight', type=float, default=.0, help='权重系数 for cap_distall loss')
    parser.add_argument('--img-distall-weight', type=float, default=.0, help='权重系数 for img_distall loss')
    parser.add_argument('--cap-circle-weight', type=float, default=0.1, help='权重系数 for cap_circle loss')
    parser.add_argument('--img-circle-weight', type=float, default=0.1, help='权重系数 for img_circle loss')
    parser.add_argument("--find-unused-parameters", action="store_true", default=False)

    args = parser.parse_args()
    if len(args.image_size)==1: args.image_size=(args.image_size[0],args.image_size[0])

    if is_main_process(): 
        for temp in range(10000):  # 找到一个新目录存放本次实验日志和模型数据
            exp_name = '%s%s-%s-%s-%d/%d' % (args.name, args.arch, args.backbone, 
                'x'.join([str(s) for s in args.image_size]), args.feature_dim, temp+1)
            save_dir = os.path.join('logs', exp_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                break
        print("logs dir: ", save_dir)
        with open(os.path.join(save_dir, "cmd.sh"), 'wt', encoding='utf-8') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')
            f.close()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(levelname)s:%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(save_dir,'train.log')),
                logging.StreamHandler(sys.stdout),
            ])

        s = '\n'.join(['\t%s: %s' % (k, str(v)) for k, v in sorted(args.__dict__.items())])
        logging.info("arguments\n"+s+"\n")
        args.save_dir = save_dir

    return args
    
if __name__ == "__main__":
    args = get_args()
    print(args)