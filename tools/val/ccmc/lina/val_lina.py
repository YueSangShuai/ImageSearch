import torchvision.transforms as transforms
import os
import shutil
import argparse
import warnings
import json
import torch
import numpy as np

def test_map(query_feature,query_label,gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp

def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

from siglip import ImageTextEmbedder
from model_image import ImageFeatureExtractor
from model_text import TextFeatureExtractor 


def test_mini(args): 
    #feature_extractor = ImageTextEmbedder(**vars(args))
    
    feature_extractor = ImageFeatureExtractor(**vars(args))
    feature_extractor_text = TextFeatureExtractor(**vars(args))
    
    # Set batch size for processing
    batch_size = 32
    
    with open(args.caption_path, 'r', encoding='utf8') as fp:
        dataset = json.load(fp)
        if args.sample_limit > 0:
            dataset = dataset[:args.sample_limit]
            print(f"Using {len(dataset)} samples from dataset")
    
    qids, gids, qfeats, gfeats = [], [], [], []
    
    for item in dataset:
        label = item["id"]
        label = torch.tensor(label)
        file_path = os.path.join(args.image_dir,item["file_path"])
        img_feat = torch.tensor(feature_extractor.image_embedding(file_path)).unsqueeze(0)
        gfeats.append(img_feat)
        gids.append(label.view(-1)) 

    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)
    
    # text
    initial_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        label = item["id"]
        captions_list = item["captions"]
        for j in range(len(captions_list)):
            caption = captions_list[j]
            initial_data.append([label,caption])
    for index in range(len(initial_data)):
        caption = initial_data[index][1]
        label = initial_data[index][0]
        label = torch.tensor(label)
        text_feat = torch.tensor(feature_extractor_text.text_embedding(caption)).unsqueeze(0)
        print(text_feat)
        qids.append(label.view(-1))  # flatten
        qfeats.append(text_feat)
        
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)
    print(qfeats.shape)
        
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(qfeats, qids, gfeats, gids)
    return ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP

def test(args): 
    feature_extractor = ImageTextEmbedder(**vars(args))
    
    with open(args.caption_path, 'r', encoding='utf8') as fp:
        dataset = json.load(fp)
        if args.sample_limit > 0:
            dataset = dataset[:args.sample_limit]
            print(f"Using {len(dataset)} samples from dataset")
    
    qids, gids, qfeats, gfeats = [], [], [], []
    
    for item in dataset:
        label = item["id"]
        label = torch.tensor(label)
        file_path = os.path.join(args.image_dir,item["file_path"])
        img_feat = torch.tensor(feature_extractor.image_embedding(file_path))
        gfeats.append(img_feat)
        gids.append(label.view(-1)) 

    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)
    
    # text
    initial_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        label = item["id"]
        captions_list = item["captions"]
        for j in range(len(captions_list)):
            caption = captions_list[j]
            initial_data.append([label,caption])
    for index in range(len(initial_data)):
        caption = initial_data[index][1]
        label = initial_data[index][0]
        label = torch.tensor(label)
        text_feat = torch.tensor(feature_extractor.text_embedding(caption))
        qids.append(label.view(-1))  # flatten
        qfeats.append(text_feat)
        
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)
    print(qfeats.shape)
        
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(qfeats, qids, gfeats, gids)
    return ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP


def Test_main(args):
    
    
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(args)

    print('{:.5f}  {:.5f}  {:.5f}  {:.5f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="CUHK-PEDES eval ")
#     parser.add_argument('--image_dir', type=str, default='/data/lina/datasets/CUHK-PEDES/imgs')
#     parser.add_argument('--caption_path', type=str,
#                         default='/data/lina/datasets/CUHK-PEDES/caption_all.json',
#                         help='path for test annotation json file')

    
#     args = parser.parse_args()

#     ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CUHK-PEDES eval ")
    parser.add_argument('--image_dir', type=str, default='/data/lina/datasets/CUHK-PEDES/imgs')
    parser.add_argument('--caption_path', type=str,
                        default='/data/lina/datasets/CUHK-PEDES/caption_all.json',
                        help='path for test annotation json file')
    parser.add_argument('--model_type', type=str, choices=['mini', 'large'], default='large',
                        help='Choose model type: mini (uses ImageFeatureExtractor/TextFeatureExtractor) or full (uses ImageTextEmbedder)')
    parser.add_argument('--sample_limit', type=int, default=1000,
                        help='Number of samples to process (0 for all)')
    
    # Add ImageFeatureExtractor arguments
    parser = ImageFeatureExtractor.add_arguments(parser)
    parser = TextFeatureExtractor.add_arguments(parser)
    
    args = parser.parse_args()
    
    if args.model_type == 'mini':
        print(f"Running mini model with {args.sample_limit} samples")
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_mini(args)
    else:
        print(f"Running large model with {args.sample_limit} samples")
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(args)
    