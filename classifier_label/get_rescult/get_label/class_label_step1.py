# 本程序实现对图像生成分类器标签
# 2025-03-15

import os
import sys
import re
from ImageSearchAPI import Siglip2API,BGEAPI,NomicAPI,PLIPAPI,ConbimeAPI,PEAPI
import torch
import argparse

# 1. 读取并解析分类器标签文件
sample_text = """
User:
请给我一组约40个关于男人、女人相对应的词语或句子，并遵循如下几个例子的格式：
male <-> female
行走中的男孩 <-> 行走中的女孩
男娃娃 <-> 女娃娃
哭闹的男宝 <-> 哭闹的女宝

LLM:
以下是按照要求整理的40组对应词语配对（中英对照保留中文）：

1. 一位男士 <-> 一位女士
2. 男老师 <-> 女老师
... ...
"""
# 标签文件由用户对LLM大模型的问答组成。
# 在问答文本中，提取出第一个形如"male <-> female"的配对。解析出来标签的名称和类别数量。
# 在这个例子中，标签名称为"male"和"female"，类别数量为2。
# 然后，把对话中的具有 "<->" 作为分隔符分开的词语或句子，作为标签的别名。
# 这样一来，一个标签可以有多个别名。我们把标签和对应的别名都提取出来，保存到一个List中。
# 该List的每个元素是一个包含标签名称和别名的List，其中第一个元素是标签名称，后续元素是别名。

def parse_class_label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    label_names = None
    for line in content.splitlines():
        if "<->" not in line: continue
        # 使用 re 去掉这种文本开始的行号：
        # 1. 一位男士 <-> 一位女士
        line = re.sub(r'^\d+\.', '', line).strip()
        this_label_names = line.split("<->")
        if label_names is None:
            label_names = [[a.strip()] for a in this_label_names]
        elif len(this_label_names) != len(label_names):
            print("Error: Mismatched number of labels in line:", line)
        else:
            for a, names in zip(this_label_names, label_names):
                names.append(a.strip())
    return label_names

# 2. 生成标签文件
def save_class_label_file(args, label_names):
    # 1）对标签及其别名进行文本嵌入向量计算
    num_labels, num_names = len(label_names), len(label_names[0])
    
    if len(args.conbime_model_path)!=0:
        image_search_api=ConbimeAPI(args)
        
    elif "mexma-siglip2" in args.model_file:
        image_search_api = Siglip2API(args)
    
    elif "BGE" in args.model_file:
        image_search_api = BGEAPI(args)
    
    elif "nomic" in args.model_file:
        image_search_api=NomicAPI(args)
    
    elif "PLIP" in args.model_file:
        image_search_api = PLIPAPI(args)
    elif "PE" in args.model_file:
        image_search_api=PEAPI(args)
    else:
        raise Exception(f"没有模型{args.model_file}")

    all_label_names = sum(label_names, [])
    embeddings = image_search_api.encode_text(all_label_names) #.view(len(label_names), len(label_names[0]), -1)
    print("Embeddings shape:", embeddings.shape)

    # 2）再遍历数据集中的每个图片，获得图片名和对应的向量表示
    for batch_records in image_search_api.iter_records(with_vectors=True, batch_size=256):
        if not batch_records:
            continue
        image_paths = [record.payload.get('path', 'unknown') for record in batch_records 
                    if hasattr(record, 'payload')]
        image_vectors = torch.tensor([record.vector for record in batch_records 
                                    if hasattr(record, 'vector')])
        # 3）计算图像向量与标签/别名向量的余弦相似度
        dist = image_vectors @ embeddings.T        
#        print("Distance shape:", dist.shape)
        # 4）策略1：选择相似度最高的标签/别名对应的类别作为分类结果 
        sim_, idx1 = torch.max(dist, dim=1)
#        print("Strategy 1:", sim_, idx1)
        # 5）策略2：选择一组标签/别名中相似度均值最高的类别作为分类结果 
        mean_sim = torch.mean(dist.view(-1, num_labels, num_names), dim=2)
        sim2_, idx2 = torch.max(mean_sim, dim=1)
#        print("Strategy 2:", sim_, idx2)
        # 6）打印分类结果，并保存到分类文件

        for image_path, i1, i2, s1, s2 in zip(image_paths, idx1, idx2, sim_, sim2_):
            i1 = i1.item()//num_names
            print(f"\t{image_path}\t{label_names[i1][0]}\t{label_names[i2][0]}\t{s1.item():.4f}\t{s2.item():.4f}")



def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="PE-Core-G14-448")
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--qdrant_host', type=str, default='localhost')
    parser.add_argument('--qdrant_port', type=int, default=6333)
    parser.add_argument('--collection_name', type=str, default='PE-Core-G14-448_person_embeddings')
    parser.add_argument('--inference_img', type=str, default='/data/yuesang/LLM/VectorIE/data/calling_man.png')
    parser.add_argument('--category', type=str, default='person')
    parser.add_argument('--label_text', type=str, default="/data/yuesang/LLM/VectorIE/classifier_label/gender.txt")
    
    ############ PLIP ##############
    parser.add_argument('--plip_model', type=str, default='MResNet_BERT')
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet')
    parser.add_argument('--txt_backbone', type=str, default='/data/yuesang/LLM/VectorIE/models/PLIP/bert-base-uncased')
    parser.add_argument('--img_dim', type=int, default=768)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--layers', type=list, default=[3, 4, 6, 3])
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=256)

    ############ conbime ##############
    parser.add_argument('--conbime_model_path', type=str, default=[], nargs='+')
    
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = additional_args()
    label_names = parse_class_label_file(args.label_text)
    save_class_label_file(args, label_names)