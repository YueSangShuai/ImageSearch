import argparse
from ImageSearchAPI import Siglip2API,BGEAPI,NomicAPI,PLIPAPI,ConbimeAPI,PEAPI
import torch
import os
import numpy as np
from tqdm import tqdm
# Check if flash_attn is available

def get_image_filelist(dir, exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
    filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            if filename.lower().endswith(exts):
                filelist.append(os.path.join(home, filename))
    return filelist

def parse_args(additional_args=None):
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--model_file', type=str, default="PE-Core-G14-448")
    parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--qdrant_host', type=str, default='localhost')
    parser.add_argument('--qdrant_port', type=int, default=6333)
    parser.add_argument('--collection_name', type=str, default='PE-Core-G14-448_person_embeddings')
    parser.add_argument('--inference_img', type=str, default='./data/calling_man.png')
    
    parser.add_argument('--category', type=str, default='person')
    parser.add_argument('--enroll_path', type=str, default=["/data/images/person_attribute/19w/"], nargs='+')

    ############ PLIP ##############
    parser.add_argument('--plip_model', type=str, default='MResNet_BERT')
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet')
    parser.add_argument('--txt_backbone', type=str, default='./models/PLIP/bert-base-uncased')
    parser.add_argument('--img_dim', type=int, default=768)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--layers', type=list, default=[3, 4, 6, 3])
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=256)
    
    ############ conbime ##############
    parser.add_argument('--conbime_model_path', type=str, default=[
                                                                   ], nargs='+')
    
    
    
    if additional_args is not None: additional_args(parser)
    return parser.parse_args()

def Test_main(args):
    
    if len(args.conbime_model_path)!=0:
        image_search_api=ConbimeAPI(args)
    
    elif "mexma-siglip2" in args.model_file:
        image_search_api = Siglip2API(args)
    
    elif "BGE" in args.model_file:
        image_search_api = BGEAPI(args)
    
    elif "nomic" in args.model_file:
        image_search_api = NomicAPI(args)
    
    elif "PLIP" in args.model_file:
        image_search_api = PLIPAPI(args)
    
    elif "PE" in args.model_file:
        image_search_api=PEAPI(args)
    else:
        raise Exception(f"没有模型{args.model_file}")
    
    
    for image_path in args.enroll_path:
        image_search_api.register_new_images(image_path, args.category, check_exist=False)

def Imwrite_main(args):
    
    if len(args.conbime_model_path)!=0:
        image_search_api=ConbimeAPI(args)
    
    elif "mexma-siglip2" in args.model_file:
        image_search_api = Siglip2API(args)
    
    elif "BGE" in args.model_file:
        image_search_api = BGEAPI(args)
    
    elif "nomic" in args.model_file:
        image_search_api = NomicAPI(args)
    
    elif "PLIP" in args.model_file:
        image_search_api = PLIPAPI(args)
    
    else:
        raise Exception(f"没有模型{args.model_file}")
    
    args.enroll_path=["/data/yuesang/LLM/VectorIE/train_embdings/test_dateset/jf_test_2024_1_676/"]
    image_files=get_image_filelist(args.enroll_path[0])
    imwrite_base ="/data/yuesang/LLM/VectorIE/train_embdings/test_dateset/jf_test_2024_1_676_npy/"
    for image_path in tqdm(image_files):
        # 1. 获取 embedding
        embedding = image_search_api.inference_once(image_path)
        # 2. 构造保存路径（保持目录结构）
        relative_path = os.path.relpath(image_path, start=os.path.commonpath(args.enroll_path))
        npy_path = os.path.join(imwrite_base, os.path.splitext(relative_path)[0] + ".npy")
        # 3. 创建目录
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        # 4. 保存向量
        np.save(npy_path, embedding.cpu().to(torch.float32).numpy())
        # print(f"Saved: {npy_path}")

if __name__ == '__main__':
    args = parse_args()
    Test_main(args)
