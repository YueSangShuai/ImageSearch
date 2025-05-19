import os
from typing import List
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchText
import uuid
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor,BertTokenizer
import torch.nn.functional as F
from PLIPmodel import Create_PLIP_Model
from torchvision import transforms
import requests
import argparse
import numpy as np
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms



# Check if FlashAttention is available (you might need to implement `is_flash_attn_available`)
def is_flash_attn_available():
    # Assuming this function checks if FlashAttention is available
    # You can implement the check based on your environment or requirements.
    return False  # Replace this with your check if necessary

@torch.inference_mode()
def load_model(model_name, device="cuda"):
    # Check if the provided device is CUDA or CPU
    device = torch.device(device)

    use_optimized = torch.cuda.is_available() and is_flash_attn_available()

    if "mexma-siglip2" in model_name:
        model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            optimized=True if use_optimized else False,
            local_files_only=True
        )
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        return model, tokenizer, processor
         
    elif "BGE" in model_name:
        model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            local_files_only=True
        )
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        return model, tokenizer, processor
        
    elif "nomic" in model_name:
        image_model_name = os.path.join(model_name, "nomic-embed-vision-v1.5")
        text_model_name = os.path.join(model_name, "nomic-embed-text-v1.5")
        
        image_model = AutoModel.from_pretrained(
            image_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            local_files_only=True
        )
        
        text_model = AutoModel.from_pretrained(
            text_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            local_files_only=True
        )
        
        image_model.to(device)
        text_model.to(device)
        
        image_tokenizer = None
        image_processor = AutoImageProcessor.from_pretrained(image_model_name, local_files_only=True, trust_remote_code=True)
        
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=True, trust_remote_code=True)
        text_processor = None
        
        return image_model, text_model, image_tokenizer, text_tokenizer, image_processor, text_processor
        
    elif "PLIP" in model_name:
        checkpoint = torch.load(model_name, map_location=device)  # Load checkpoint directly to the specified device
        return checkpoint
    
    elif "PE" in model_name:
        model = pe.CLIP.from_config(model_name, pretrained=True)
        preprocess = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        
        return model,tokenizer,preprocess
        

class ImageSearchAPI(ABC):
    def __init__(self, args):
        print("load model from", args.model_file)
        # if not os.path.exists(args.model_file):
        #     print(f'{args.model_file} is not a valid dir')
        #     return
        
        model, tokenizer, processor = load_model(args.model_file, args.device)
                
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.qdrant_client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
        self.collection_name = args.collection_name
        self.device = args.device
        
    @abstractmethod
    def inference_once(self,img_path):
        pass
    
    @abstractmethod
    def encode_text(self, query_texts: List[str]):
        pass
    
    
    def init_qdrant_collection(self,dimsion):
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=dimsion, distance=models.Distance.COSINE)
                )
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
            raise

    def image_in_qdrant(self, img_path: str) -> bool:
        result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="path",
                        match=MatchText(text=img_path)
                    )
                ]
            ),
            limit=1  # 设置适当的限制
        )    
        return result

    @abstractmethod
    def register_new_images(self, image_dir, category="person", check_exist=True):
        """该方法必须在子类中实现"""
        pass

    def save_data(self, points):
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def iter_records(self, with_vectors=False, batch_size=100):
        """迭代遍历集合中的所有记录，使用分页方式处理大数据集"""
        offset = None
        while True:
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=with_vectors
            )
            
            # Qdrant的scroll方法返回(points, next_offset)的元组
            points, next_offset = results
            
            # 如果没有数据，退出循环
            if not points:
                break
                
            # 返回当前批次的数据
            yield points
            
            # 如果没有更多数据，退出循环
            if next_offset is None:
                break
                
            # 更新下一次查询的偏移量
            offset = next_offset

class Siglip2API(ImageSearchAPI):
    def __init__(self, args):
        # 根据设备初始化设备
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        super().__init__(args)  # 调用父类的初始化方法
        self.dismision = self.inference_once(args.inference_img).shape[1]  # 获取嵌入维度
        if len(args.conbime_model_path) == 0:
            self.init_qdrant_collection(self.dismision)
    
    @torch.no_grad()
    def inference_once(self, img_path):
        # 加载并处理图像
        img = self.load_image(img_path)

        # 确保模型在正确的设备上
        embedding = self.model.encode_images(**img.to(self.device), normalize=True)
        return embedding   

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 对一组文本进行编码
        text_inputs = self.tokenizer(query_texts, padding="max_length", max_length=128, truncation=True, return_tensors='pt').to(self.device)
        text_embedding = self.model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float().cpu()
        return text_embedding
    
    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持多种图像格式
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0:
                            run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")

    def load_image(self, img_path: str):
        img = Image.open(img_path)
        return self.load_pil_image(img)

    @torch.no_grad()
    def load_pil_image(self, img: Image.Image):
        img = img.convert('RGB')
        img = self.processor(images=[img], return_tensors="pt").to(self.device)
        return img.to(torch.bfloat16) 

class BGEAPI(ImageSearchAPI):
    def __init__(self, args):
        # 根据设备初始化设备
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        super().__init__(args)  # 调用父类的初始化方法
        
        # 确保模型在正确的设备上
        self.model.to(self.device)  # 将模型加载到指定设备
        
        # 处理器设置
        self.model.set_processor(args.model_file)
        
        # 获取嵌入维度
        self.dismision = self.inference_once(args.inference_img).shape[1]
        
        # 如果没有组合模型路径，则初始化 Qdrant 集合
        if len(args.conbime_model_path) == 0:
            self.init_qdrant_collection(self.dismision)
    
    @torch.no_grad()
    def inference_once(self, img_path):
        # 确保图像数据加载到正确的设备
        embedding = self.model.encode(img_path)
        return embedding
    
    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 对一组文本进行编码，并确保数据和模型在同一设备上
        text_embedding = self.model.encode(text=query_texts)
        text_embedding = text_embedding.float().cpu()
        return text_embedding
    
    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持多种图像格式
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0:
                            run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)
                        
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")

class NomicAPI(ImageSearchAPI):
    def __init__(self, args):
        print("load model from", args.model_file)
        
        image_model_name = os.path.join(args.model_file, "nomic-embed-vision-v1.5")
        text_model_name = os.path.join(args.model_file, "nomic-embed-text-v1.5")
        
        if not os.path.exists(image_model_name):
            print(f'{image_model_name} is not a valid dir')
            return
        
        if not os.path.exists(text_model_name):
            print(f'{text_model_name} is not a valid dir')
            return
        
        # 选择设备并加载模型
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        # 加载模型和处理器
        image_model, text_model, image_tokenizer, text_tokenizer, image_processor, text_processor = load_model(args.model_file, args.device)
        
        # 确保模型在正确的设备上
        self.image_model = image_model.to(self.device).eval()
        self.text_model = text_model.to(self.device).eval()
        
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        
        self.image_processor = image_processor
        self.text_processor = text_processor
        
        self.qdrant_client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
        self.collection_name = args.collection_name
        
        # 获取嵌入维度
        self.dismision = self.inference_once(args.inference_img).shape[1]
        if len(args.conbime_model_path) == 0:
            self.init_qdrant_collection(self.dismision)

    @torch.no_grad()
    def inference_once(self, img_path):
        # 加载图像并将其转换到设备上
        image = Image.open(img_path)
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        img_emb = self.image_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        return img_embeddings
    
    @torch.no_grad()
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 对一组文本进行编码，并确保文本输入和模型在相同设备上
        encoded_input = self.text_tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        model_output = self.text_model(**encoded_input)
        
        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1).float().cpu()
        
        return text_embeddings

    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持多种图像格式
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0:
                            run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)
                        
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")

class PLIPAPI(ImageSearchAPI):
    def __init__(self, args):
        print("load model from", args.model_file)
        token_path = os.path.join(args.model_file, 'bert-base-uncased')
        model_path = os.path.join(args.model_file, 'PLIP_RN50.pth.tar')
        
        if not os.path.exists(model_path):
            print(f'{model_path} is not a valid dir')
            return
        
        if not os.path.exists(token_path):
            print(f'{token_path} is not a valid dir')
            return
        
        # 选择设备并加载模型
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 将模型加载到指定设备
        model = Create_PLIP_Model(args).to(self.device)
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.eval()
        
        self.qdrant_client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
        self.collection_name = args.collection_name
        self.device = self.device
        
        # 确保tokenizer在CPU上工作，因为BERT tokenizer不需要GPU
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(args.model_file, 'bert-base-uncased'))
        
        # 获取嵌入维度
        self.dismision = self.inference_once(args.inference_img).shape[1]
        if len(args.conbime_model_path) == 0:
            self.init_qdrant_collection(self.dismision)

    @torch.no_grad()
    def load_image(self, img_path: str):
        transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.357, 0.323, 0.328),
                                 (0.252, 0.242, 0.239))
        ])
        img = Image.open(img_path).convert('RGB')
        return transform(img).unsqueeze(0)

    @torch.no_grad()
    def inference_once(self, img_path):
        img = self.load_image(img_path)
        img_embeddings = self.model.get_image_embeddings(img.to(self.device))  # Ensure the image tensor is on the correct device
        return img_embeddings

    @staticmethod
    def translate_to_english(text: str) -> str:    
        url = "http://10.12.16.205:8989/translate"
        headers = {"Authorization": "shiliangfadongji"}
        data = {"from": "zh", "to": "en", "text": text}
        response = requests.post(url, headers=headers, json=data)
        return response.json().get("result", text)

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 如果文本是中文，进行翻译
        if any(u'一' <= char <= u'鿿' for char in query_texts):
            query_texts = self.translate_to_english(query_texts)
            print("Translated query:", query_texts)
        
        result = self.tokenizer(query_texts, padding="max_length", max_length=64, truncation=True, return_tensors='pt')
        tokens, mask = result["input_ids"], result["attention_mask"]
        tokens, mask = tokens.to(self.device), mask.to(self.device)  # 将tokens和mask移到正确的设备
        text_embedding = self.model.get_text_global_embedding(tokens, mask)
        text_embedding = text_embedding.detach().cpu().numpy()
        
        return text_embedding

    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持多种图像格式
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0:
                            run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)
                        
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")
  
class ConbimeAPI(ImageSearchAPI):
    def __init__(self,args):
        self.Siglip2_Search=None
        self.BGE_Search=None
        self.Nomic_Search=None
        self.PLIP_Search=None

        for model_path in args.conbime_model_path:
            args.model_file=model_path
            if "mexma-siglip2" in args.model_file:
                self.Siglip2_Search = Siglip2API(args)
            elif "BGE" in args.model_file:
                self.BGE_Search = BGEAPI(args)
            elif "nomic" in args.model_file:
                self.Nomic_Search = NomicAPI(args)
            elif "PLIP" in args.model_file:
                self.PLIP_Search = PLIPAPI(args)

        if not (self.Siglip2_Search or self.BGE_Search or self.Nomic_Search or self.PLIP_Search):
            raise ValueError("no match model!")
        
        self.qdrant_client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
        self.collection_name = args.collection_name
        self.device = args.device
        
        self.dismision=self.inference_once(args.inference_img).shape[1]
        self.init_qdrant_collection(self.dismision)
        
        
    @torch.no_grad()
    def inference_once(self,img_path):
        embdings_list=[]
        
        if self.Nomic_Search:
            embdings_list.append(self.Nomic_Search.inference_once(img_path))
        
        if self.Siglip2_Search:
            embdings_list.append(self.Siglip2_Search.inference_once(img_path))
        if self.BGE_Search:
            embdings_list.append(self.BGE_Search.inference_once(img_path))

        if self.PLIP_Search:
            embdings_list.append(self.PLIP_Search.inference_once(img_path))
        
        embdings=torch.cat(embdings_list, dim=1)
        
        return embdings
    
    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        
        embdings_list=[]
        
        
        if self.Nomic_Search:
            embdings_list.append(self.Nomic_Search.encode_text(query_texts))
        
        if self.Siglip2_Search:
            embdings_list.append(self.Siglip2_Search.encode_text(query_texts))
        if self.BGE_Search:
            embdings_list.append(self.BGE_Search.encode_text(query_texts))

        if self.PLIP_Search:
            embdings_list.append(self.PLIP_Search.encode_text(query_texts))
            
        embdings=torch.cat(embdings_list, dim=1)
        return embdings
    
    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0: run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")
  

class PEAPI(ImageSearchAPI):
    def __init__(self, args):
        # 根据设备初始化设备
        self.device = args.device
        self.model = pe.CLIP.from_config(args.model_file, pretrained=True).to(self.device)  # 将模型移到指定设备
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        self.processor = transforms.get_image_transform(self.model.image_size)
        self.qdrant_client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
        self.collection_name = args.collection_name
        
        self.dismision = self.inference_once(args.inference_img).shape[1]  # 获取嵌入维度
        if len(args.conbime_model_path) == 0:
            self.init_qdrant_collection(self.dismision)
        
    @torch.no_grad()
    def inference_once(self, img_path):
        # 图像处理
        img = self.processor(Image.open(img_path)).unsqueeze(0).to(self.device)  # 确保图像在正确的设备上
        embedding = self.model.encode_image(img, normalize=True)
        return embedding 

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 对文本进行编码
        text_inputs = self.tokenizer(query_texts).to(self.device)  # 确保文本在正确的设备上
        text_embedding = self.model.encode_text(text_inputs, normalize=True)
        text_embedding = text_embedding.float().cpu()
        return text_embedding
    
    @torch.no_grad()
    def register_new_images(self, image_dir, category="person", check_exist=True):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持多种图像格式
                    run = True
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0:
                            run = False
                    if run:
                        print("Processing new image:", img_path)
                        embedding = self.inference_once(img_path=img_path)  # 图像嵌入处理
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={"path": img_path, "category": category}
                            )
                        )
                        if len(points) >= 64:
                            self.save_data(points)
                            points = []
        if points:
            self.save_data(points)
            print(f"Registered {len(points)} new image embeddings")
        else:
            print("No new images to register")
    
    


    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_file', type=str, default="/data/yuesang/LLM/VectorIE/models/mexma-siglip2")
    # parser.add_argument('--model_file', type=str, default="/data/yuesang/LLM/VectorIE/models/BGE-VL-base")
    # parser.add_argument('--model_file', type=str, default="/data/yuesang/LLM/VectorIE/models/BGE-VL-Large")
    parser.add_argument('--model_file', type=str, default="/data/yuesang/LLM/VectorIE/models/nomic")
    # parser.add_argument('--model_file', type=str, default="PE-Core-L14-336")
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--qdrant_host', type=str, default='localhost')
    parser.add_argument('--qdrant_port', type=int, default=6333)
    parser.add_argument('--collection_name', type=str, default='test')
    parser.add_argument('--inference_img', type=str, default='/data/yuesang/LLM/VectorIE/data/calling_man.png')
    parser.add_argument('--category', type=str, default='person')
    parser.add_argument('--label_text', type=str, default="/data/yuesang/LLM/VectorIE/classifier_label/age.txt")
    
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
    parser.add_argument('--conbime_model_path', type=str, default=["/data/yuesang/LLM/VectorIE/models/nomic",
                                                                   "/data/yuesang/LLM/VectorIE/models/mexma-siglip2"
                                                                   ], nargs='+')
    
    args=parser.parse_args()
    if "mexma-siglip2" in args.model_file:
        image_search_api = Siglip2API(args)
    
    elif "BGE" in args.model_file:
        image_search_api = BGEAPI(args)
    
    elif "nomic" in args.model_file:
        image_search_api = NomicAPI(args)
    
    elif "PLIP" in args.model_file:
        image_search_api = PLIPAPI(args)
    
    elif "PE" in args.model_file:
        image_search_api=PEAPI(args)
    
    embdings=image_search_api.inference_once("/data/yuesang/LLM/VectorIE/classifier_label/image-3.png")
    print(embdings.shape)
    warnup=10
    test_range=100
    
    from tqdm import tqdm
    
    for i in tqdm(range(warnup)):
        image_search_api.encode_text("/data/yuesang/LLM/VectorIE/classifier_label/image-3.png")
        
    import time
    start=time.time()
    

    
    for i in tqdm(range(test_range)):
        image_search_api.encode_text("/data/yuesang/LLM/VectorIE/classifier_label/image-3.png")
        
    end=time.time()
    inference_time=end-start
    
    print(inference_time*1000/test_range)


    
    