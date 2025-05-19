import os
import pickle
from typing import List, Dict
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchText
import uuid
import requests
import time
from typing import Generator

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

# Check if flash_attn is available
def is_flash_attn_available():
    try:
        import flash_attn
        return True
    except ImportError:
        return False

# Load model and tokenizer
@torch.inference_mode()
def load_model(model_name, device="cuda"):
    use_optimized = torch.cuda.is_available() and is_flash_attn_available()
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

class ImageSearchAPI:
    def __init__(self, 
        model_file: str="mexma-siglip2", 
        device: str='cuda' if torch.cuda.is_available() else 'cpu', 
        qdrant_host: str='localhost', 
        qdrant_port: int=6333,
        collection_name: str='sl2_person_embeddings'
        ):

        print("load model from", model_file)
        if not os.path.exists(model_file):
            print(f'{model_file} is not a valid dir')
            return
        model, tokenizer, processor = load_model(model_file, device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.qdrant_client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.init_qdrant_collection()
        self.device = device

    def init_qdrant_collection(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1152, distance=models.Distance.COSINE)
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

    @torch.no_grad()
    def register_new_images(self, 
        image_dir,                # Directory containing images to register
        category="person",        # Category of the images  
        check_exist=True,         # Whether to check if the image is already registered  
        detector=None,            # Detector to use for object/person detection    
        **payload                 # Additional payload to store with the image  
    ):
        print(f"Registering image embeddings from {image_dir}, category={category}")
        total = 0
        points = []
        for root, _, files in os.walk(image_dir):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    if check_exist:
                        res = self.image_in_qdrant(img_path)
                        if len(res[0]) > 0: 
                            print(f"Image {img_path} already exists in Qdrant, skipping.")
                            continue
                    print("Processing new image:", img_path)
                    if detector:
                        detections = detector(img_path)
                        if not detections:
                            print(f"No persons detected in {img_path}")
                            continue
                        for detection in detections:
                            img = self.load_pil_image(detection["pil_image"])
                            meta_info = {**payload, "path": img_path, "category": category, "bbox": detection["bbox"]}
                            embedding = self.model.encode_images(**img, normalize=True)
                            point_id = str(uuid.uuid4())
                            points.append(
                                models.PointStruct(
                                    id=point_id,
                                    vector=embedding.float().cpu().numpy().flatten().tolist(),
                                    payload={**payload, "path": img_path, "category": category, "bbox": detection["bbox"]}
                                )
                            )
                    else:
                        img = self.load_image(img_path)
                        embedding = self.model.encode_images(**img, normalize=True)
                        point_id = str(uuid.uuid4())
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.float().cpu().numpy().flatten().tolist(),
                                payload={**payload, "path": img_path, "category": category}
                            )
                        )
                    total += 1
                    if len(points) >= 64:
                        self.save_data(points)
                        points = []
        if points:
            self.save_data(points)
        print(f"Registered {total} new image embeddings")

    @torch.no_grad()
    def register_an_image(self, img_path, image=None, category="person", **payload):
        print(f"Registering image embeddings from {img_path}, category={category}")
        if image is None:
            img = self.load_image(img_path)
        else:
            img = self.load_pil_image(image)
        embedding = self.model.encode_images(**img, normalize=True)
        point_id = str(uuid.uuid4())
        points = [
            models.PointStruct(
                id=point_id,
                vector=embedding.float().cpu().numpy().flatten().tolist(),
                payload={**payload, "path": img_path, "category": category}
            )
        ]
        self.save_data(points)
        print(f"Registered 1 new image embedding")

    def save_data(self, points):
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    @torch.no_grad()
    def search(self, query_text: str, top_k: int = 5, threshold: float = 0.5, payload: dict = None, return_payload: bool = False):
        print("Searching for query:", query_text)
        
        # 计时：文本编码阶段开始
        text_embedding_start = time.time()
        text_inputs = self.tokenizer(query_text, padding="max_length", max_length=128, truncation=True, return_tensors='pt').to(self.device)
        text_embedding = self.model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float().cpu().numpy().flatten().tolist()
        text_embedding_end = time.time()
#        print(f"文本编码阶段耗时: {text_embedding_end - text_embedding_start:.4f} 秒")
        
        # 计时：向量搜索阶段开始
        vector_search_start = time.time()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=text_embedding,
            limit=top_k, score_threshold=threshold
        )
        vector_search_end = time.time()
#        print(f"向量搜索阶段耗时: {vector_search_end - vector_search_start:.4f} 秒")
        total_time = vector_search_end - text_embedding_start
#        print(f"总耗时: {total_time:.4f} 秒")
        if return_payload: return search_result, total_time

        results = [(hit.payload['path'], hit.score) for hit in search_result]
        # for result in results:
        #     print(f"{result[0]}: {result[1]}")
        return results, total_time

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]):
        # 对一组文本进行编码
        text_inputs = self.tokenizer(query_texts, padding="max_length", max_length=128, truncation=True, return_tensors='pt').to(self.device)
        text_embedding = self.model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float().cpu()
        return text_embedding
    
    @torch.no_grad()
    def search_by_embedding(self, query_embedding, top_k: int = 5, threshold: float = 0.5):
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k, score_threshold=threshold
        )
        results = [(hit.payload['path'], hit.score) for hit in search_result]
        return results

    @torch.no_grad()
    def search_by_image(self, query_image, top_k: int = 5, threshold: float = 0.5) -> List[str]:
        """
        使用图像作为查询进行搜索
        :param query_image: PIL图像对象
        :param top_k: 返回结果数量
        :param threshold: 相似度阈值
        :return: 搜索结果列表和查询耗时
        """
        print("Searching with image query")
        
        # 计时：图像编码阶段开始
        image_embedding_start = time.time()
        
        # 处理图像并提取特征
        img_tensor = self.load_pil_image(query_image)
        image_embedding = self.model.encode_images(**img_tensor, normalize=True)
        image_embedding = image_embedding.float().cpu().numpy().flatten().tolist()
        
        image_embedding_end = time.time()
        print(f"图像编码阶段耗时: {image_embedding_end - image_embedding_start:.4f} 秒")
        
        # 计时：向量搜索阶段开始
        vector_search_start = time.time()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=image_embedding,
            limit=top_k, 
            score_threshold=threshold
        )
        vector_search_end = time.time()
        print(f"向量搜索阶段耗时: {vector_search_end - vector_search_start:.4f} 秒")
        
        total_time = vector_search_end - image_embedding_start
        print(f"总耗时: {total_time:.4f} 秒")

        results = [(hit.payload['path'], hit.score) for hit in search_result]
        return results, total_time

    def load_image(self, img_path: str):
        img = Image.open(img_path)
        return self.load_pil_image(img)

    @torch.no_grad()
    def load_pil_image(self, img: Image.Image):
        img = img.convert('RGB')
        img = self.processor(images=[img], return_tensors="pt").to(self.device)
        return img.to(torch.bfloat16)

    def delete_image(self, img_path: str):
        print("Deleting image:", img_path)

        result = self.image_in_qdrant(img_path)
        if result:
            print(result)
            point_id = result[0][0].id
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            os.remove(img_path)
            print(f"Deleted image: {img_path}")
        else:
            print(f"Image not found: {img_path}")

    def count_records(self) -> int:
        """Returns the number of records in the Qdrant collection."""
        result = self.qdrant_client.count(
            collection_name=self.collection_name
        )
        return result.count

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

    def delete_all_records(self):
        """Deletes all records from the Qdrant collection."""
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector="*"
        )

