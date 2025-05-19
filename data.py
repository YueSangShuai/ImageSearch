import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
import numpy as np 
import os
from PIL import Image
from transformers import AutoTokenizer


"""sample data
[
    {
        "id": 1,
        "file_path": "Part1/1/0.jpg",
        "attributes": [
            "woman,short hair,black jacket,blue denim jeans,black sneakers,black backpack"
        ],
        "captions": [
            "A woman with black hair and she is wearing a black jacket with blue jeans paired with black shoes."
        ],
        "prompt_caption": [
            "The woman has short hair. She is wearing a black jacket, blue denim jeans and black sneakers. She is carrying a black backpack. "
        ]
    },
    {
        "id": 1,
        "file_path": "Part1/1/1.jpg",
        "attributes": [
            "girl,dark hair,black jacket,blue jeans,black shoes,unknown"
        ],
        "captions": [
            "A woman wearing a black jacket over a Gray shirt, a pair of blue jeans and a pair of black shoes."
        ],
        "prompt_caption": [
            "A girl with dark hair wearing a black jacket and blue jeans with black shoes."
        ]
    },
    ... ...
"""

class ImageAugTransformEval:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def __call__(self, img):
        return self.transform(img)

class ImageAugTransformTrain(ImageAugTransformEval):
    def __init__(self, image_size):
        super().__init__(image_size)
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize(image_size),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def text_to_bytes(text, max_seq_length=100):
    # Convert text to UTF-8 bytes with padding
    char_bytes = []
    for char, i in zip(text, range(max_seq_length)):
        char_utf8 = char.encode('utf-8')
        char_bytes.append(list(char_utf8) + [0] * (4 - len(char_utf8))) # Pad to 4 bytes
    l = max_seq_length-len(char_bytes)
    if l>0:
        char_bytes+=[(0,0,0,0)]*l
    return torch.tensor(char_bytes, dtype=torch.uint8)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

class PairedImageTextDataset(Dataset):
    """
    Placeholder Dataset. Replace with your actual image/text loading and preprocessing.
    Needs image loading (e.g., PIL) + transforms (e.g., torchvision.transforms)
    Needs text tokenization (e.g., Hugging Face AutoTokenizer)
    """
    def __init__(self, json_fn, embedding_path, text_len=200, train=True, image_size=(224, 112)):
        self.text_len = text_len
        self.root = os.path.dirname(json_fn)
        self.embedding_path = embedding_path
        import json
        with open(json_fn, 'r') as f:
            self.all_data = json.load(f)
        self.training = train
        self.image_size = image_size
        self.id2index = {}
        if train:
            self.train()
        else:
            self.eval()
    def __len__(self):
        return len(self.data)
    def eval(self):
        self.training = False
        self.image_transform = ImageAugTransformEval(self.image_size)
        self.data = self.all_data[:5000]
        self.id2index = {}
        for i, item in enumerate(self.data):
            if item['id'] not in self.id2index:
                self.id2index[item['id']] = len(self.id2index)
        return self
    def train(self):
        self.training = True
        self.image_transform = ImageAugTransformTrain(self.image_size)
        self.data = self.all_data[5000:]
        self.id2index = {}
        for i, item in enumerate(self.data):
            if item['id'] not in self.id2index:
                self.id2index[item['id']] = len(self.id2index)
        return self
    def num_class(self):
        return len(self.id2index)
    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item['file_path']
        attributes = item['attributes']
        attributes = [attr for attr in attributes[0].split(',')]
        captions = item['captions']
        prompt_caption = item['prompt_caption'][0]
        image_file = os.path.join(self.root, file_path)
        img = Image.open(image_file).convert('RGB')
        img = self.image_transform(img)
        out_fn = os.path.join(self.embedding_path, file_path + ".bin")
        index = np.random.randint(0, len(captions)) if self.training else 0
        caption = captions[index]
        attributes_list = [x for x in attributes if x != 'unknown']
        np.random.shuffle(attributes_list)
        attributes = ",".join(attributes_list)
        # input_ids = self.tokenizer(
        #     [caption, prompt_caption, attributes], 
        #     return_tensors="pt", padding=True, truncation=True, max_length=self.text_len).input_ids
        # caption, prompt_caption, attributes = input_ids[0], input_ids[1], input_ids[2]
        person_id = self.id2index[item['id']]
        if os.path.exists(out_fn):
            data = torch.load(out_fn, map_location="cpu") # which saved by tools/extract.py
            image_embedding = data['image_embedding'].float().cpu()
            text_embedding_attr = data['text_embedding_attr'].float().cpu()
            text_embedding_captions = data['text_embedding_captions'].float().cpu()
            text_embedding_caption = text_embedding_captions[index].float().cpu()
            text_embedding_prompt_caption = data['text_embedding_prompt_caption'].float().cpu()
            return {
                "id": person_id,
                'image': img,
                'image_embedding': image_embedding[0],
                'caption': caption,
                'caption_embedding': text_embedding_caption,
                'attributes': attributes,
                'attributes_embedding': text_embedding_attr[0],
                'prompt_caption': prompt_caption,
                'prompt_caption_embedding': text_embedding_prompt_caption[0],
                'file_path': file_path,
                'has_embedding': True
            }
        else:
            return {
                "id": person_id,
                'image': img,
                'caption': caption,
                'attributes': attributes,
                'prompt_caption': prompt_caption,
                'file_path': file_path,
                'has_embedding': False
            }
    @staticmethod
    def collate_fn(batch):
        captions = [item['caption'] for item in batch]
        prompt_captions = [item['prompt_caption'] for item in batch]
        attributes = [item['attributes'] for item in batch]
        
        caption_ids = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=200).input_ids
        prompt_caption_ids = tokenizer(prompt_captions, return_tensors="pt", padding=True, truncation=True, max_length=200).input_ids
        attributes_ids = tokenizer(attributes, return_tensors="pt", padding=True, truncation=True, max_length=200).input_ids
        
        ret = {
            'caption': caption_ids,
            'caption_embedding': torch.stack([item['caption_embedding'] for item in batch]),
            'prompt_caption': prompt_caption_ids,
            'prompt_caption_embedding': torch.stack([item['prompt_caption_embedding'] for item in batch]),
            'attributes': attributes_ids,
            'attributes_embedding': torch.stack([item['attributes_embedding'] for item in batch]),
            'id': torch.tensor([item['id'] for item in batch]),
            'image': torch.stack([item['image'] for item in batch]),
            'image_embedding': torch.stack([item['image_embedding'] for item in batch]),
            'file_path': [item['file_path'] for item in batch],
            'has_embedding': torch.tensor([item['has_embedding'] for item in batch])
        }
        return ret

if __name__ == '__main__':
    DB = PairedImageTextDataset('SYNTH-PEDES/synthpedes-dataset.json', 'embeddings', train=True)
    for i in range(10):
        j = np.random.randint(0, min(10000,len(DB)))
        print(f"{i}:{j}")
        data = DB[j]
        for k, v in data.items():
            print(f"\t{k}: {v.shape if isinstance(v, torch.Tensor) else (len(v), v) if type(v) in (list, tuple, str) else v}")

"""
0:7749
        image_embedding: torch.Size([1152])
        text_embedding_attr: torch.Size([1152])
        text_embedding_caption: torch.Size([1152])
        text_embedding_prompt_caption: torch.Size([1152])
        caption: (83, 'A woman wearing a blue shirt, a pair of purple pants and a watch on her left wrist.')
        image: torch.Size([3, 224, 112])
        attributes: (6, ['woman', 'dark hair', 'blue shirt', 'purple pants', 'white shoes', 'unknown'])
        prompt_caption: (107, 'A woman with dark hair is wearing a blue shirt. She is also wearing a pair of purple pants and white shoes.')
        file_path: (16, 'Part1/907/15.jpg')
1:275
        image_embedding: torch.Size([1152])
        text_embedding_attr: torch.Size([1152])
        text_embedding_caption: torch.Size([1152])
        text_embedding_prompt_caption: torch.Size([1152])
        caption: (92, 'A woman wearing a long sleeve, Gray jacket, a pair of white pants and a pair of white shoes.')
        image: torch.Size([3, 224, 112])
        attributes: (6, ['woman', 'unknown', 'gray jacket', 'white pants', 'white shoes', 'unknown'])
        prompt_caption: (79, 'A man is wearing a gray jacket.He is also wearing white pants with white shoes.')
        file_path: (15, 'Part1/383/4.jpg')
"""