import os
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

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


class ImageTextEmbedder:
    def __init__(self, 
        model_file: str="mexma-siglip2", 
        device: str='cuda' if torch.cuda.is_available() else 'cpu'):

        print("load model from", model_file)
        if not os.path.exists(model_file):
            print(f'{model_file} is not a valid dir')
            return
        model, tokenizer, processor = load_model(model_file, device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
    @torch.no_grad()
    def image_embedding(self, img_path, image=None):
        if image is None:
            img = self.load_image(img_path)
        else:
            img = self.load_pil_image(image)
        embedding = self.model.encode_images(**img, normalize=True)
        return embedding
    
    @torch.no_grad()
    def text_embedding(self, query_texts: str):
        # 对一组文本进行编码
        text_inputs = self.tokenizer(query_texts, padding="max_length", max_length=128, truncation=True, return_tensors='pt').to(self.device)
        text_embedding = self.model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float()
        return text_embedding

    def load_image(self, img_path: str):
        img = Image.open(img_path)
        return self.load_pil_image(img)

    @torch.no_grad()
    def load_pil_image(self, img: Image.Image):
        img = img.convert('RGB')
        img = self.processor(images=[img], return_tensors="pt").to(self.device)
        return img.to(torch.bfloat16)

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
def read_json(fn: str):
    import json
    with open(fn, 'r') as f:
        return json.load(f)


def extract_attributes_captions_prompt_caption(embedder: ImageTextEmbedder, json_data: list[dict], root: str, output_root: str, count: int = 100000000000):
    i = 0
    for item in json_data:
        file_path = item['file_path']
        attributes = item['attributes']
        captions = item['captions']
        out_fn = os.path.join(output_root, file_path + ".bin")
        if os.path.exists(out_fn):
            continue
        prompt_caption = item['prompt_caption']
        image_embedding = embedder.image_embedding(os.path.join(root, file_path))
        text_embedding_attr = embedder.text_embedding(','.join([attr for attr in attributes[0].split(',') if attr.strip()!="unknown"]))
        text_embedding_captions = embedder.text_embedding(captions)
        text_embedding_prompt_caption = embedder.text_embedding(prompt_caption)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        torch.save({
            'image_embedding': image_embedding.cpu(),
            'text_embedding_attr': text_embedding_attr.cpu(),
            'text_embedding_captions': text_embedding_captions.cpu(),
            'text_embedding_prompt_caption': text_embedding_prompt_caption.cpu()
        }, out_fn)
        i += 1
        print(f"Extracted embeddings for {i}[{image_embedding.shape}]: {file_path} -> {out_fn}")
        if i >= count:
            break
    print("All embeddings extracted.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, default="mexma-siglip2")
    parser.add_argument("--root", type=str, default="SYNTH-PEDES")
    parser.add_argument("--output_root", type=str, default="embeddings")
    parser.add_argument("--count", type=int, default=100000000000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embedder = ImageTextEmbedder(args.model_file, args.device)
    json_data = read_json(args.root + "/synthpedes-dataset.json")
    extract_attributes_captions_prompt_caption(embedder, json_data, args.root, args.output_root, args.count)    