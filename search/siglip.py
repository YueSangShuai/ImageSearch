import os
import torch
from PIL import Image

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
    if device == "cuda":
        use_optimized = torch.cuda.is_available() and is_flash_attn_available()
    else:
        use_optimized = False
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
    @staticmethod
    def add_arguments(parser):
        if "--model-file" not in parser._option_string_actions:
            parser.add_argument("--model-file", type=str, default="mexma-siglip2", help="Path to the model file.")
        if "--use-cuda" not in parser._option_string_actions:
            parser.add_argument("--use-cuda", action='store_true',
                                help="Use CUDA for inference.")
    def __init__(self, 
        model_file: str="mexma-siglip2", 
        use_cuda: bool=True,
        **kargs):
        device = "cuda" if use_cuda else "cpu"
        print("load model from", model_file)
        model, tokenizer, processor = load_model(model_file, device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
    @torch.no_grad()
    def image_embedding(self, img_path=None, image=None):
        if image is None:
            img = self.load_image(img_path)
        elif isinstance(image, Image.Image):
            img = self.load_pil_image(image)
        else: # convert to PIL Image
            img = Image.fromarray(image)
            img = self.load_pil_image(img)
        img = img.to(torch.bfloat16)
        embedding = self.model.encode_images(**img, normalize=True)
        return embedding.float().cpu().numpy()
    
    @torch.no_grad()
    def text_embedding(self, query_texts: str):
        # 对一组文本进行编码
        text_inputs = self.tokenizer(query_texts, padding="max_length", max_length=128, truncation=True, return_tensors='pt').to(self.device)
        text_embedding = self.model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float()
        return text_embedding.float().cpu().numpy()

    def load_image(self, img_path: str):
        img = Image.open(img_path).convert('RGB')
        return self.load_pil_image(img)

    @torch.no_grad()
    def load_pil_image(self, img: Image.Image):
        img = self.processor(images=[img], return_tensors="pt").to(self.device)
        return img