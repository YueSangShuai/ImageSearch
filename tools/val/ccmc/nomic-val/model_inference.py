
import torch

from contrastors.config import Config
from contrastors.dataset.image_text_loader import get_local_image_text_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import argparse
from contrastors.models.dual_encoder import DualEncoderConfig,DualEncoder
from transformers import PreTrainedModel
from tabulate import tabulate 
from PIL import Image
import os
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor,BertTokenizer

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




class Emove_inference:
    def __init__(self,args):
        model_name=args.vision_model
        self.device =args.device
        
        image_model, text_model, image_tokenizer, text_tokenizer, image_processor, text_processor = load_model(model_name, args.device)
        
                # 确保模型在正确的设备上
        self.image_model = image_model.to(self.device).eval()
        self.text_model = text_model.to(self.device).eval()
        
        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        
        self.image_processor = image_processor
        self.text_processor = text_processor

    @torch.no_grad()
    def inference_image(self, img_path):
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
    def inference_text(self, txt_info):
        # 对一组文本进行编码，并确保文本输入和模型在相同设备上
        encoded_input = self.text_tokenizer(txt_info, padding=True, truncation=True, return_tensors='pt').to(self.device)
        model_output = self.text_model(**encoded_input)
        
        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1).float().cpu()
        
        return text_embeddings
        