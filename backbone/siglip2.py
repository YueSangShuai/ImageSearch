from transformers import AutoModel, AutoImageProcessor
import torch.nn as nn
from torch import Tensor

class Base(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        # self.processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
        self.out = nn.Linear(768, out_features)
    def forward(self, x: Tensor, return_features=False) -> Tensor:
        y = self.model.get_image_features(pixel_values=x)
        y = self.out(y)
        if return_features: return [y]
        return y

"""
import torch
from transformers.image_utils import load_image
from transformers import AutoModel, AutoImageProcessor

# load the model and processor
ckpt = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoImageProcessor.from_pretrained(ckpt)

# load the image
image = load_image("000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)
x = inputs['pixel_values']
print(x.shape, x.min(), x.max())
# run inference
image_embeddings = model.get_image_features(**inputs)    

print(image_embeddings.shape)
"""