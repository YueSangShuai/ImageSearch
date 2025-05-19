import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
model = model.cuda()

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("/data/yuesang/LLM/VectorIE/classifier_label/image-3.png")).unsqueeze(0).cuda()
text = tokenizer(["a diagram", "a dog", "a cat"]).cuda()

with torch.no_grad(), torch.autocast("cuda"):
    
    image_features, text_features, logit_scale = model(image, text)
    text_probs = (logit_scale * image_features @ text_features.T)
    print(logit_scale)
    image_features=model.encode_image(image,normalize=True)
    text_features=model.encode_text(text,normalize=True)
    print((image_features @ text_features.T))


