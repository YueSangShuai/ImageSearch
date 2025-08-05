class ImageTextDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["image_path"]
        txt_path = self.data[idx]["text_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"加载失败：{img_path}, {txt_path}, 错误：{e}")
            img = None
            text = None

        return img_path, txt_path, img, text


def collate_fn(batch):
    imgs, texts, img_paths, txt_paths = [], [], [], []
    for img_path, txt_path, img, text in batch:
        if img is not None and text is not None:
            imgs.append(img)
            texts.append(text)
            img_paths.append(img_path)
            txt_paths.append(txt_path)
    return imgs, texts, img_paths, txt_paths


def tensor_cosine_similarity(tensor1, tensor2, dim=-1, eps=1e-8):
    dot_product = torch.sum(tensor1 * tensor2, dim=dim)
    norm1 = torch.norm(tensor1, dim=dim)
    norm2 = torch.norm(tensor2, dim=dim)
    return dot_product / (norm1 * norm2 + eps)


def run_inference(model, processor, tokenizer, dataloader, device):
    similarities = []

    for imgs, texts, _, _ in tqdm(dataloader, desc="Inferencing"):
        if len(imgs) == 0:
            continue

        # 图像处理
        img_inputs = processor(images=imgs, return_tensors="pt").to(device)
        img_inputs = {k: v.to(dtype=torch.bfloat16) for k, v in img_inputs.items()}

        with torch.inference_mode():
            img_embs = model.module.encode_images(**img_inputs, normalize=True)

        # 文本处理
        text_inputs = tokenizer(
            texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            text_embs = model.module.encode_texts(**text_inputs, normalize=True)

        # 计算余弦相似度
        batch_sim = tensor_cosine_similarity(img_embs, text_embs)
        similarities.extend(batch_sim.cpu().tolist())

    return similarities


def main():
    # ===== 初始化 =====
    json_file = "/data/yuesang/LLM/contrastors/data/MALS/add/test.json"
    model_path = "c"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型、tokenizer、processor
    base_model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, optimized=True
    )
    model = torch.nn.DataParallel(base_model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # ===== 构建数据加载器 =====
    dataset = ImageTextDataset(json_file)
    dataloader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=16, collate_fn=collate_fn
    )

    # ===== 执行推理 =====
    similarities = run_inference(model, processor, tokenizer, dataloader, device)

    # ===== 输出结果 =====
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\n✅ 总图文对数：{len(similarities)}")
        print(f"✅ 平均余弦相似度：{avg_sim:.4f}")
    else:
        print("❌ 未能成功处理任何图文对。")


if __name__ == "__main__":
    main()