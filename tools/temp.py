import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# 1. 模型加载
def load_nomic_model():
    """从Hugging Face加载nomic-embed-text-v1.5模型"""
    model_name = "/data/yuesang/LLM/VectorIE/models/nomic/nomic-embed-text-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
    model.eval()
    
    def get_embeddings(texts):
        with torch.no_grad():
            texts = [f"search_query: {text}" for text in texts]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192,
                
            )
            outputs = model(** inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            return torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()
    
    return get_embeddings

def load_mexma_model():
    """加载mexma-siglip2模型（返回tokenizer和嵌入函数，修复解包错误）"""
    model_name = "/data/yuesang/LLM/VectorIE/models/mexma-siglip2"
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
    model.eval()
    
    def get_embeddings(texts):
        with torch.no_grad():
            inputs = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            )
            outputs = model.encode_texts(**inputs, normalize=True)  # 移除return_dict参数
            if outputs.dtype == torch.bfloat16:
                outputs = outputs.to(dtype=torch.float32)
            # 兼容元组和字典输出
            # last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            # cls_emb = last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(outputs, dim=-1).cpu().numpy()
    
    # 关键修复：返回tokenizer和嵌入函数的元组
    return tokenizer, get_embeddings

# 初始化模型（现在可以正确解包）
nomic_embed_fn = load_nomic_model()
mexma_tokenizer, mexma_embed_fn = load_mexma_model()  # 这里需要两个返回值

# 2. 数据集加载（保持不变）
def load_sts_dataset():
    try:
        dataset = load_dataset("mteb/stsbenchmark-sts")["test"]
        sentences1 = [item["sentence1"] for item in dataset if "sentence1" in item]
        sentences2 = [item["sentence2"] for item in dataset if "sentence2" in item]
        scores = [item["score"] for item in dataset if "score" in item]
        
        if len(sentences1) < 100 or len(sentences2) < 100:
            raise ValueError("STS数据集样本不足")
        return sentences1, sentences2, scores
    except Exception as e:
        print(f"STS数据集加载失败: {e}，使用备用数据")
        sentences1 = ["a cat is sitting", "a dog is running", "the sky is blue"] * 50
        sentences2 = ["a feline is seated", "a canine is moving", "the air is blue"] * 50
        scores = [4.5, 4.2, 2.1] * 50
        return sentences1, sentences2, scores

def load_classification_dataset():
    texts = []
    labels = []
    try:
        dataset = load_dataset("ag_news")["train"]
        if hasattr(dataset, 'to_pandas'):
            df = dataset.to_pandas()
            if "text" in df.columns and "label" in df.columns:
                texts = df["text"].dropna().tolist()[:10000]
                labels = df["label"].dropna().astype(int).tolist()[:10000]
        
        if not texts:
            for item in dataset[:10000]:
                if isinstance(item, dict) and "text" in item and "label" in item:
                    texts.append(str(item["text"]))
                    labels.append(int(item["label"]))
                elif isinstance(item, (list, tuple)) and len(item)>=2:
                    texts.append(str(item[1]))
                    labels.append(int(item[0]))
        
        if len(texts) < 100:
            raise ValueError("ag_news数据集有效样本不足")
    
    except Exception as e:
        print(f"ag_news数据集加载失败: {e}，使用备用数据集")
        sample_texts = [
            "I love this movie", "This is a great film", "Wonderful experience",
            "Terrible movie", "I hate this film", "Awful experience"
        ]
        sample_labels = [1, 1, 1, 0, 0, 0]
        texts = sample_texts * 200
        labels = sample_labels * 200
    
    if len(texts) == 0 or len(labels) == 0 or len(texts) != len(labels):
        raise RuntimeError("无法加载有效的分类数据集")
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载数据
sts_s1, sts_s2, sts_scores = load_sts_dataset()
print(f"成功加载STS数据集: {len(sts_s1)}个样本")

clf_train_texts, clf_test_texts, clf_train_labels, clf_test_labels = load_classification_dataset()
print(f"成功加载分类数据集: 训练集{len(clf_train_texts)}个，测试集{len(clf_test_texts)}个")

# 3. 语义相似度测试
def test_semantic_similarity(embed_fn, name, s1, s2, true_scores):
    print(f"\n=== 测试 {name} 语义相似度 ===")
    emb1 = np.array(embed_fn(s1))
    emb2 = np.array(embed_fn(s2))
    
    cos_sim = np.diag(np.dot(emb1, emb2.T) / (
        np.linalg.norm(emb1, axis=1)[:, None] * 
        np.linalg.norm(emb2, axis=1)[None, :]
    ))
    
    corr, p_value = pearsonr(cos_sim, true_scores)
    print(f"皮尔逊相关系数: {corr:.4f} (p值: {p_value:.4f})")
    return corr

# 4. 文本分类测试
def test_text_classification(embed_fn, name, train_texts, train_labels, test_texts, test_labels):
    print(f"\n=== 测试 {name} 文本分类 ===")
    train_emb = np.array(embed_fn(train_texts))
    test_emb = np.array(embed_fn(test_texts))
    
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(train_emb, train_labels)
    
    pred = clf.predict(test_emb)
    acc = accuracy_score(test_labels, pred)
    print(f"分类准确率: {acc:.4f}")
    return acc

# 执行测试
nomic_sts_corr = test_semantic_similarity(nomic_embed_fn, "nomic-embed-text-v1.5", sts_s1, sts_s2, sts_scores)
mexma_sts_corr = test_semantic_similarity(mexma_embed_fn, "mexma-siglip2", sts_s1, sts_s2, sts_scores)

nomic_clf_acc = test_text_classification(nomic_embed_fn, "nomic-embed-text-v1.5", 
                                         clf_train_texts, clf_train_labels, 
                                         clf_test_texts, clf_test_labels)
mexma_clf_acc = test_text_classification(mexma_embed_fn, "mexma-siglip2", 
                                         clf_train_texts, clf_train_labels, 
                                         clf_test_texts, clf_test_labels)

# 结果汇总
print("\n=== 模型精度对比汇总 ===")
print(f"语义相似度（皮尔逊相关系数）：")
print(f"nomic-embed-text-v1.5: {nomic_sts_corr:.4f}")
print(f"mexma-siglip2: {mexma_sts_corr:.4f}")
print(f"\n文本分类准确率：")
print(f"nomic-embed-text-v1.5: {nomic_clf_acc:.4f}")
print(f"mexma-siglip2: {mexma_clf_acc:.4f}")
