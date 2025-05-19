import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os
import json
import requests
import time

class JSONLDataset(Dataset):
    def __init__(self, json_fn, text_len=200, max_count=0):
        self.max_count = max_count
        self.text_len = text_len
        self.data = self.load_data(json_fn)
        # print(f"Loaded {len(self.data)} samples from {json_fn}")
        # print(f"First sample: {self.data[0]}")
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
                if self.max_count > 0 and len(samples) >= self.max_count:
                    break
        return samples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data))
        item = self.data[i]
        captions = item['text'].split('. ')
        zh_captions = item['zh'].split('。 ')
        if len(captions) != len(zh_captions): 
            ret = {
                "idx": idx,
                'text': item['text'],
                'zh': item['zh'],
            }    
        else:        
            i = np.random.randint(0, len(captions))
            ret = {
                "idx": idx,
                'text': captions[i],
                'zh': zh_captions[i],
            }
        return ret

class ParquetDataset(Dataset):
    def __init__(self, parquet_fn, text_len=200, max_count=0):
        self.text_len = text_len
        self.max_count = max_count
        self.data = self.load_data(parquet_fn)
    def load_data(self, path):
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=['human_caption'])
        print(path, table.num_rows)
        # Convert to pandas DataFrame for easier indexing
        df = table.to_pandas()
        if self.max_count > 0:
            df = df.head(self.max_count)
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        caption = item['human_caption'][0]
        return {
            "idx": idx,
            'text': caption
        }
    def translate(self, text):
        url = "http://127.0.0.1:8989/translate"
        headers = {"Authorization": "shiliangfadongji"}
        data = {"from": "en", "to": "zh", "text": text}
        response = requests.post(url, headers=headers, json=data)
        query_text = response.json().get("result", text)
        return query_text
    def write(self, path):
        i = 0
        with open(path, 'a', encoding='utf-8') as f:
            for i in range(len(self)):
                item = self[i]
                item['zh'] = self.translate(item['text'])
                s = json.dumps(item, ensure_ascii=False)
                f.write(s)
                f.write('\n')
                if i%100==0:
                    print(f"{i} items written")

if __name__ == '__main__':
    # DB = JSONLDataset('dataset/pretrain_hq.jsonl', 256)
    DB = ParquetDataset("/data/datasets/HumanCaption-10M/HumanCaption-10M.parquet", 256, max_count=0)
    print(len(DB))
    DB.write("d.jsonl")
    for i in range(10):
        j = np.random.randint(0, min(10000,len(DB)))
        print(f"{i}:{j}")
        data = DB[j]
        for k, v in data.items():
            print(f"\t{k}: {v.shape if isinstance(v, torch.Tensor) else (len(v), v) if type(v) in (list, tuple, str) else v}")
