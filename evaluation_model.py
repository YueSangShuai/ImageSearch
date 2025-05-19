import torch
import torch.nn.functional as F
import os
import json
from tools.roc import ROC
from model import Model
from data import PairedImageTextDataset
from torch.utils.data import Dataset, DataLoader

def evaluate_db(val_dataloader, model, device):
    i = 0
    roc = ROC()
    print("start evaluate")
    model.eval()
    datas = []
    for items in val_dataloader:
        file_paths = items['file_path']
        person_ids = items['id']
        image_embeddings, text_embeddings = model(items)
        for image_embedding, text_embedding, person_id, file_path in zip(image_embeddings, text_embeddings, person_ids, file_paths):
            id_name, fn_name = os.path.split(file_path)[-2:]
            fn_name = f"{person_id}/{id_name}_{fn_name}"
            score = (image_embedding*text_embedding).sum().item()
            roc.add_name(fn_name, fn_name+"_caption", score)
            for person_id2, f2, image_embedding2, text_embedding2 in datas:
                if person_id2 == person_id:
                    continue
                score = (image_embedding*text_embedding2).sum().item()
                roc.add_name(fn_name, f2+"_caption", score)
            datas.append((person_id, fn_name, image_embedding, text_embedding))
            i += 1
            print(f"{i}: {file_path}")
    roc.stat(1)
    
def load_trained_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    args = state['args']
    model = Model(args).to(device)
    model.load_state_dict(state['state_dict'])
    val_dataset = PairedImageTextDataset(args.data, args.embedding_path, text_len=args.max_seq_length, train=False, image_size=args.image_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        collate_fn=val_dataset.collate_fn
    )
    return model, val_dataloader
    

if __name__ == '__main__':
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 2:
        print("Usage: python evaluation_model.py <model_path> [<device>]")
        sys.exit(1)
    if len(sys.argv) > 2:
        device = torch.device(sys.argv[2])
    model, val_dataloader = load_trained_model(sys.argv[1], device)
    with torch.no_grad():
        evaluate_db(val_dataloader, model, device)
