import argparse
import torch

def parse_args(additional_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="mexma-siglip2")
    parser.add_argument('--category', type=str, default='person')
    parser.add_argument('--qdrant_host', type=str, default='localhost')
    parser.add_argument('--qdrant_port', type=int, default=6333)
    parser.add_argument('--collection_name', type=str, default='sl2_person_embeddings')

    parser.add_argument('--threshold', default=0.15, type=float)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    if additional_args is not None: additional_args(parser)

    return parser.parse_args()
