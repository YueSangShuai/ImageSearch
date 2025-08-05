import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Dict, Any
import argparse
from PIL import Image
from transformers import AutoTokenizer

class TextFeatureExtractor:
    """
    A class dedicated to extracting feature embeddings from text using an ONNX model.
    This class focuses solely on the feature extraction process, independent of detection.
    """
    
    @staticmethod
    def add_arguments(parser=None):
        """
        Adds command-line arguments specific to feature extraction.
        
        Args:
            parser: Argument parser to add arguments to. If None, creates a new parser.
            
        Returns:
            argparse.ArgumentParser: Parser with added arguments.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Feature extraction using ONNX model.")
            
        parser.add_argument("--text-model-path", type=str, default="model_text.onnx",
                          help="Path to the ONNX feature extraction model file.")
        parser.add_argument("--tokenizer", type=str, default="tokenizer",
                          help="Path to the tokenizer configuration file.")
        parser.add_argument("--text-cls-token", type=int, default=-1,
                          help="CLS token ID for text processing. Default: -1 (no CLS token)")
                          
        if "--use-cuda" not in parser._option_string_actions:
            parser.add_argument("--use-cuda", type=bool, default=False,
                              help="Whether to attempt using CUDA for inference.")
        return parser
    
    def __init__(self, 
                 text_model_path: str,
                 tokenizer: str = "tokenizer",
                 text_cls_token: int = -1,
                 use_cuda: bool = False, **kargs):
        self.use_cuda = use_cuda
        self.text_len = 128

        print(f"Loading text ONNX model: {text_model_path}")
        self.text_session = self._create_text_session(text_model_path)
        self.text_input_name = self.text_session.get_inputs()[0].name
        print(self.text_session.get_inputs())
        self.text_len = self.text_session.get_inputs()[0].shape[1]
        # Assuming the embedding is the first output. Adjust if your model differs.
        self.text_output_name = self.text_session.get_outputs()[0].name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.text_cls_token = text_cls_token
        if self.text_cls_token != -1:
            self.tokenizer.cls_token_id = self.text_cls_token
            self.tokenizer.sep_token_id = 0
        print("Initialization complete.")
        print(f"  Text model input: '{self.text_input_name}', output: '{self.text_output_name}'")
        print(f"  Text model input shape: {self.text_session.get_inputs()[0].shape}")
        print(f"  Text model output shape: {self.text_session.get_outputs()[0].shape}")
        print(f"  Using CUDA for text model: {self.use_cuda} (Providers: {self.text_session.get_providers()})")
    
    def _create_text_session(self, model_path: str) -> ort.InferenceSession:
        """Creates the ONNX Runtime session for the text model."""
        so = ort.SessionOptions()
        so.log_severity_level = 3 # Suppress verbose logs

        providers = []
        provider_options = []

        if self.use_cuda and ort.get_device() == 'GPU':
             try:
                providers.append('CUDAExecutionProvider')
                provider_options.append({'device_id': 0})
             except Exception as e:
                print(f"Warning: Could not initialize CUDAExecutionProvider for text model. {e}")
                if 'CUDAExecutionProvider' in providers: providers.remove('CUDAExecutionProvider')
                if provider_options: provider_options.pop()

        providers.append('CPUExecutionProvider')
        provider_options.append({})

        print(f"  Attempting to create text session with providers: {providers}")
        try:
            session = ort.InferenceSession(model_path, so, providers=providers, provider_options=provider_options if any(p!={} for p in provider_options) else None)
            return session
        except Exception as e:
            print(f"  Error creating text ONNX session with providers {providers}: {e}")
            print("  Falling back to default CPU provider for text model.")
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, so, providers=providers)
            return session

    def _preprocess_text(self, text: str) -> Dict[str, np.ndarray]:
        # text_bytes = text_to_bytes(text, self.text_len)
        # return text_bytes[None, ...]
        if self.text_cls_token != -1:
            text = f"<s>{text}"
        ret = self.tokenizer(text, max_length=1024, truncation=True, return_tensors='np', add_special_tokens=False).input_ids
        print(ret)
        return ret

    def text_embedding(self, query_texts: str) -> np.ndarray:
        """Generates a normalized embedding vector for the input query_texts."""
        if not query_texts:
            return np.array([])

        # 1. Preprocess text
        model_inputs = self._preprocess_text(query_texts)
        print(f"  Preprocessed text: {model_inputs.shape}, {model_inputs.dtype}")
        # 2. Run ONNX Inference
        outputs = self.text_session.run([self.text_output_name], {self.text_input_name: model_inputs})
        text_embedding = outputs[0]
        print(f"  Text embedding: {text_embedding.shape}, {text_embedding.dtype}")
        if text_embedding.ndim > 1 and text_embedding.shape[0] == 1:
            text_embedding = text_embedding.squeeze(0)

        # 3. Normalize the embedding (L2 norm)
        norm = np.linalg.norm(text_embedding)
        if norm == 0:
            return text_embedding # Avoid division by zero
        normalized_embedding = text_embedding / norm

        return normalized_embedding

