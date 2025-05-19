import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Dict, Any
import argparse
from PIL import Image

class ImageFeatureExtractor:
    """
    A class dedicated to extracting feature embeddings from images using an ONNX model.
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
            
        parser.add_argument("--feature-model-path", type=str, default="model_image.onnx",
                          help="Path to the ONNX feature extraction model file.")
        parser.add_argument("--feature-input-shape", type=parse_shape, default="224,112",
                          help="Input shape (Height,Width) for the feature model, comma-separated (default: '224,112').")
        parser.add_argument("--normalize-mean", type=float, default=0.5,
                          help="Mean value for image normalization.")
        parser.add_argument("--normalize-std", type=float, default=0.5,
                          help="Standard deviation for image normalization.")
        if "--use-cuda" not in parser._option_string_actions:
            parser.add_argument("--use-cuda", type=bool, default=False,
                              help="Whether to attempt using CUDA for inference.")
        return parser
    
    def __init__(self, 
                 feature_model_path: str,
                 feature_input_shape: Tuple[int, int] = (224, 112),
                 normalize_mean: float = 0.5,
                 normalize_std: float = 0.5,
                 use_cuda: bool = False, **kargs):
        """
        Initializes the ImageFeatureExtractor with the specified ONNX model and parameters.
        
        Args:
            model_path (str): Path to the ONNX feature extraction model file.
            input_shape (Tuple[int, int]): Expected input shape (Height, Width) for the model.
            normalize_mean (float): Mean value for image normalization.
            normalize_std (float): Standard deviation for image normalization.
            use_cuda (bool): Whether to attempt using CUDA for inference.
        """
        self.model_path = feature_model_path
        self.input_shape = feature_input_shape
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.use_cuda = use_cuda
        
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Initialized ImageFeatureExtractor with model: {feature_model_path}")
        print(f"Input Name: {self.input_name}, Output Name: {self.output_name}")
        print(f"Input Shape (H, W): {self.input_shape}")
        print(f"Normalization: mean={self.normalize_mean}, std={self.normalize_std}")
        print(f"Using CUDA: {self.use_cuda} (Providers: {self.session.get_providers()})")
    
    def _create_session(self) -> ort.InferenceSession:
        """
        Creates an ONNX Runtime session for feature extraction.
        
        Returns:
            ort.InferenceSession: Configured inference session.
        """
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Suppress verbose logs
        
        providers = []
        provider_options = []
        
        if self.use_cuda and ort.get_device() == 'GPU':
            try:
                providers.append('CUDAExecutionProvider')
                provider_options.append({'device_id': 0})
                print("CUDAExecutionProvider added for feature extraction.")
            except Exception as e:
                print(f"Warning: Could not initialize CUDAExecutionProvider. {e}")
                if 'CUDAExecutionProvider' in providers:
                    providers.remove('CUDAExecutionProvider')
                if provider_options:
                    provider_options.pop()
        
        providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        print(f"Attempting to create session with providers: {providers}")
        try:
            session = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=providers,
                provider_options=provider_options if any(p != {} for p in provider_options) else None
            )
            print(f"Session created successfully with providers: {session.get_providers()}")
            return session
        except Exception as e:
            print(f"Error creating ONNX session with providers {providers}: {e}")
            print("Falling back to default CPU provider.")
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(self.model_path, session_options, providers=providers)
            print(f"Session created successfully with providers: {session.get_providers()}")
            return session
    
    def preprocess_image(self, image) -> np.ndarray:
        """
        Preprocesses an image for feature extraction.
        
        Args:
            image: Input image (PIL.Image or np.ndarray in RGB format).
            
        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        # Convert to PIL Image if not already
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:  # Assume RGB
                image = Image.fromarray(image)
            elif image.ndim == 2:  # Grayscale
                image = Image.fromarray(image).convert('RGB')
        
        # Resize to target input shape (W, H)
        target_h, target_w = self.input_shape
        resized_img = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        np_img = np.array(resized_img, dtype=np.float32) / 255.0
        
        # Normalize: (pixel - mean) / std
        normalized_img = (np_img - self.normalize_mean) / self.normalize_std
        
        # Transpose from HWC to CHW format (3, H, W)
        chw_img = normalized_img.transpose((2, 0, 1))
        
        # Add batch dimension (1, 3, H, W)
        batch_img = np.expand_dims(chw_img, axis=0)
        batch_img = np.ascontiguousarray(batch_img)
        
        return batch_img
    
    def extract_features(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Extracts features from a preprocessed image tensor.
        
        Args:
            preprocessed_image (np.ndarray): Preprocessed image tensor with shape (1, 3, H, W).
            
        Returns:
            np.ndarray: Extracted feature vector.
        """
        try:
            features = self.session.run(
                [self.output_name],
                {self.input_name: preprocessed_image}
            )[0]
            return features[0]  # Return the first (and only) feature vector
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            return np.array([])  # Return empty array on error

    def load_image(self, img_path: str):
        img = Image.open(img_path)
        return self.load_pil_image(img)

    def load_pil_image(self, img: Image.Image):
        img = img.convert('RGB')
        img = self.preprocess_image(img)
        return img

    def image_embedding(self, img_path=None, image=None):
        if image is None:
            img = self.load_image(img_path)
        else:
            img = self.load_pil_image(image)
        return self.extract_features(img)


def parse_shape(shape_str: str) -> Tuple[int, int]:
    """
    Parses a 'height,width' string into a tuple of integers.
    
    Args:
        shape_str (str): String in format 'height,width'.
        
    Returns:
        Tuple[int, int]: Parsed height and width.
    """
    try:
        parts = shape_str.split(',')
        if len(parts) != 2:
            raise ValueError("Shape must be in 'height,width' format.")
        height = int(parts[0].strip())
        width = int(parts[1].strip())
        return height, width
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid shape format: {shape_str}. {e}")