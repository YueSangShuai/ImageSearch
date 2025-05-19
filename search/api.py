import numpy as np
import onnxruntime as ort
import json
import os
import sys
import argparse
from typing import List, Dict, Tuple, Any
import time
from transformers import AutoTokenizer
from model_image import ImageFeatureExtractor
from model_text import TextFeatureExtractor

class ImageTextSearch:
    """
    Performs text-to-image search using pre-computed image embeddings
    and a text embedding ONNX model.
    """

    @staticmethod
    def add_arguments(parser=None):
        if parser is None:
            import argparse
            parser = argparse.ArgumentParser()
        parser.add_argument("--image-features-path", type=str, default="collection/person_features_features.npy",
                            help="Path to the .npy file containing image feature vectors.")
        parser.add_argument("--image-metadata-path", type=str, default="collection/person_features_metadata.json",
                            help="Path to the .json file containing image metadata.")
        if "--use-cuda" not in parser._option_string_actions:
            parser.add_argument("--use-cuda", action='store_true',
                            help="Use CUDA for the text model inference (if available).")
        return parser

    def __init__(self,
                 text_extractor: Any,
                 image_features_path: str=None,
                 image_metadata_path: str=None,
                 use_cuda: bool = None, **kwargs):

        print("Initializing ImageTextSearch...")
        self.collection_name = image_features_path.split("/")[-1].split(".")[0]
        self.text_extractor = text_extractor

        # --- Load Image Features & Metadata ---
        print(f"Loading image features: {image_features_path}")
        try:
            self.image_features = np.load(image_features_path)
            if self.image_features.ndim != 2:
                raise ValueError(f"Image features should be a 2D array (N, dim), but got shape {self.image_features.shape}")
            print(f"  Loaded {self.image_features.shape[0]} image features with dimension {self.image_features.shape[1]}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load image features from {image_features_path}: {e}")

        print(f"Loading image metadata: {image_metadata_path}")
        try:
            with open(image_metadata_path, 'r') as f:
                self.image_metadata = json.load(f)
            print(f"  Loaded metadata for {len(self.image_metadata)} images.")
        except Exception as e:
            raise RuntimeError(f"Failed to load image metadata from {image_metadata_path}: {e}")

        # --- Validate consistency ---
        if self.image_features.shape[0] != len(self.image_metadata):
            print(f"Warning: Number of image features ({self.image_features.shape[0]}) does not match number of metadata entries ({len(self.image_metadata)}). Check your input files.")
            # Decide how to handle: raise error, trim, or proceed with caution?
            # Let's trim to the minimum length for now.
            min_len = min(self.image_features.shape[0], len(self.image_metadata))
            print(f"  Trimming to {min_len} entries.")
            self.image_features = self.image_features[:min_len, :]
            self.image_metadata = self.image_metadata[:min_len]

        if self.image_features.shape[0] == 0:
             print("Warning: No image features loaded. Search will not return results.")

        # --- Pre-normalize image features for efficient cosine similarity ---
        print("Normalizing image features...")
        norms = np.linalg.norm(self.image_features, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms[norms == 0] = 1e-10
        self.normalized_image_features = self.image_features / norms


    def search(self, query_text: str, top_n: int = 5, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Searches for the top_n most similar images to the query_text.

        Args:
            query_text (str): The text query.
            top_n (int): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                                  'metadata': The metadata of the matched image.
                                  'score': The cosine similarity score.
                                  Sorted by score in descending order.
        """
        if self.normalized_image_features.shape[0] == 0:
            print("No image features loaded, cannot perform search.")
            return []

        if top_n <= 0:
            print("Warning: top_n must be positive.")
            return []

        # 1. Get normalized embedding for the query text
        text_vector = self.text_extractor.text_embedding(query_text)
        if text_vector.size == 0:
            print("Could not generate text embedding for the query.")
            return []
        # Ensure text_vector is 1D for dot product
        text_vector = text_vector.flatten()

        # Check dimension compatibility
        if text_vector.shape[0] != self.normalized_image_features.shape[1]:
            print(f"Error: Text embedding dimension ({text_vector.shape[0]}) does not match image feature dimension ({self.normalized_image_features.shape[1]}).")
            return []

        # 2. Calculate Cosine Similarities
        # Since both text_vector and image_features are L2 normalized,
        # the dot product is equivalent to cosine similarity.
        similarities = self.normalized_image_features @ text_vector

        # 3. Get Top N results
        num_results = min(top_n, len(similarities)) # Handle cases where top_n > num_images
        top_indices = np.argsort(similarities)[::-1][:num_results] # Get indices sorted descending

        # 4. Format results
        results = []
        for idx in top_indices:
            results.append({
                'metadata': self.image_metadata[idx],
                'score': float(similarities[idx]) # Convert numpy float to standard float
            })

        return results

    def count_records(self):
        return len(self.image_metadata)
    def get_collections(self):
        return [self.collection_name]
    def set_collection(self, collection_name):
        self.collection_name = collection_name
    def search_in_collections(self, query, top_k, threshold, collections, return_payload=True):
        start_time = time.time()
        ret =  self.search(query, top_k, threshold)
        search_time = time.time() - start_time
        return ret, search_time
        

# Example Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform text-to-image search using ONNX models and pre-computed features.")
    parser.add_argument("--image-root", type=str, default="xm_images", 
                        help="Path to the root directory containing image files.")
    parser.add_argument("--queries", type=str, nargs='+', default=["a girl wearing red coat", "a man wearing a gray pands"],
                        help="The text query to search for.")
    parser.add_argument("--top-n", type=int, default=2,
                        help="Number of top results to return (default: 5).")
    ImageTextSearch.add_arguments(parser)    # Add arguments for ImageTextSearch
    TextFeatureExtractor.add_arguments(parser)
    args = parser.parse_args()

    # --- Initialize Search Engine ---
    text_extractor = TextFeatureExtractor(**args.__dict__)
    search_engine = ImageTextSearch(text_extractor=text_extractor, **args.__dict__)

    # --- Perform Search ---
    for query in args.queries:
        print("\n"+"-"*10)
        print(f"Query: '{query}'")
        start_time = time.time() # Import time if not already done
        search_results = search_engine.search(query, top_n=args.top_n)
        end_time = time.time()
        print(f"\tSearch completed in {end_time - start_time:.4f} seconds.")

        if search_results:
            for i, result in enumerate(search_results):
                print(f"{i+1}. Score: {result['score']:.4f}")
                meta = result['metadata']
                print(f"   Image: {os.path.join(args.image_root, meta.get('image_path', 'N/A'))}") # Use .get for safety
                print(f"   BBox: {meta.get('bbox', 'N/A')}")
        else:
            print("\nNo results found.")
