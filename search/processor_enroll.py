from PIL import Image
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import json
from typing import List, Tuple, Dict, Any
from det import PedestrianHeadDetector
import argparse

class ImageEnroller:
    """
    Manages the process of detecting pedestrians in images and extracting feature embeddings
    using a dedicated ImageFeatureExtractor for feature extraction.
    """
    def add_arguments(parser):
        parser.add_argument("--image-dir", type=str, default="xm_images",
                            help="Path to the directory containing images to process.")
        parser.add_argument("--output-prefix", type=str, default="output/person_features",
                            help="Prefix for the output files (e.g., 'output/person_features'). '.npy' and '.json' will be appended.")
    def __init__(self,
                 detector: PedestrianHeadDetector,
                 feature_extractor: Any,
                 image_dir: str,
                 output_prefix: str,
                 **kargs):
        """
        Initializes the ImageEnroller with a detector and a feature extractor.
        
        Args:
            detector (PedestrianHeadDetector): An initialized instance for detecting persons.
            feature_extractor (ImageFeatureExtractor): An initialized instance for extracting features.
            image_dir (str): Path to the directory containing images to process.
            output_prefix (str): Prefix for the output files (e.g., 'output/person_features'). '.npy' and '.json' will be appended.
            **kargs: Additional keyword arguments.
        """
        if not isinstance(detector, PedestrianHeadDetector):
            raise TypeError("Detector must be an instance of PedestrianHeadDetector")

        self.detector = detector
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir
        self.output_prefix = output_prefix
        self.all_features: List[np.ndarray] = []
        self.all_metadata: List[Dict[str, Any]] = []

        print("Initialized ImageEnroller with provided detector and feature extractor.")

    def process_directory(self, image_dir: str):
        """
        Processes all images in a directory to detect persons and extract features.
        
        Args:
            image_dir (str): Path to the directory containing images.
        """
        if not os.path.isdir(image_dir):
            print(f"Error: Directory not found: {image_dir}")
            return

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        total_files = len(image_files)
        print(f"Found {total_files} images in {image_dir}. Starting processing...")

        # Clear previous results if any
        self.all_features = []
        self.all_metadata = []
        start_time = time.time()
        total_features_extracted = 0

        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)
            progress_percent = (i + 1) / total_files * 100
            sys.stdout.write(f"\rProcessing image {i+1}/{total_files} ({progress_percent:.1f}%): {filename}...")
            sys.stdout.flush()

            try:
                img = Image.open(image_path).convert('RGB')
                if img is None:
                    continue
            except Exception as e:
                sys.stdout.write(f"\nError processing image {filename}: {e}\n")
                continue

            # 1. Detect Persons (Bodies)
            detections = self.detector.detect(img)
            person_detections = [d for d in detections if d['class_name'] == 'body']

            if not person_detections:
                continue

            img_w, img_h = img.size  # PIL Image 使用 size 属性 (width, height)
            for det in person_detections:
                x1, y1, x2, y2 = det['box']
                # Ensure coordinates are valid and crop is not empty
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x1 >= x2 or y1 >= y2:
                    continue

                # 2. Crop the detected person
                crop = img.crop((x1, y1, x2, y2))
                if crop.size == 0:
                    continue

                # 3. Preprocess and extract features using ImageFeatureExtractor
                # preprocessed_crop = self.feature_extractor.preprocess_image(crop)
                # feature_vec = self.feature_extractor.extract_features(preprocessed_crop)
                feature_vec = self.feature_extractor.image_embedding(image=crop)

                # 4. Store feature and metadata if extraction was successful
                if feature_vec.size > 0:
                    meta = {
                        'image_path': filename,
                        'bbox': [x1, y1, x2, y2],
                        'detection_score': float(det['score'])
                    }
                    self.all_features.append(feature_vec)
                    self.all_metadata.append(meta)
                    total_features_extracted += 1


        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nFinished processing {total_files} images in {total_time:.2f} seconds.")
        print(f"Extracted {total_features_extracted} features in total.")

    def get_results(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Returns the extracted features and metadata.
        
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]:
                - A NumPy array of all extracted features (N, feature_dim).
                - A list of metadata dictionaries corresponding to each feature vector.
        """
        if not self.all_features:
            return np.array([]), []
        return np.vstack(self.all_features), self.all_metadata

    def save_results(self, output_prefix: str):
        """
        Saves the extracted features and metadata to files.
        
        Args:
            output_prefix (str): The prefix for the output files (e.g., 'output/extracted_data').
                                 '.npy' and '.json' will be appended.
        """
        if not self.all_features:
            print("No features were extracted. Nothing to save.")
            return

        features_array = np.vstack(self.all_features)
        features_path = f"{output_prefix}_features.npy"
        metadata_path = f"{output_prefix}_metadata.json"

        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        try:
            np.save(features_path, features_array)
            print(f"Saved {features_array.shape[0]} features to {features_path}")
            with open(metadata_path, 'w') as f:
                json.dump(self.all_metadata, f, indent=4)
            print(f"Saved metadata for {len(self.all_metadata)} features to {metadata_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def process(self):
        print("\nStarting directory processing...")
        self.process_directory(self.image_dir)
        print("\nSaving results...")
        self.save_results(self.output_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect persons in images and extract feature embeddings.")
    FeatureExtractor.add_arguments(parser)
    PedestrianHeadDetector.add_arguments(parser)
    ImageFeatureExtractor.add_arguments(parser)
    
    args = parser.parse_args()
    
    detector = PedestrianHeadDetector(**vars(args))
    feature_extractor = ImageFeatureExtractor(**vars(args))

    processor = FeatureExtractor(detector=detector, feature_extractor=feature_extractor, **vars(args))
    processor.process()