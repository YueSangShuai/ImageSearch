import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import json
from typing import List, Tuple, Dict, Any
from det import PedestrianHeadDetector
import argparse
from model_image import ImageFeatureExtractor
from processor_enroll import ImageEnroller

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect persons in images and extract feature embeddings by ONNX.")

    ImageEnroller.add_arguments(parser)
    PedestrianHeadDetector.add_arguments(parser)
    ImageFeatureExtractor.add_arguments(parser)
    
    args = parser.parse_args()
    
    detector = PedestrianHeadDetector(**vars(args))
    feature_extractor = ImageFeatureExtractor(**vars(args))

    processor = ImageEnroller(detector=detector, feature_extractor=feature_extractor, **vars(args))
    processor.process()