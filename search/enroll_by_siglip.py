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
from siglip import ImageTextEmbedder
from processor_enroll import ImageEnroller

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect persons in images and extract feature embeddings by Siglip Model.")

    ImageEnroller.add_arguments(parser)
    PedestrianHeadDetector.add_arguments(parser)
    ImageTextEmbedder.add_arguments(parser)
    
    args = parser.parse_args()
    
    detector = PedestrianHeadDetector(**vars(args))
    feature_extractor = ImageTextEmbedder(**vars(args))

    processor = ImageEnroller(detector=detector, feature_extractor=feature_extractor, **vars(args))
    processor.process()