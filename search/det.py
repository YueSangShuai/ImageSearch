import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageDraw, ImageFont

class PedestrianHeadDetector:
    """
    A class to detect pedestrians (body) and heads using an ONNX model.

    Attributes:
        model_path (str): Path to the ONNX model file.
        input_shape (Tuple[int, int]): The expected input shape (height, width) for the model.
        class_names (Dict[float, str]): Mapping from class ID (float) to class name (str).
        score_threshold (float): Minimum confidence score to keep a detection.
        session (ort.InferenceSession): The ONNX Runtime inference session.
        input_name (str): The name of the model's input node.
        output_name (str): The name of the model's output node.
    """
    input_shape: Tuple[int, int] = (640, 640)
    class_names: Dict[float, str] = {0.0: 'body', 1.0: 'head'}

    @staticmethod
    def add_arguments(parser=None):
        if parser is None: 
            import argparse
            parser = argparse.ArgumentParser(description="Perform person/head detection using ONNX model.")
        parser.add_argument('--person_model', type=str, help='Path to ONNX model file', default="ph640-w640-v11-ph3.onnx")
        parser.add_argument('--person_threshold', type=float, default=0.4, 
                        help='Detection confidence threshold for person (default: 0.4)')
        if "--use-cuda" not in parser._option_string_actions:
            parser.add_argument("--use-cuda", action='store_true',
                            help="Use CUDA for the model inference (if available).")
        return parser

    def __init__(self,
                 person_model: str = None,
                 person_threshold: float = None,
                 use_cuda: bool = None, **kwargs):
        """
        Initializes the PedestrianHeadDetector.

        Args:
            person_model (str): Path to the ONNX model file.
            person_threshold (float): Minimum confidence score for detections. Defaults to 0.4.
            use_cuda (bool): Whether to attempt using CUDA for inference. Defaults to False.
        """
        self.model_path = person_model
        self.score_threshold = person_threshold
        self.use_cuda = use_cuda

        self.session = self._create_inference_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Initialized detector with model: {person_threshold}")
        print(f"Input Name: {self.input_name}, Output Name: {self.output_name}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Class Names: {self.class_names}")
        print(f"Score Threshold: {self.score_threshold}")
        print(f"Using CUDA: {self.use_cuda} (Providers: {self.session.get_providers()})")

    def _create_inference_session(self) -> ort.InferenceSession:
        """Creates the ONNX Runtime inference session."""
        so = ort.SessionOptions()
        so.log_severity_level = 3  # Suppress verbose logs

        providers = []
        provider_options = []

        if self.use_cuda and ort.get_device() == 'GPU':
            try:
                providers.append('CUDAExecutionProvider')
                provider_options.append({'device_id': 0})
                print("CUDAExecutionProvider added.")
            except Exception as e:
                print(f"Warning: Could not initialize CUDAExecutionProvider. {e}")
                providers.remove('CUDAExecutionProvider')
                provider_options.pop()

        providers.append('CPUExecutionProvider')
        provider_options.append({})

        print(f"Attempting to create session with providers: {providers}")
        try:
            session = ort.InferenceSession(self.model_path, so, providers=providers, provider_options=provider_options if any(provider_options) else None)
            print(f"Session created successfully with providers: {session.get_providers()}")
            return session
        except Exception as e:
            print(f"Error creating ONNX session with providers {providers}: {e}")
            print("Falling back to default CPU provider.")
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(self.model_path, so, providers=providers)
            print(f"Session created successfully with providers: {session.get_providers()}")
            return session

    def _preprocess(self, img: Image.Image) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocesses the input image using letterboxing.

        Args:
            img (Image.Image): Input image in PIL format.

        Returns:
            Tuple[np.ndarray, float, Tuple[float, float]]:
                - Preprocessed image tensor (1, 3, H, W) ready for the model.
                - Scaling ratio (r).
                - Padding (dw, dh).
        """
        im_rgb = img.convert('RGB')
        shape = im_rgb.size  # current shape [width, height]
        new_shape = self.input_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape != (new_unpad[1], new_unpad[0]):  # resize
            im_resized = im_rgb.resize((new_unpad[0], new_unpad[1]), resample=Image.Resampling.LANCZOS)
        else:
            im_resized = im_rgb

        # Create a new image with padding
        im_padded = Image.new('RGB', (new_shape[1], new_shape[0]), color=(114, 114, 114))
        im_padded.paste(im_resized, (int(round(dw - 0.1)), int(round(dh - 0.1))))

        # Convert to numpy array and then to NCHW format
        image_tensor = np.array(im_padded).transpose((2, 0, 1))  # HWC to CHW
        image_tensor = np.expand_dims(image_tensor, 0)  # Add batch dimension -> NCHW
        image_tensor = np.ascontiguousarray(image_tensor)

        # Normalize (0-255 -> 0.0-1.0) and convert to float32
        image_tensor = image_tensor.astype(np.float32) / 255.0

        return image_tensor, r, (dw, dh)

    def _postprocess(self, outputs: np.ndarray, ratio: float, dwdh: Tuple[float, float], original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Postprocesses the model outputs to get final detections.

        Args:
            outputs (np.ndarray): Raw output from the ONNX model. Expected shape [N, 7]
            ratio (float): Scaling ratio used during preprocessing.
            dwdh (Tuple[float, float]): Padding (dw, dh) added during preprocessing.
            original_shape (Tuple[int, int]): Original image shape (height, width).

        Returns:
            List[Dict]: A list of detected objects.
        """
        detections = []
        if outputs is None or len(outputs) == 0:
            return detections

        # Adjust coordinates back to original image space
        outputs[:, 1:5] -= np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]])  # Adjust for padding
        outputs[:, 1:5] /= ratio  # Adjust for scaling

        # Clip coordinates to image boundaries
        orig_h, orig_w = original_shape
        outputs[:, 1] = np.clip(outputs[:, 1], 0, orig_w)  # x1
        outputs[:, 2] = np.clip(outputs[:, 2], 0, orig_h)  # y1
        outputs[:, 3] = np.clip(outputs[:, 3], 0, orig_w)  # x2
        outputs[:, 4] = np.clip(outputs[:, 4], 0, orig_h)  # y2

        for i, (*_, x1, y1, x2, y2, class_id, score) in enumerate(outputs):
            if score < self.score_threshold:
                continue

            box = [int(x1), int(y1), int(x2), int(y2)]
            class_name = self.class_names.get(float(class_id), f"Unknown_{class_id}")

            detections.append({
                'box': box,
                'score': float(score),
                'class_id': float(class_id),
                'class_name': class_name
            })

        return detections

    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Performs detection on a single image.

        Args:
            image (Image.Image): Input image in PIL format.

        Returns:
            List[Dict]: A list of detected objects.
        """
        if image is None:
            print("Error: Input image is None.")
            return []

        original_shape = image.size[::-1]  # (height, width)
        im_tensor, ratio, dwdh = self._preprocess(image)

        # Run inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: im_tensor})[0]
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            return []

        # Postprocess results
        detections = self._postprocess(outputs, ratio, dwdh, original_shape)

        return detections

    @staticmethod
    def draw_detections(image: Image.Image, detections: List[Dict], line_thickness: int = 2) -> Image.Image:
        """
        Draws bounding boxes and labels on the image.

        Args:
            image (Image.Image): The image on which to draw.
            detections (List[Dict]): The list of detections from the detect method.
            line_thickness (int): Thickness of the bounding box lines.

        Returns:
            Image.Image: The image with detections drawn on it.
        """
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for det in detections:
            box = det['box']
            score = det['score']
            class_name = det['class_name']
            x1, y1, x2, y2 = box

            # Assign color based on class (green for body, blue for head)
            color = (0, 255, 0) if class_name == 'body' else (0, 0, 255)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)

            # Prepare label text
            label = f"{class_name}: {score:.2f}"
            
            # Calculate text position
            label_y = y1 - 10 if y1 - 10 > 0 else y1 + 10
            
            # Draw filled rectangle behind text for better visibility
            text_bbox = draw.textbbox((x1, label_y), label, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=color)
            
            # Draw text
            draw.text((x1, label_y), label, fill=(255, 255, 255), font=font)

        return img_draw

    @staticmethod
    def resize_for_display(image: Image.Image, max_height: int = 1080, max_width: int = 1920) -> Image.Image:
        """Resizes an image for display if it exceeds maximum dimensions."""
        w, h = image.size
        if h > max_height or w > max_width:
            ratio = min(max_height / h, max_width / w)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
        return image


if __name__ == '__main__':
    import os
    import sys

    parser = PedestrianHeadDetector.add_arguments()
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--show-images", action='store_true',
                        help="Display images with detections (if available)")
    args = parser.parse_args()
    
    # Initialize Detector
    try:
        detector = PedestrianHeadDetector(
            **args.__dict__
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)

    # Process Images in Directory
    IMAGE_DIR = args.image_dir
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(image_files)
    print(f"Found {total_files} images in {IMAGE_DIR}")

    start_time = time.time()

    for i, filename in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, filename)
        print(f"\n--- Processing image {i+1}/{total_files}: {filename} ---")

        # Read Image
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Warning: Could not read image {image_path}. Skipping. Error: {e}")
            continue

        # Perform Detection
        inference_start = time.time()
        detections = detector.detect(img)
        inference_time = time.time() - inference_start

        print(f"Detected {len(detections)} objects in {inference_time:.4f} seconds:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['box']} (Score: {det['score']:.2f})")

        # Draw and Show Results (Optional)
        if args.show_images:
            img_with_detections = detector.draw_detections(img, detections)
            img_display = detector.resize_for_display(img_with_detections, max_height=1080, max_width=1920)
            img_display.show()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_files if total_files > 0 else 0

    print(f"\n--- Finished processing {total_files} images ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds")