import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import cv2
import argparse, os

class PedestrianDetector:
    def __init__(self, onnx_path, threshold=0.5):
        """
        初始化行人检测器
        :param onnx_path: ONNX模型文件路径
        :param threshold: 检测阈值（默认0.5）
        """
        self.session = ort.InferenceSession(onnx_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("行人检测器 providers:", self.session._providers)
        self.threshold = threshold
        self.input_size = 640  # 模型输入尺寸
        self.label_name = {0: "person"}  # 类别标签映射
    
    @staticmethod
    def _resize_with_aspect_ratio(image, size):
        """
        保持宽高比的图像缩放方法
        :return: 缩放后的图像，缩放比例，宽高偏移量
        """
        original_width, original_height = image.size
        ratio = min(size / original_width, size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        
        new_image = Image.new("RGB", (size, size))
        new_image.paste(resized_image, ((size - new_width) // 2, (size - new_height) // 2))
        return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2
    
    def _preprocess(self, image_pil):
        """预处理函数，返回缩放后的张量和尺寸信息"""
        resized_img, ratio, pad_w, pad_h = self._resize_with_aspect_ratio(image_pil, self.input_size)
        transform = T.Compose([T.ToTensor()])
        return transform(resized_img).unsqueeze(0), ratio, (pad_w, pad_h)
    
    def _postprocess(self, outputs, ratio, padding, threshold):
        """后处理函数，筛选行人检测结果并转换坐标"""
        labels, boxes, scores = outputs
        mask = (labels == 0) & (scores > threshold)
        
        valid_boxes = boxes[mask]
        valid_scores = scores[mask]
        
        # 将坐标转换回原始图像尺寸
        pad_w, pad_h = padding
        converted_boxes = []
        for box in valid_boxes:
            converted_boxes.append([
                max(0, int((box[0] - pad_w) / ratio)),  # xmin
                max(0, int((box[1] - pad_h) / ratio)),  # ymin
                int((box[2] - pad_w) / ratio),          # xmax
                int((box[3] - pad_h) / ratio)           # ymax
            ])
        return converted_boxes, valid_scores
    
    def detect_image(self, image_path=None, image_pil=None, threshold=None, output_path="result.jpg"):
        """
        执行图像行人检测
        :param image_path: 输入图像路径
        :param output_path: 输出图像路径
        :return: 检测结果列表（每个元素为(xmin, ymin, xmax, ymax, score)）
        """
        if image_path is not None and image_pil is None:
            image_pil = Image.open(image_path).convert("RGB")
        threshold = threshold if threshold is not None else self.threshold
        original_size = image_pil.size
        input_tensor, ratio, padding = self._preprocess(image_pil)
        orig_size_tensor = torch.tensor([[self.input_size, self.input_size]])

        # 执行推理
        outputs = self.session.run(
            None,
            {
                "images": input_tensor.numpy(),
                "orig_target_sizes": orig_size_tensor.numpy()
            }
        )
        
        # 处理后处理
        boxes, scores = self._postprocess(outputs, ratio, padding, threshold)
        
        # 绘制并保存结果
        if output_path:
            self._draw_boxes(image_pil, boxes, scores)
            image_pil.save(output_path)
        
        return [(box[0], box[1], box[2], box[3], score) \
            for box, score in zip(boxes, scores)]
    
    def process_video(self, video_path, threshold=None, output_path="result.mp4"):
        """
        执行视频行人检测
        :param video_path: 输入视频路径
        :param output_path: 输出视频路径
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        threshold = threshold if threshold is not None else self.threshold
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换格式并检测
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor, ratio, padding = self._preprocess(frame_pil)
            orig_size_tensor = torch.tensor([[self.input_size, self.input_size]])

            outputs = self.session.run(
                None,
                {
                    "images": input_tensor.numpy(),
                    "orig_target_sizes": orig_size_tensor.numpy()
                }
            )

            boxes, scores = self._postprocess(outputs, ratio, padding, threshold)
            
            # 绘制框并写入视频文件
            if out is not None:
                self._draw_boxes(frame_pil, boxes, scores)
                video_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(video_frame)
        
        cap.release()
        if out is not None:
            out.release()
    
    def _draw_boxes(self, image, boxes, scores):
        """在图像上绘制行人检测框"""
        draw = ImageDraw.Draw(image)
        for box, score in zip(boxes, scores):
            draw.rectangle(box, outline="red", width=2)
            text = f"{self.label_name[0]}: {score:.2f}"
            draw.text((box[0]+5, box[1]+5), text, fill="red")

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using ONNX Runtime')
    parser.add_argument('--model', type=str, help='Path to ONNX model file', default="det_coco_s.onnx")
    parser.add_argument('--input', type=str, required=True, help='Input file path (image/video)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # 校验输入文件是否存在
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # 自动确定输出路径
    if not args.output:
        if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
            args.output = 'result.jpg'
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            args.output = 'result.mp4'
        else:
            print("Error: Could not determine output format from input file extension")
            return
    
    # 初始化检测器
    detector = PedestrianDetector(args.model, args.threshold)
    
    # 根据文件类型处理
    try:
        if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {args.input}")
            detections = detector.detect_image(args.input, threshold=args.threshold, output_path=args.output)
            print(f"Saved result to {args.output}")
            print(f"Detected {len(detections)} persons")
            
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: {args.input}")
            detector.process_video(args.input, threshold=args.threshold, output_path=args.output)
            print(f"Saved result to {args.output}")
            
        else:
            print("Error: Unsupported file format. Supported formats: images(png/jpg/jpeg), videos(mp4/avi/mov)")
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()