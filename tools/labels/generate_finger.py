import os
import json

def generate_split_json(split_name, image_dirs, output_path):
    """
    为单个数据集划分（train/val/test）生成 JSON 文件（不带 split 字段）
    
    Args:
        split_name (str): 数据集划分名称（如 "train", "val", "test"）
        image_dirs (list): 当前划分的图像目录列表
        output_path (str): 输出 JSON 文件路径（如 "train.json"）
    """
    dataset = []  # 存储当前划分的所有样本

    # 遍历当前划分的所有图像目录
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            print(f"⚠️ 跳过无效目录: {img_dir}")
            continue

        # 遍历目录下的所有文件和子目录
        for root, _, files in os.walk(img_dir):
            for file in files:
                # 仅处理图像文件（支持常见格式）
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue

                # 图像完整路径
                img_full_path = os.path.join(root, file)
                
                # 生成对应的 txt 路径（与图像同目录、同名，扩展名改为 .txt）
                txt_full_path = os.path.splitext(img_full_path)[0] + ".txt"
                
                # 检查 txt 文件是否存在（若不存在则创建空文件）
                if not os.path.exists(txt_full_path):
                    with open(txt_full_path, 'w') as f:
                        pass  # 创建空文件（后续可手动添加描述）
                    print(f"⚠️ 对应的 txt 文件不存在，已创建空文件: {txt_full_path}")

                # 根据文件夹名确定标签（0=活体，1=伪造）
                label = -1  # 默认无效标签
                if 'live' in root.lower() or 'Live' in root.lower():
                    label = 0
                elif 'spoof' in root.lower() or 'Fake' in root.lower() or 'fake' in root.lower():
                    label = 1
                else:
                    print(f"⚠️ 无法匹配标签，跳过图像: {img_full_path}")
                    continue

                # 构造样本字典（不带 split 字段）
                sample = {
                    "image_path": img_full_path,  # 图像绝对路径
                    "text_path": txt_full_path,   # 对应的 txt 绝对路径
                    "finger": label                # 标签（0=活体，1=伪造）
                }
                
                dataset.append(sample)

    # 写入 JSON 文件（缩进美化，确保中文正常显示）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 完成：{split_name} 集生成 {len(dataset)} 条样本，保存至 {output_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 定义数据集划分目录（直接使用你提供的路径）
    dataset_config = {
        "train": [
            "/data/yuesang/finger/finger_for_yolo/finger_from_ML",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zam180/Fake/train",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zam180/Live/train",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zam180_676979/train",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zkteco/train",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zkteco/ZLK_V2/train",
            "/data/yuesang/finger/finger_for_yolo/wet_finger",
        ],
        "val": [
            "/data/yuesang/finger/finger_for_yolo/finger_from_zam180/Fake/val",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zam180/Live/val",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zkteco/val",
            "/data/yuesang/finger/finger_for_yolo/finger_from_zkteco/ZLK_V2/val"
        ],
        "test": [
            "/data/yuesang/finger/test_library/test_zam180V2"
        ]
    }

    # 输出根目录（存放所有 JSON 文件）
    output_root = "/data/yuesang/LLM/contrastors/finger_json/"
    os.makedirs(output_root, exist_ok=True)

    # 为每个划分生成独立的 JSON 文件（不带 split 字段）
    for split_name, img_dirs in dataset_config.items():
        output_json = os.path.join(output_root, f"{split_name}.json")  # 文件名：train.json/val.json/test.json
        generate_split_json(
            split_name=split_name,       # 数据集划分名称（仅用于打印日志）
            image_dirs=img_dirs,         # 当前划分的图像目录列表
            output_path=output_json      # 输出 JSON 路径
        )