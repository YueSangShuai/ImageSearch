import os
import json
from typing import List, Dict


def load_json(file_path: str) -> List[Dict]:
    """加载单个JSON文件，返回列表（若文件不存在或格式错误则返回空列表）"""
    if not os.path.exists(file_path):
        print(f"警告：文件不存在 - {file_path}（已跳过）")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 确保加载的数据是列表（JSON文件需为列表格式）
        if isinstance(data, list):
            return data
        else:
            print(f"警告：{file_path} 内容不是列表格式（已跳过）")
            return []
    except json.JSONDecodeError as e:
        print(f"警告：{file_path} 格式错误 - {str(e)}（已跳过）")
        return []


def merge_json_files(folder1: str, folder2: str, output_folder: str) -> None:
    """
    合并两个文件夹下的 train.json、val.json、test.json
    输出：每个类型的合并结果（单列表形式）
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 需要合并的文件名（train/val/test）
    json_types = ["train.json", "val.json", "test.json"]
    
    for json_type in json_types:
        # 1. 获取两个文件夹中对应JSON的路径
        path1 = os.path.join(folder1, json_type)
        path2 = os.path.join(folder2, json_type)
        
        # 2. 加载两个文件的数据（均为列表）
        data1 = load_json(path1)
        data2 = load_json(path2)
        
        # 3. 合并为单列表（直接拼接两个列表）
        merged_data = data1 + data2  # 单列表形式：[item1, item2, ...]（来自folder1和folder2）
        
        # 4. 保存合并结果
        output_path = os.path.join(output_folder, json_type)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"已合并 {json_type} | 总条数：{len(merged_data)}（来自 {len(data1)} + {len(data2)}）")
        print(f"合并结果保存至：{output_path}\n")


if __name__ == "__main__":
    # 配置文件夹路径（替换为你的两个文件夹路径）
    folder_a = "/data/yuesang/LLM/contrastors/data/MALS/merged_json"    # 第一个文件夹（含train.json等）
    folder_b = "/data/yuesang/LLM/contrastors/data/pa100k_label"   # 第二个文件夹（含train.json等）
    output_folder = "/data/yuesang/LLM/contrastors/data/MALS/add"     # 合并结果保存路径
    
    # 执行合并
    merge_json_files(folder_a, folder_b, output_folder)