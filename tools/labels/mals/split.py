import os
import json
import random

def split_and_merge_json(input_dir, output_dir, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """
    按比例划分JSON文件并合并为单层列表（无嵌套）
    """
    # 1. 检查输入目录
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在：{input_dir}")
    
    # 2. 获取所有JSON文件路径
    json_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.json') and os.path.isfile(os.path.join(input_dir, f))
    ]
    if not json_files:
        raise ValueError(f"输入目录中没有找到JSON文件：{input_dir}")
    total = len(json_files)
    print(f"找到 {total} 个JSON文件，准备处理...")
    
    # 3. 打乱文件顺序
    random.shuffle(json_files)
    
    # 4. 按比例划分文件列表
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    train_files = json_files[:train_size]
    val_files = json_files[train_size : train_size + val_size]
    test_files = json_files[train_size + val_size :]
    
    # 5. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. 合并函数（核心修复：处理嵌套列表，确保最终为单层）
    def merge_json_files(file_list, save_path):
        """合并文件内容，确保输出为单层列表（无嵌套）"""
        merged_data = []
        for file in file_list:
            file_path = os.path.join(input_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 关键修复：如果读取到的是列表，则将其元素逐个添加（解嵌套）
                    if isinstance(data, list):
                        merged_data.extend(data)  # 展开列表，避免嵌套
                    else:
                        merged_data.append(data)  # 如果是单个对象，直接添加
                        
            except Exception as e:
                print(f"警告：读取 {file} 失败，已跳过 - {str(e)}")
        
        # 保存合并结果（确保是单层列表）
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"已合并 {len(merged_data)} 条数据到 {save_path}（单层列表）")
        return merged_data
    
    # 7. 合并并保存三个集合
    train_save_path = os.path.join(output_dir, "train.json")
    val_save_path = os.path.join(output_dir, "val.json")
    test_save_path = os.path.join(output_dir, "test.json")
    
    train_data = merge_json_files(train_files, train_save_path)
    val_data = merge_json_files(val_files, val_save_path)
    test_data = merge_json_files(test_files, test_save_path)
    
    # 8. 打印统计
    print("\n最终统计：")
    print(f"总有效数据量：{len(train_data) + len(val_data) + len(test_data)}")
    print(f"训练集：{len(train_data)} 条")
    print(f"验证集：{len(val_data)} 条")
    print(f"测试集：{len(test_data)} 条")


if __name__ == "__main__":
    input_directory = "/data/yuesang/LLM/contrastors/data/MALS/json"
    output_directory = "/data/yuesang/LLM/contrastors/data/MALS/merged_json"
    
    try:
        split_and_merge_json(input_directory, output_directory)
        print("\n所有操作完成！输出为单层列表JSON")
    except Exception as e:
        print(f"处理失败：{str(e)}")