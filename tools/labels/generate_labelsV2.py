import os
import json

# 年龄和性别的类别映射（根据你的实际需求调整）
age_classes_dict = {
    "ageless15": 0,
    "age16-30": 1,
    "age31-45": 2,
    "age46-60": 3,
    "ageabove60": 4
}
gender_classes_dict = {
    "female": 0,
    "male": 1
}

def get_age_label(path):
    """从文件名中提取年龄标签（返回数值）"""
    for k, v in age_classes_dict.items():
        if k in path.lower():  # 不区分大小写匹配关键词（如"age16-30"在路径中出现）
            return v
    return -1  # 未匹配到

def get_gender_label(path):
    """从文件名中提取性别标签（返回数值）"""
    for k, v in gender_classes_dict.items():
        if k in path.lower():  # 不区分大小写匹配关键词（如"female"在路径中出现）
            return v
    return -1  # 未匹配到

def collect_images_to_json(folder_list, output_json_path, image_extensions=None, recursive=True):
    """
    收集图像路径及其对应的txt路径、年龄、性别标签，保存为JSON格式
    参数：
        folder_list: 图像文件夹列表（每个文件夹对应一个数据集划分，如train/val）
        output_json_path: 输出JSON文件路径（如"dataset.json"）
        image_extensions: 允许的图像扩展名（默认：{.jpg, .jpeg, .png, .bmp}）
        recursive: 是否递归遍历子目录（默认：True）
    返回：
        None（直接生成JSON文件）
    """
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    dataset = []  # 存储所有样本的列表（最终转为JSON）

    for folder in folder_list:
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            print(f"⚠️ 跳过无效目录: {folder}")
            continue

        # 遍历文件（递归或非递归）
        file_iter = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
        for dirpath, _, filenames in file_iter:
            for fname in filenames:
                # 检查是否为图像文件
                ext = os.path.splitext(fname)[1].lower()
                if ext not in image_extensions:
                    continue
                
                # 图像完整路径（绝对路径）
                img_full_path = os.path.join(dirpath, fname)
                
                # 生成对应的txt路径（与图像同目录、同名，扩展名改为.txt）
                txt_full_path = os.path.splitext(img_full_path)[0] + ".txt"
                
                # 检查txt文件是否存在（可选，但建议添加）
                if not os.path.exists(txt_full_path):
                    print(f"⚠️ 对应的txt文件不存在，跳过图像: {img_full_path}")
                    continue
                
                # 提取年龄和性别标签
                age_label = get_age_label(img_full_path)
                gender_label = get_gender_label(img_full_path)
                
                # 跳过标签缺失的样本
                if age_label == -1 or gender_label == -1:
                    print(f"⚠️ 无法匹配标签，跳过文件: {img_full_path}")
                    continue
                
                # 构造样本字典（包含图像路径、txt路径、年龄标签、性别标签）
                sample = {
                    "image_path": img_full_path,  # 图像绝对路径
                    "text_path": txt_full_path,   # 对应的txt绝对路径
                    "age": age_label,       # 年龄标签（数值）
                    "gender": gender_label  # 性别标签（数值）
                }
                
                dataset.append(sample)

    # 写入JSON文件（确保中文正常显示，缩进美化）
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 完成：共收集 {len(dataset)} 条有效样本，保存至 {output_json_path}")

# ==================== 示例使用 ====================
if __name__ == "__main__":
    # 训练集文件夹（假设每个文件夹是独立划分，如train/val/test）
    train_folders = [
        "/data/yuesang/person_attribute/19w"  # 替换为你的训练集路径
    ]
    # 验证集文件夹（可选多个）
    val_folders = [
        "/data/yuesang/LLM/contrastors/data/jf_test_2024_1_676",
        "/data/yuesang/LLM/contrastors/data/4949"
    ]

    # 生成训练集JSON（包含图像、txt路径及标签）
    collect_images_to_json(
        folder_list=train_folders,
        output_json_path="train.json",
        image_extensions={'.jpg', '.jpeg', '.png'},  # 自定义允许的扩展名
        recursive=True  # 递归遍历子目录
    )

    # 生成验证集JSON
    collect_images_to_json(
        folder_list=val_folders,
        output_json_path="val.json",
        image_extensions={'.jpg', '.jpeg', '.png'},
        recursive=True
    )