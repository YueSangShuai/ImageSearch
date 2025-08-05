import os
import glob

def read_labels(label_file_path):
    """
    读取标签文件，返回一个字典，键是ID，值是对应的描述
    """
    id_to_description = {}
    with open(label_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 直接按第一个空格分割（假设ID和描述之间只有1个空格）
            parts = line.split(' ', 1)
            if len(parts) == 2:
                id_, description = parts
                id_to_description[id_] = description
    return id_to_description

def process_images(image_folder, label_file_path):
    """
    处理图片文件夹，为每张图片创建与原文件名相同的TXT描述文件
    """
    # 检查标签文件是否存在
    if not os.path.exists(label_file_path):
        print(f"警告: 标签文件 {label_file_path} 不存在，跳过该文件夹")
        return
    
    # 读取标签
    id_to_description = read_labels(label_file_path)
    
    # 获取所有图片文件（支持.bmp/.jpg/.png）
    image_files = glob.glob(os.path.join(image_folder, '*.bmp')) + \
                  glob.glob(os.path.join(image_folder, '*.jpg')) + \
                  glob.glob(os.path.join(image_folder, '*.png'))+ \
                  glob.glob(os.path.join(image_folder, '*.jpeg'))
    
    # 处理每张图片
    for image_path in image_files:
        filename = os.path.basename(image_path)
        id_ = filename.split('_')[0]  # 假设ID是第一个下划线前的部分
        original_filename = os.path.splitext(filename)[0]  # 去掉扩展名
        
        if id_ in id_to_description:
            description = id_to_description[id_]
            # 生成与原文件同名的TXT文件
            output_file_path = os.path.join(image_folder, f"{original_filename}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(description)
            print(f"已生成: {output_file_path}")
        else:
            print(f"警告: ID {id_} 未在标签文件中找到，跳过图片 {filename}")

if __name__ == "__main__":
    
    # image_folders = [
    #     # "/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/CAVIAR4REID/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/CUHK/archive/",
    #     "/data/yuesang/LLM/contrastors/data/PETA/GRID/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/i-LID/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/MIT/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/PRID/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/SARC3D/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/TownCentre/archive/",
    #     # "/data/yuesang/LLM/contrastors/data/PETA/VIPeR/archive/",
    # ]
    
    image_folder="/data/yuesang/LLM/contrastors/data/pa100k_label/data"
    label_file_path="/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/first/checked_test_lb.txtdescrip.txt"
    process_images(image_folder,label_file_path)
    label_file_path="/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/first/checked_train_lb.txtdescrip.txt"
    process_images(image_folder,label_file_path)
    label_file_path="/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/first/checked_val_lb.txtdescrip.txt"
    process_images(image_folder,label_file_path)

        
    