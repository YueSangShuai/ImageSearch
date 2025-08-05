import os
import json

PA_100k_attribute_labels = [
    'Female',
    'AgeOver60', 'Age18-60', 'AgeLess18',
    'Front', 'Side', 'Back',
    'Hat', 'Glasses',
    'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve', 'UpperStride',
    'UpperLogo', 'UpperPlaid', 'UpperSplice', 
    'LowerStripe', 'LowerPattern',
    'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress',
    'boots',
    'ublack', 'ugray', 'ublue', 'ugreen','uwhite', 'upurple', 'ured', 'ubrown', 'uyellow', 'upink', 'uorange', 'ubeige', 'ustriped_color', 'umulticolor',
    'lwhite', 'lpink', 'lred', 'lgreen','lyellow', 'lpurple', 'lbrown', 'lblack', 'lorange', 'lblue', 'lgray', 'lbeige', 'lstriped_color', 'lmulticolor',
    "hair_length"
]


attr_names = [
    # 人物基础特征
    "Female",                  # 性别
    'age',
    "hair_length",             # 头发长度（短/长）
    
    # 配饰携带
    "wearing_hat",             # 是否戴帽子
    "carrying_backpack,handbag",       # 是否携带背包
    "not_wearing_hat",   # 是否不携带背包
    "not_carrying_backpack,handbag",    # 是否不携带手提包

    # 衣着特征
    "sleeve_length",           # 袖长（长/短）
    "lower_body_length",       # 下装长度（长/短）
    "lower_body_type",         # 下装类型（裙子/裤子）
    
    # 上装颜色（正向）
    "ublack",        # 上装为黑色
    "uwhite",        # 上装为白色
    "ured",          # 上装为红色
    "upurple",       # 上装为紫色
    "uyellow",       # 上装为黄色
    "ublue",         # 上装为蓝色
    "ugreen",        # 上装为绿色
    "ugray",         # 上装为灰色
    
    # 下装颜色（正向）
    "lblack",        # 下装为黑色
    "lwhite",        # 下装为白色
    "lpurple",       # 下装为紫色
    "lyellow",       # 下装为黄色
    "lblue",         # 下装为蓝色
    "lgreen",        # 下装为绿色
    "lpink",         # 下装为粉色
    "lgray",         # 下装为灰色
    "lbrown"         # 下装为棕色
]


def get_filelist(dir):
    """获取文件夹下所有图片路径（保持原结构）"""
    filelist = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.json'}
    for home, dirs, files in os.walk(dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in image_extensions:
                filelist.append(os.path.join(home, filename))
    return filelist


def get_json_info(json_dir):
    """获取指定文件夹中所有JSON文件的完整路径"""
    if not os.path.isdir(json_dir):
        print(f"错误：文件夹不存在 - {json_dir}")
        return []
    
    json_info = []
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            original_path = os.path.join(json_dir, filename)
            json_info.append(original_path)
    return json_info


def save_results_to_json(json_dir, image_dir, output_dir):
    """
    处理数据并将结果保存到指定路径的多个JSON文件
    
    :param json_dir: JSON文件所在文件夹路径
    :param image_dir: 图片所在根目录
    :param output_dir: 输出JSON文件的指定路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到：{output_dir}")
    
    # 获取JSON文件列表
    json_data = get_json_info(json_dir)
    total_files = len(json_data)
    processed_count = 0
    
    for file_idx, file_path in enumerate(json_data, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            
            rescult2json=[]
            for json_info in json_content:
                # 构建结果字典
                item = {
                    "image_path": os.path.join(image_dir, json_info["image"]),
                    "text_path": os.path.join(image_dir, os.path.splitext(json_info["image"])[0] + ".txt")
                }
                
                # 写入caption到txt文件
                caption = json_info["caption"]
                with open(item["text_path"], 'w', encoding='utf-8') as txt_f:
                    txt_f.write(caption)
                
                # 添加属性标签
                labels = json_info["label"]
                for i, label in enumerate(labels):
                    if i < len(attr_names):  # 防止索引越界
                        if label==0:
                            label=1
                        elif label==1:
                            label=0
                        item[attr_names[i]] = label
                    else:
                        print(f"警告：标签索引 {i} 超出属性列表长度，已忽略")
                
                # 生成输出JSON文件名
                image_basename = os.path.splitext(os.path.basename(json_info["image"]))[0]
                output_filename = f"{image_basename}_result.json"
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                
                rescult2json.append(item)

                
                
                processed_count += 1
            # 保存为独立JSON文件
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(rescult2json, out_f, ensure_ascii=False, indent=2)
            print(f"处理进度：{file_idx}/{total_files} - 已处理文件：{os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错：{str(e)}")
    
    print(f"\n处理完成！共生成 {processed_count} 个JSON文件")


if __name__ == "__main__":
    # 配置路径
    json_directory = "/data/yuesang/person_attribute/MALS/gene_attrs"
    image_dir = "/data/yuesang/person_attribute/MALS/"
    output_directory = "/data/yuesang/LLM/contrastors/data/MALS/json"  # 指定输出路径
    
    # 执行保存操作
    save_results_to_json(json_directory, image_dir, output_directory)
