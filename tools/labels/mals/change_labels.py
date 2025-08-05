import os
import json

# PA-100k标签列表（目标格式）
PA_100k_attribute_labels = [
    'Female',
    'AgeOver60', 'Age18-60', 'AgeLess18',  # 年龄标签（索引1-3）
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

# MALS标签列表（源格式）
attr_names = [
    "Female",        # 0
    'age',           # 1（年龄段标签，重点处理）
    "hair_length",   # 2
    "wearing_hat",   # 3
    "carrying_backpack,handbag",  #4
    "not_wearing_hat",  #5
    "not_carrying_backpack,handbag",  #6
    "sleeve_length",  #7
    "lower_body_length",  #8
    "lower_body_type",  #9
    # 上装颜色（10-17）
    "ublack", "uwhite", "ured", "upurple", "uyellow", "ublue", "ugreen", "ugray",
    # 下装颜色（18-26）
    "lblack", "lwhite", "lpurple", "lyellow", "lblue", "lgreen", "lpink", "lgray", "lbrown"
]


def mals_to_pa_full_format(mals_labels):
    """
    MALS转PA-100k格式（重点处理age标签）
    规则：
    - 若mals的age=0 → AgeLess18=1，Age18-60=0，AgeOver60=0
    - 若mals的age=1 → AgeLess18=0，Age18-60=1，AgeOver60=1
    """
    # 1. 初始化PA标签值（默认-1）
    pa_labels = [-1 for _ in PA_100k_attribute_labels]
    # 2. 建立PA标签索引映射（快速定位标签位置）
    pa_label_index = {label: idx for idx, label in enumerate(PA_100k_attribute_labels)}

    # 3. 处理通用标签（除age外的其他标签）
    mals_to_pa = {
        "Female": ["Female"],
        "wearing_hat": ["Hat"],
        "carrying_backpack,handbag": ["Backpack", "HandBag"],
        "sleeve_length": ["LongSleeve", "ShortSleeve"],
        "lower_body_type": ["Trousers", "Skirt&Dress"],
        # 上装颜色
        "ublack": ["ublack"],
        "uwhite": ["uwhite"],
        "ured": ["ured"],
        "upurple": ["upurple"],
        "uyellow": ["uyellow"],
        "ublue": ["ublue"],
        "ugreen": ["ugreen"],
        "ugray": ["ugray"],
        # 下装颜色
        "lblack": ["lblack"],
        "lwhite": ["lwhite"],
        "lpurple": ["lpurple"],
        "lyellow": ["lyellow"],
        "lblue": ["lblue"],
        "lgreen": ["lgreen"],
        "lpink": ["lpink"],
        "lgray": ["lgray"],
        "lbrown": ["lbrown"],
        "hair_length": ["hair_length"]
    }

    # 先处理非age的通用标签
    for attr_idx, (attr_name, label_value) in enumerate(zip(attr_names, mals_labels)):
        # 跳过age（单独处理）和无映射的标签
        if attr_name == "age" or attr_name not in mals_to_pa:
            continue
        # 赋值通用标签
        for pa_label in mals_to_pa[attr_name]:
            if pa_label in pa_label_index:
                pa_idx = pa_label_index[pa_label]
                pa_labels[pa_idx] = label_value

    # 4. 单独处理age标签（核心逻辑）
    # 获取MALS中age的标签值（先确定age在mals_labels中的索引）
    age_attr_idx = attr_names.index("age")  # age在attr_names中的索引（固定为1）
    age_value = mals_labels[age_attr_idx] if age_attr_idx < len(mals_labels) else -1

    # 根据age值设置PA的三个年龄标签
    if age_value == 0:
        # age=0 → AgeLess18=1，其他年龄标签=0
        pa_labels[pa_label_index["AgeLess18"]] = 1
        pa_labels[pa_label_index["Age18-60"]] = 0
        pa_labels[pa_label_index["AgeOver60"]] = 0
    elif age_value == 1:
        # age=1 → AgeLess18=0，其他年龄标签=1
        pa_labels[pa_label_index["AgeLess18"]] = 0
        pa_labels[pa_label_index["Age18-60"]] = 1
        pa_labels[pa_label_index["AgeOver60"]] = 1
    # 若age不是0/1（如-1），则保持默认-1

    return pa_labels


def get_filelist(dir):
    filelist = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.json'}
    for home, dirs, files in os.walk(dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in image_extensions:
                filelist.append(os.path.join(home, filename))
    return filelist


def get_json_info(json_dir):
    if not os.path.isdir(json_dir):
        print(f"错误：文件夹不存在 - {json_dir}")
        return []
    json_info = []
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            json_info.append(os.path.join(json_dir, filename))
    return json_info


def save_results_to_json(json_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到：{output_dir}")
    
    json_data = get_json_info(json_dir)
    total_files = len(json_data)
    processed_count = 0
    
    for file_idx, file_path in enumerate(json_data, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            
            result_list = []
            for json_info in json_content:
                # 基础路径信息
                item = {
                    "image_path": os.path.join(image_dir, json_info["image"]),
                    "text_path": os.path.join(image_dir, os.path.splitext(json_info["image"])[0] + ".txt")
                }
                
                # 写入caption
                caption = json_info["caption"]
                with open(item["text_path"], 'w', encoding='utf-8') as txt_f:
                    txt_f.write(caption)
                
                # 处理标签：先反转（0↔1），再转换为PA格式
                original_labels = json_info["label"]
                reversed_labels = []
                for label in original_labels:
                    if label == 0:
                        reversed_labels.append(1)
                    elif label == 1:
                        reversed_labels.append(0)
                    else:
                        reversed_labels.append(-1)
                
                # 转换为PA格式（含age特殊处理）
                pa_format_labels = mals_to_pa_full_format(reversed_labels)
                
                # 添加PA标签到item
                for pa_label, value in zip(PA_100k_attribute_labels, pa_format_labels):
                    item[pa_label] = value
                
                result_list.append(item)
                processed_count += 1
            
            # 保存结果
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(result_list, out_f, ensure_ascii=False, indent=2)
            print(f"处理进度：{file_idx}/{total_files} - {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 出错：{str(e)}")
    
    print(f"\n处理完成！共生成 {processed_count} 条数据")


if __name__ == "__main__":
    json_directory = "/data/yuesang/person_attribute/MALS/gene_attrs"
    image_dir = "/data/yuesang/person_attribute/MALS/"
    output_directory = "/data/yuesang/LLM/contrastors/data/MALS/json"
    save_results_to_json(json_directory, image_dir, output_directory)