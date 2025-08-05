import os
import json

# 标签顺序表
attribute_labels = [
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
    'lwhite', 'lpink', 'lred', 'lgreen','lyellow', 'lpurple', 'lbrown', 'lblack', 'lorange', 'lblue', 'lgray', 'lbeige', 'lstriped_color', 'lmulticolor'
]

def generate_json_from_directory_with_labels(txt_path,root_dir, output_json_path):
    items = []
    
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split()
            img_name = parts[0].replace(".jpg", "")
            img_path=os.path.join(root_dir, img_name+".jpg")
            txt_path = os.path.join(root_dir, img_name+".txt")

            label_values = list(map(int, parts[1:]))  # ['0', '1', ...] => [0, 1, ...]
            
            if len(label_values) != len(attribute_labels):
                print(f"[标签数量不匹配] {txt_path} 有 {len(label_values)} 个标签，期望 {len(attribute_labels)} 个")
                continue


            item = {
                "image_path": img_path,
                "text_path": txt_path,
            }
            
            for attribute_label,label_value in zip(attribute_labels, label_values):
                item[attribute_label]=label_value
            
            items.append(item)
            
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成 JSON：{output_json_path}，共 {len(items)} 条记录")
    

# 示例用法
if __name__ == "__main__":
    input_dir = "/data/yuesang/LLM/contrastors/data/pa100k_label/label/check/checked_test_lb.txt"  # 替换为你的路径
    root_dir="/data/yuesang/LLM/contrastors/data/pa100k_label/data"
    output_json = "/data/yuesang/LLM/contrastors/data/pa100k_label/test.json"
    generate_json_from_directory_with_labels(input_dir,root_dir, output_json)
