import json
from collections import Counter

def extract_all_attributes(json_file_path):
    counter = Counter()
    with open(json_file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # 直接读取整个 JSON 列表

    for item in data_list:
        attr_strs = item.get("attributes", [])
        for attr_str in attr_strs:
            attrs = [a.strip().lower() for a in attr_str.split(",")]
            counter.update(attrs)

    return counter

# 使用方法
if __name__ == "__main__":
    json_file = "/data/lina/datasets/SYNTH-PEDES/generated_captions.json"  # 输入文件路径
    output_json = "attribute_counts.json"  # 输出文件路径

    attr_counter = extract_all_attributes(json_file)

    # 保存到 JSON 文件（按出现次数排序）
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dict(attr_counter.most_common()), f, indent=4, ensure_ascii=False)

    print(f"属性频次统计结果已保存到: {output_json}")
