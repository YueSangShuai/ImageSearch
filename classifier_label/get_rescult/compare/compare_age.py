import os
import re

# 年龄段字典（确保 key 全部是小写）
age_dict = {
    "ageless15": 0,
    "age16-30": 1,
    "age31-45": 2,
    "age46-60": 3,
    "ageabove60": 4
}

get_age_dict = {
    "0~12岁的人": 0,
    "13~19岁的人": 1,
    "20~39岁的人": 2,
    "40~60岁的人": 3,
    "60岁以上的人": 4
}

# 自定义异常类
class ExtractionError(Exception):
    pass

is_conbime=False
# 假设数据文件路径
file_path = '/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/label2/age_output.txt'

# 存储数值列表
age_numeric_list = []
labels_numeric_lists = []  # 用于存储每行所有标签的数字表示

# 读取文件并逐行处理
with open(file_path, 'r') as file:
    for line in file:
        # 去除行末的换行符
        line = line.strip()

        # 分割数据行
        data_parts = line.split()

        try:
            # 提取图片路径
            img_path = data_parts[0]

            # 从文件名中提取年龄信息
            file_name = os.path.basename(img_path)
            # 使用正则表达式匹配年龄信息（忽略大小写）
            age_match = re.search(r'_(ageless15|age16-30|age31-45|age46-60|ageabove60)', file_name, re.IGNORECASE)
            if age_match:
                age_in_filename = age_match.group(1).lower()  # 统一转换小写
            else:
                raise ExtractionError(f"无法从文件名提取年龄信息: {file_name}")


            # 获取年龄对应的数值
            age_numeric = age_dict.get(age_in_filename, None)
            if age_numeric is None:
                raise ExtractionError(f"无法从文件名解析年龄数值: {age_in_filename}")

            # 提取所有标签（假设它们是第二个字段到最后一个字段）
            if is_conbime:
            # 提取所有标签（假设它们是第二个字段到最后一个字段）
                txt_labels = [data_parts[i] for i in range(1, len(data_parts))]
            else:
                txt_labels = [data_parts[i] for i in range(1, int(len(data_parts)/2)+1)]
                
            
            # 将标签转换为数字并存入列表
            label_numeric = []
            for label in txt_labels:
                label_numeric_value = get_age_dict.get(label, None)
                if label_numeric_value is None:
                    raise ExtractionError(f"无法从标签解析年龄数值: {label}")
                label_numeric.append(label_numeric_value)

            age_numeric_list.append(age_numeric)
            labels_numeric_lists.append(label_numeric)

        except ExtractionError as e:
            print(f"错误: {e}")

# 计算与每个标签的匹配数量
match_counts = [0] * len(labels_numeric_lists[0])  # 初始化为每个标签的匹配计数
total_samples = len(age_numeric_list)

for i in range(len(labels_numeric_lists[0])):
    # 对每个标签进行匹配
    match_counts[i] = sum(1 for g, labels in zip(age_numeric_list, labels_numeric_lists) if g == labels[i])

# 输出每个标签的相似度百分比
for i, match_count in enumerate(match_counts):
    similarity = match_count / total_samples * 100 if total_samples > 0 else 0
    print(f"与 label{i + 1} 相同的数量: {match_count} / {total_samples} ({similarity:.2f}%)")
