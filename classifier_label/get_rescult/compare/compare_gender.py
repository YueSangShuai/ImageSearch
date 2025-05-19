import os
import re

# 定义性别字典
gender_dict = {"female": 0, "male": 1}
get_gender_dict = {"female": 0, "male": 1}

# 自定义异常类
class ExtractionError(Exception):
    pass

is_conbime=False
# 假设这是你的数据文件路径
file_path = '/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/label2/gender_output.txt'
gender_numeric_list = []
labels_numeric_lists = []  # 用于保存每行所有标签的数字表示

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

            # 从文件路径提取性别信息
            file_name = os.path.basename(img_path)

            # 使用正则表达式查找文件名中的性别 (忽略大小写)
            gender_match = re.search(r'_(male|female)_', file_name, re.IGNORECASE)

            if gender_match:
                gender_in_filename = gender_match.group(1).lower()  # 提取匹配的性别部分并转换为小写
            else:
                raise ExtractionError(f"无法从文件名提取性别信息: {file_name}")

            # 根据gender_dict获取数字
            gender_numeric = gender_dict.get(gender_in_filename, None)
            if gender_numeric is None:
                raise ExtractionError(f"无法从文件名提取性别数字: {gender_in_filename}")

            if is_conbime:
            # 提取所有标签（假设它们是第二个字段到最后一个字段）
                txt_labels = [data_parts[i] for i in range(1, len(data_parts))]
            else:
                txt_labels = [data_parts[i] for i in range(1, int(len(data_parts)/2)+1)]
                
                
            # 将标签转换为数字并存入列表
            label_numeric = []
            for label in txt_labels:
                label_numeric_value = get_gender_dict.get(label.lower(), None)
                if label_numeric_value is None:
                    raise ExtractionError(f"无法从标签提取性别数字: {label}")
                label_numeric.append(label_numeric_value)

            gender_numeric_list.append(gender_numeric)
            labels_numeric_lists.append(label_numeric)

        except ExtractionError as e:
            print(f"错误: {e}")

# 计算与每个标签的匹配数量
match_counts = [0] * len(labels_numeric_lists[0])  # 初始化为每个标签的匹配计数
total_samples = len(gender_numeric_list)

for i in range(len(labels_numeric_lists[0])):
    # 对每个标签进行匹配
    match_counts[i] = sum(1 for g, labels in zip(gender_numeric_list, labels_numeric_lists) if g == labels[i])

# 输出每个标签的相似度百分比
for i, match_count in enumerate(match_counts):
    similarity = match_count / total_samples * 100 if total_samples > 0 else 0
    print(f"与 label{i + 1} 相同的数量: {match_count} / {total_samples} ({similarity:.2f}%)")
