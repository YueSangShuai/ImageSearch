import os
import re
from tqdm import tqdm

# 年龄和性别字典
age_dict = {
    "ageless15": 0,
    "age16-30": 1,
    "age31-45": 2,
    "age46-60": 3,
    "ageabove60": 4
}
gender_dict = {
    "female": 0,
    "male": 1
}

# 英文描述文本
age_text_en = {
    "ageless15": "under 15 years old",
    "age16-30": "between 16 and 30 years old",
    "age31-45": "between 31 and 45 years old",
    "age46-60": "between 46 and 60 years old",
    "ageabove60": "above 60 years old"
}
gender_text_en = {
    "female": "female",
    "male": "male"
}

# 支持的图片后缀
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

# 你的总图像目录
root_folder = "/data/yuesang/person_attribute/19w"  # 顶层目录

# 遍历所有子文件夹和文件
for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if ext.lower() not in image_extensions:
            continue  # 跳过非图像文件

        # 正则提取性别和年龄信息（性别在前，年龄在后）
        match = re.search(r"(female|male).*(ageless15|age16-30|age31-45|age46-60|ageabove60)", name, flags=re.IGNORECASE)
        if not match:
            print(f"跳过无法识别的文件名: {filename}")
            continue

        gender_label, age_label = match.groups()
        gender_label = gender_label.lower()
        age_label = age_label.lower()

        # 构建描述文本
        text = f"This is a {gender_text_en[gender_label]} {age_text_en[age_label]}."

        # 保存到同名txt文件
        txt_path = os.path.join(dirpath, name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
