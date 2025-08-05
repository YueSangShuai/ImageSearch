import re

# 原始路径
input_path = "/data/yuesang/LLM/contrastors/src/contrastors/configs/attributes/peta_attributes.yaml"
output_path = "/data/yuesang/LLM/contrastors/src/contrastors/configs/attributes/swap.yaml"

# 读取原始内容
with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# 定义正则替换函数：调换 no 和 yes 的顺序
def swap_yes_no_block(match):
    yes_line = match.group("yes")
    no_line = match.group("no")
    return f"text_en: {{\n        no: {no_line},\n        yes: {yes_line}\n    }}"

# 正则匹配 text_en 中 yes/no 内容
pattern = re.compile(
    r'text_en: {\s*yes: (?P<yes>".*?"),\s*no: (?P<no>".*?")\s*}', 
    re.DOTALL
)

# 替换
swapped_content = pattern.sub(swap_yes_no_block, content)

# 写入新文件
with open(output_path, "w", encoding="utf-8") as f:
    f.write(swapped_content)

print(f"输出路径：{output_path}")
