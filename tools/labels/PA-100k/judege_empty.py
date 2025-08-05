import os

def find_empty_txt_files(dir_path):
    empty_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                # 判断是否为空
                if os.path.getsize(full_path) == 0:
                    empty_files.append(full_path)
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            empty_files.append(full_path)
    return empty_files

# 示例用法
directory = "/data/yuesang/LLM/contrastors/data/pa100k_label/data"  # 修改为你的目录
empty_txts = find_empty_txt_files(directory)

if empty_txts:
    print(f"发现 {len(empty_txts)} 个空的txt文件:")
    for f in empty_txts:
        print(f)
else:
    print("没有空的txt文件。")
