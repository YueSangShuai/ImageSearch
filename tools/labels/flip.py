import os
import shutil
from pathlib import Path
from tqdm import tqdm
# 原始图片根目录
src_dir = Path("/data/images/person_attribute/19w")
# 平铺后的输出目录
dst_dir = Path("/data/yuesang/person_attribute/train")

dst_dir.mkdir(parents=True, exist_ok=True)

# 支持的图片后缀
image_suffix = [".jpg", ".jpeg", ".png", ".bmp"]

count = 0

for filepath in tqdm(src_dir.rglob("*")):
    if filepath.suffix.lower() in image_suffix:
        # 获取相对于src_dir的路径，比如 "subfolder1/xxx.jpg"
        relative_path = filepath.relative_to(src_dir)
        # 拿到父文件夹名
        folder_name = relative_path.parent.name
        # 新文件名 = 父文件夹名_原文件名
        new_filename = f"{folder_name}_{filepath.name}"
        dst_file = dst_dir / new_filename
        
        shutil.copy(filepath, dst_file)
        count += 1

print(f"✔️ Done! Copied {count} images into {dst_dir}")
