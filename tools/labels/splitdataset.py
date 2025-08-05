import os
import random

def split_dataset(image_dir, train_txt, val_txt, val_ratio=0, seed=42):
    # 获取所有图像路径
    all_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    # 打乱
    random.seed(seed)
    random.shuffle(all_images)

    # 划分
    val_size = int(len(all_images) * val_ratio)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]

    # 写入 txt 文件
    with open(train_txt, 'w') as f:
        for path in train_images:
            f.write(path + '\n')

    with open(val_txt, 'w') as f:
        for path in val_images:
            f.write(path + '\n')

    print(f"共 {len(all_images)} 张图片：训练集 {len(train_images)}，验证集 {len(val_images)}")
    print(f"训练集写入: {train_txt}")
    print(f"验证集写入: {val_txt}")

# 示例用法：
split_dataset(
    image_dir='/data/yuesang/person_attribute/train',  # 替换成你的路径
    train_txt='train.txt',
    val_txt='val.txt'
)
