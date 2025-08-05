import os

# 设置你的图片文件夹路径
image_folder = "/data/yuesang/person_attribute"  # TODO: 改成你的实际路径

# 支持的图片后缀
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

# 遍历文件夹，找出非图片文件并删除
non_image_files = []

for filename in os.listdir(image_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in image_extensions:
        file_path = os.path.join(image_folder, filename)
        non_image_files.append(filename)
        try:
            os.remove(file_path)
            print(f"已删除非图片文件: {filename}")
        except Exception as e:
            print(f"删除 {filename} 失败，原因：{e}")

if not non_image_files:
    print("全部都是图片文件，无需删除！")
