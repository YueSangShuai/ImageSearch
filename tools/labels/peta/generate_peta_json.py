import json
import os

binary_classes = [
    "accessoryHeadphone", "personalLess15", "personalLess30", "personalLess45", "personalLess60", "personalLarger60",
    "carryingBabyBuggy", "carryingBackpack", "hairBald", "footwearBoots", "lowerBodyCapri", "carryingOther",
    "carryingShoppingTro", "carryingUmbrella", "lowerBodyCasual", "upperBodyCasual", "personalFemale", "carryingFolder",
    "lowerBodyFormal", "upperBodyFormal", "accessoryHairBand", "accessoryHat", "lowerBodyHotPants", "upperBodyJacket",
    "lowerBodyJeans", "accessoryKerchief", "footwearLeatherShoes", "upperBodyLogo", "hairLong", "lowerBodyLongSkirt",
    "upperBodyLongSleeve", "lowerBodyPlaid", "lowerBodyThinStripes", "carryingLuggageCase", "personalMale",
    "carryingMessengerBag", "accessoryMuffler", "accessoryNothing", "carryingNothing", "upperBodyNoSleeve",
    "upperBodyPlaid", "carryingPlasticBags", "footwearSandals", "footwearShoes", "hairShort", "lowerBodyShorts",
    "upperBodyShortSleeve", "lowerBodyShortSkirt", "footwearSneaker", "footwearStocking", "upperBodyThinStripes",
    "upperBodySuit", "carryingSuitcase", "lowerBodySuits", "accessorySunglasses", "upperBodySweater",
    "upperBodyThickStripes", "lowerBodyTrousers", "upperBodyTshirt", "upperBodyOther", "upperBodyVNeck"
]

def load_labels(onehot_txt_path):
    id_to_vector = {}
    with open(onehot_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != (1 + len(binary_classes)):
                continue
            id = parts[0]
            values = list(map(int, parts[1:]))
            id_to_vector[id] = values
    return id_to_vector

def generate_json(image_dir, onehot_txt_path, output_json_path, label_dir="labels"):
    id_to_vector = load_labels(onehot_txt_path)
    results = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.bmp', '.jpg', '.png')):
            continue
        id = filename.split("_")[0]

        if id in id_to_vector:
            vector = id_to_vector[id]
            txt_name=os.path.splitext(filename)[0]+".txt"
            item = {
                "image_path": os.path.join(image_dir, filename),
                "text_path": os.path.join(image_dir, txt_name)
            }
            # 将binary_classes对应的标签映射为键值对
            for cls, val in zip(binary_classes, vector):
                item[cls] = val
            results.append(item)
        else:
            print(f"[警告] 找不到 ID {id} 的标签")

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"已生成 {len(results)} 条样本到 {output_json_path}")

def get_filelist(dir):
    """获取文件夹下所有图片路径（保持原结构）"""
    filelist = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    for home, dirs, files in os.walk(dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in image_extensions:
                filelist.append(os.path.join(home, filename))
    return filelist

txt_path = [
    "/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/CAVIAR4REID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/CUHK/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/GRID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/i-LID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/MIT/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/PRID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/SARC3D/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/TownCentre/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/VIPeR/archive/",
]

for txt in txt_path:
    save_path=os.path.join(os.path.split(txt)[0],"one_hot_labels.txt")
    generate_json(txt, save_path,os.path.join(txt,"dataset.json"))
    print(len(get_filelist(txt)))



train_json=[

    "/data/yuesang/LLM/contrastors/data/PETA/CUHK/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/GRID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/i-LID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/MIT/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/PRID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/SARC3D/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/TownCentre/archive/",
]

val_json=[

    "/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/CAVIAR4REID/archive/",
    "/data/yuesang/LLM/contrastors/data/PETA/VIPeR/archive/",
]


def merge_jsons(dir_list):
    merged = []
    for d in dir_list:
        json_path = os.path.join(d, "dataset.json")
        if not os.path.isfile(json_path):
            print(f"[警告] 找不到: {json_path}")
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
            merged.extend(data)
    return merged

# 合并数据
train_data = merge_jsons(train_json)
val_data = merge_jsons(val_json)

# 输出结果
with open("train_peta.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("val_peta.json", "w") as f:
    json.dump(val_data, f, indent=4)

print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")



# # 示例调用
# generate_json(
#     image_dir="/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive",
#     onehot_txt_path="/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive/one_hot_labels.txt",
#     output_json_path="dataset.json"
# )
