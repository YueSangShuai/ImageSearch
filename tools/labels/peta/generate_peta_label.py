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

def generate_onehot(label_line, binary_classes):
    parts = label_line.strip().split()
    name_or_id = parts[0]
    attributes = set(parts[1:])  # 属性集合
    onehot = [1 if attr in attributes else 0 for attr in binary_classes]
    return name_or_id, onehot

def process_txt_label(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            name, label_vec = generate_onehot(line, binary_classes)
            fout.write(name + ' ' + ' '.join(map(str, label_vec)) + '\n')




txt_path = [
    "/data/yuesang/LLM/contrastors/data/PETA/3DPeS/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/CAVIAR4REID/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/CUHK/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/GRID/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/i-LID/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/MIT/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/PRID/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/SARC3D/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/TownCentre/archive/Label.txt",
    "/data/yuesang/LLM/contrastors/data/PETA/VIPeR/archive/Label.txt",
]


for txt in txt_path:
    save_path=os.path.join(os.path.split(txt)[0],"one_hot_labels.txt")
    process_txt_label(txt, save_path)
