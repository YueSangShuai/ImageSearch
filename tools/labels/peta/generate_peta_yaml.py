import yaml

binary_classes=[
"accessoryHeadphone",
"personalLess15",
"personalLess30",
"personalLess45",
"personalLess60",
"personalLarger60",
"carryingBabyBuggy",
"carryingBackpack",

"hairBald",
"footwearBoots",
"lowerBodyCapri",
"carryingOther",
"carryingShoppingTro",
"carryingUmbrella",
"lowerBodyCasual",
"upperBodyCasual",
"personalFemale",
"carryingFolder",

"lowerBodyFormal",
"upperBodyFormal",
"accessoryHairBand",
"accessoryHat",
"lowerBodyHotPants",
"upperBodyJacket",
"lowerBodyJeans",
"accessoryKerchief",
"footwearLeatherShoes",
"upperBodyLogo",
"hairLong",

"lowerBodyLongSkirt",
"upperBodyLongSleeve",
"lowerBodyPlaid",
"lowerBodyThinStripes",
"carryingLuggageCase",
"personalMale",
"carryingMessengerBag",
"accessoryMuffler",
"accessoryNothing",
"carryingNothing",

"upperBodyNoSleeve",
"upperBodyPlaid",
"carryingPlasticBags",
"footwearSandals",
"footwearShoes",
"hairShort",
"lowerBodyShorts",
"upperBodyShortSleeve",
"lowerBodyShortSkirt",
"footwearSneaker",
"footwearStocking",
"upperBodyThinStripes",
"upperBodySuit",

"carryingSuitcase",
"lowerBodySuits",
"accessorySunglasses",
"upperBodySweater",
"upperBodyThickStripes",
"lowerBodyTrousers",

"upperBodyTshirt",
"upperBodyOther",
"upperBodyVNeck",
]

mult_classes=[
"footwear",
"hair",
"lowerbody",
"upperbody",
]

temp_dict={}

for binary in binary_classes:
    temp_dict[binary]={}
    temp_dict[binary]["nc"]=2
    temp_dict[binary]["text_en"]={}

for mult in mult_classes:
    temp_dict[binary]={}
    temp_dict[binary]["nc"]=5
    temp_dict[binary]["text_en"]={}
    
with open("peta_attributes.yaml", "w", encoding="utf-8") as f:
    yaml.dump(temp_dict, f, 
              default_flow_style=False,
              sort_keys=False,
              allow_unicode=True)

print("PETA 属性 YAML 文件已生成：peta_attributes.yaml")
    
