import yaml

attribute_labels = [
    'Female',
    'AgeOver60', 'Age18-60', 'AgeLess18',
    'Front', 'Side', 'Back',
    'Hat', 'Glasses',
    'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve', 'UpperStride',
    'UpperLogo', 'UpperPlaid', 'UpperSplice', 
    'LowerStripe', 'LowerPattern',
    'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress',
    'boots',
    'ublack', 'ugray', 'ublue', 'ugreen','uwhite', 'upurple', 'ured', 'ubrown', 'uyellow', 'upink', 'uorange', 'ubeige', 'ustriped_color', 'umulticolor',
    'lwhite', 'lpink', 'lred', 'lgreen','lyellow', 'lpurple', 'lbrown', 'lblack', 'lorange', 'lblue', 'lgray', 'lbeige', 'lstriped_color', 'lmulticolor'
]

temp_dict={}

for binary in attribute_labels:
    temp_dict[binary]={}
    temp_dict[binary]["nc"]=2
    temp_dict[binary]["text_en"]={}


    
with open("/data/yuesang/LLM/contrastors/src/contrastors/configs/attributes/pa_attributes.yaml", "w", encoding="utf-8") as f:
    yaml.dump(temp_dict, f, 
              default_flow_style=False,
              sort_keys=False,
              allow_unicode=True)

print("PETA 属性 YAML 文件已生成：peta_attributes.yaml")
    
