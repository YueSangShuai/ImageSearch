import json
import os


# 原始标签列表
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

test_attrs=['ublack', 'ugray', 'ublue', 'ugreen','uwhite', 'upurple', 'ured', 'ubrown', 'uyellow', 'upink', 'uorange',
 'lwhite', 'lpink', 'lred', 'lgreen','lyellow', 'lpurple', 'lbrown', 'lblack', 'lorange', 'lblue', 'lgray']




def get_json_data(json_path,image_dir,output_json):
    with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            rescult2json=[]
            for image_name,labels in json_content.items():
                image_path=os.path.join(image_dir, image_name)
                text_path=os.path.join(image_dir, os.path.splitext(image_name)[0]+".txt")
                
                with open(text_path, 'w', encoding='utf-8') as txt_f:
                    txt_f.write("")
                
                
                attr_dict={
                    "image_path":image_path,
                    "text_path":text_path
                }

                for attr in attribute_labels:
                    attr_dict[attr]=-1
                
                for test_attr,label in zip(test_attrs,labels):
                    attr_dict[test_attr]=label
                
                rescult2json.append(attr_dict)
                
            with open(output_json, 'w', encoding='utf-8') as out_f:
                json.dump(rescult2json, out_f, ensure_ascii=False, indent=2)


if __name__=="__main__":
    json_path="/data/yuesang/LLM/contrastors/data/test/pred_lb_checked.json"
    image_path="/data/yuesang/LLM/contrastors/data/test/nvr_data"
    output_json="/data/yuesang/LLM/contrastors/data/test/test.json"
    get_json_data(json_path,image_path,output_json)