import os

def get_dict(label_file):
    result = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            result.append({
                "image_path": values[0],
                "label1": values[1],
                "label2": values[2],
                "label1_score": values[3],
                "label2_score": values[4]
            })
    return result


def build_path_dict(dict_list):
    return {item["image_path"]: item for item in dict_list}


def get_rescult(all_dicts, output_file):
    if len(all_dicts) < 2:
        print("请至少传入两个列表：一个主列表，和一个或多个对比列表")
        return

    main_dict = all_dicts[0]
    other_dicts = all_dicts[1:]
    path_maps = [build_path_dict(d) for d in other_dicts]

    
    
    os.makedirs(os.path.split(output_file)[0],exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in main_dict:
            image_path = item["image_path"]
            labels = [item["label1"], item["label2"]]
            scores = [item["label1_score"], item["label2_score"]]

            for path_map in path_maps:
                match = path_map.get(image_path, {})
                labels.append(match.get("label1", ""))
                labels.append(match.get("label2", ""))
                scores.append(match.get("label1_score", ""))
                scores.append(match.get("label2_score", ""))

            f.write(f"{image_path}\t" + "\t".join(labels) + "\t" + "\t".join(scores) + "\n")

    print(f"✅ 已写入结果到: {output_file}")


            
if __name__=="__main__":
    labels=[
        "/data/yuesang/LLM/VectorIE/classifier_label/output/Nomic/normal/gender_output.txt",
        "/data/yuesang/LLM/VectorIE/classifier_label/output/MexMA-SigLIP2/normal/gender_output.txt",
           ]
    dict_list = [get_dict(label) for label in labels]
    get_rescult(dict_list, "/data/yuesang/LLM/VectorIE/classifier_label/output/conbime_Nomic_MexMA/normal/gender_output.txt")
    