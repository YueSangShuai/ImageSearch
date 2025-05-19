# 本程序实现对图像生成分类器标签的合并整理
# 2025-03-15

import sys
import os

'''
之前的程序按照如下格式输出分类结果：
    print(f"\t{image_path}\t{label_names[i1][0]}\t{label_names[i2][0]}\t{s1.item():.4f}\t{s2.item():.4f}")
这个程序分析这个结果，选择最佳的标签。选择方法是这样的：
请看一个例子
data/SYNTH-PEDES/Part3/78923/4.jpg      male    male    0.1189  0.0626
image_path 是 data/SYNTH-PEDES/Part3/78923/4.jpg
label1 是 male
label2 是 male
注意，image_path可以分解成两部分，第一部分是身份id，第二部分是图片id。
本例中，身份id是data/SYNTH-PEDES/Part3/78923，图片id是4.jpg
因此我们把同一个身份id的每个图片的标签分析出来，选择最多出现的标签作为最终标签就可以了。
最后，我们把每一个id的最终标签赋予到该身份的每一个图片。
'''

def choice_label(label_file, output_file,n):
    id2label = {}
    

    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 3:
                continue
            
            
            
            image_path = values[0]
            txt_labels=[values[i] for i in range(1,int(len(values)/2)+1)]
            if n<1 or n>int(len(values)/2)+1:
                raise ValueError("n must in range")
            
            
            is_id = "id" in image_path.split("/")[-2]
            if not is_id:
                continue
            
            
            uid = os.path.split(image_path)[0]
            if uid not in id2label:
                id2label[uid] = {}
            ulabels = id2label[uid]
            
            if txt_labels[n-1] in ulabels:
                ulabels[txt_labels[n-1]] += 1
            else:
                ulabels[txt_labels[n-1]] = 1
            

    os.makedirs(os.path.split(output_file)[0],exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for uid, ulabels in id2label.items():
            max_label = max(ulabels.items(), key=lambda x: x[1])[0]
            for f in os.listdir(uid):
                ext = os.path.splitext(f)[1]
                if ext.lower() in (".jpg", ".png", ".jpeg"):
                    out_f.write(f"{uid}/{f}\t{max_label}\t{max_label}\n")


if __name__ == "__main__":
    
    n=2
    bool=f"label{n}"
    
    label_input_file = "/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/normal/age_output.txt"
    output_file = f"/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/{bool}/age_output.txt"
    choice_label(label_input_file, output_file,n)
    
    
    label_input_file = "/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/normal/gender_output.txt"
    output_file = f"/data/yuesang/LLM/VectorIE/classifier_label/output/PE-Core-G14-448/{bool}/gender_output.txt"
    choice_label(label_input_file, output_file,n)