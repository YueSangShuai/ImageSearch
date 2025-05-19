# 打对数据集中的行人图像打标签
## 注册
目前该框架按照ImageSearch实现了以下几种多模态模型的推理:BGE-VL系列，mexma-siglip2，nomic，PLIP  
### 1.BGE-VL系列，mexma-siglip2
BGE-VL系列，mexma-siglip2可以按照[模型地址](http://10.12.16.123:5000/repository/models)下载完即可使用 <br>
### 2.noic
noic的模型的图像模型和文本模型是分开的需要两个一起下载，然后添加额外的configuration_hf_nomic_bert.py和modeling_hf_nomic_bert.py，下载代码如下:
```
from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="nomic-ai/nomic-bert-2048",
        filename="configuration_hf_nomic_bert.py",
        local_dir=model_path
    )
    hf_hub_download(
        repo_id="nomic-ai/nomic-bert-2048",
        filename="modeling_hf_nomic_bert.py",
        local_dir=model_path
    )
```
将下载好的文件放入到nomic文本模型和图像模型的文件夹下，并且修改其中的config文件 <br>
nomic-embed-text-v1.5
```
"auto_map": {                                                  
    "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
    "AutoModel": "modeling_hf_nomic_bert.NomicBertModel",
    "AutoModelForMaskedLM": "modeling_hf_nomic_bert.NomicBertForPreTraining"                                  
  },
```
nomic-embed-vision-v1.5
```
"auto_map": {
    "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
    "AutoModel": "modeling_hf_nomic_bert.NomicVisionModel"
  },
```
### 3.PLIP
PLIP模型的下载地址如下[PLIP](http://10.12.16.123:5000/download/zkteco/PLIP.tgz) <br>
PLIP模型需要额外的库用来将中文部分转英文部分[bert](http://10.12.16.123:5000/repository/models?path=bert-base-uncased) <br>
最终的文件结构如下:<br>
![alt text](image-3.png)

### 4.多模型注册
采用上述的模型进行注册并将注册后的向量进行融合,设置conbime_model_path的参数,里面需要填写模型的路径可以参考上述



### 4.运行
注册相关的代码位于./get_rescult/enroll/function.py中，注册时候主要关注下面这几个参数model_file（模型路径），inference_img（需要输入一张图片用来获得qdrant注册向量的大小），collection_name（注册数据库的名称）enroll_path（注册图片的路径）其他默认即可
对于PLIP模型需要额外设置txt_backbone的路径及上述提到的[bert](http://10.12.16.123:5000/repository/models?path=bert-base-uncased)这个模型 <br>


## 标签
总共有三种打标签方式一共四种结果分别如下:
### 1、 策略一与策略二
get_rescult/get_label/class_label_step1.py 可以获得策略1：选择相似度最高的标签/别名对应的类别作为分类结果，策略2：选择一组标签/别名中相似度均值最高的类别作为分类结果 <br>
需要设置和funtion基本一致

```
python -m class_label_step1.py >output.txt
```

### 2、 合并整理(策略一&&策略二)同id的标签
get_rescult/get_label/class_label_step2.py
根据第一种打标结果分别设置label_input_file和output_file即可

```
python -m class_label_step2.py
```

### 3、 合并整理(策略一||策略二)同id的标签
get_rescult/get_label/class_label_step3.py
根据第一种打标结果分别设备label_input_file和output_file，然后通过bool来选择你是要根据策略一还是策略2

```
python -m class_label_step3.py
```


## 验证精度
### 1、 性别
get_rescult/compare/compare_gender.py
设置file_path的路径以及is_conbime参数，当你的文档里面带有标签分数时需要设置is_conbime=False,没有标签分数时设置True即可

```
python -m compare_gender.py
```
### 2、 年龄
get_rescult/compare/compare_age.py
设置file_path的路径以及is_conbime参数，当你的文档里面带有标签分数时需要设置is_conbime=False,没有标签分数时设置True即可

```
python -m compare_age.py
```
