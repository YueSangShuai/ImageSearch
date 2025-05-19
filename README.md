# 图像-文本匹配 模型蒸馏 方法

这是一个基于PyTorch的深度学习模型，用于学习图像和文本的联合嵌入空间，实现跨模态匹配。

## 1. 项目背景

基于clip/Siglip2的图像-文本匹配模型，实现了较好的跨模态匹配性能，从而可以实现较好的文本到图像的检索性能。

然而，由于Siglip2视觉编码器和文本编码器非常庞大，难以在低端设备上运行，因此本项目的目的，就是对其进行蒸馏，使用较小的视觉编码器和文本编码器来学习它们在特定数据集（如行人数据集）上的跨模态匹配。

## 2. 主要功能

- 蒸馏Siglip2模型
- 图像-文本联合嵌入学习
- 支持分布式训练和混合精度训练
- 多种损失函数组合优化

## 3. 模型架构

本项目的模型，尽量使用能够在既有的计算资源上运行的算子。

- **图像编码器**：可配置的CNN骨干网络
- **文本编码器**：基于Transformer的字符级编码器（为了在低端设备上运行，不使用传统的分词器方法）

- **损失函数**：
  - 跨模态相似度蒸馏损失
  - 模态内关系蒸馏损失
  - 模态内、跨模态长记忆对比损失
  - 跨模态相似度损失

## 4. 安装


```bash
# 下载源码
git clone ...

# 安装依赖
pip install torch torchvision numpy pillow tensorboard
```

*项目结构*

```
person/
├── main.py         # 训练主程序
├── model.py        # 模型定义
├── text_encoder.py # 文本编码器
├── data.py         # 数据加载
├── opts.py         # 参数配置
... ...
└── losses.py       # 损失函数实现
```

## 5. 使用方法

### 5.1 数据准备

 - 下载数据集（如：SYNTH-PEDES）
    - SYNTH-PEDES/
 - 下载 Siglip2 预训练模型(huggingface)
    - visheratin/mexma-siglip2
    - google/siglip2-so400m-patch16-512
    - facebook/MEXMA
 - 使用 tools/extract_label.py 提取文本和图像特征
    ```bash
    python tools/extract_label.py --data SYNTH-PEDES --embedding_path embeddings
    ```
    将提取的特征保存到 embeddings 目录下

我们初步评估了 Siglip2 的行人检索能力，结果如下：

```
python evaluation_labels.py
... ... 
        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    12.2132%  90.5400%  98.5000%  99.9200%  100.0000%  100.0000%
SCORE   0.0641772 0.153719  0.180852  0.201203  0.213413  0.217483
Verify Count=12466498. Score in [ 0 - 0 ]
```

大概在 87.7868% 的召回率和精确率。


### 5.2 训练模型

*单机训练* 
```bash
CUDA_VISIBLE_DEVICES=7 python main.py --backbone backbone.mf3.ANet2 --data SYNTH-PEDES/synthpedes-dataset.json --embedding_path embeddings
```

*分布式训练* 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --backbone backbone.mf3.ANet2 --data SYNTH-PEDES/synthpedes-dataset.json --embedding_path embeddings
```

*关键参数*

 - `--backbone`: 图像编码器骨干网络
 - `--data`: 数据集路径
 - `--embedding_path`: 预计算特征路径（即使用siglip2预先计算的文本和图像特征）
 - `--feature_dim`: 新模型的嵌入特征的维度
 - `--max_seq_length`: 文本最大长度
 - `--dist_rel_weight`: 跨模态关系蒸馏损失权重
 - `--rkda_weight`: 角度知识蒸馏损失权重

*训练示例*

```
richard@ubuntu02:/cache/richard/work/person$ python main.py --compile
... ...
2025-04-07 09:28:26,714-INFO:Iter:19940| dist_rel: 1.16826 1.33665| id_img: 0.08200 0.09867| id_cap: 0.09143 0.10874| id_img_cap: 0.04163 0.04315| d_pos: 0.66125 0.67552| s_neg: 0.02730 0.03424| total: 2.07187 2.29696| lr: 0.000996
2025-04-07 09:29:03,757-INFO:Iter:19960| dist_rel: 1.13122 1.33616| id_img: 0.08053 0.09866| id_cap: 0.09231 0.10874| id_img_cap: 0.04188 0.04314| d_pos: 0.65069 0.67539| s_neg: 0.01689 0.03423| total: 2.01352 2.29632| lr: 0.000996
2025-04-07 09:29:40,784-INFO:Iter:19980| dist_rel: 1.10517 1.33588| id_img: 0.09349 0.09864| id_cap: 0.10820 0.10872| id_img_cap: 0.04169 0.04314| d_pos: 0.66298 0.67531| s_neg: 0.01882 0.03422| total: 2.03035 2.29591| lr: 0.000996
2025-04-07 09:30:17,823-INFO:Iter:20000| dist_rel: 1.33219 1.33548| id_img: 0.10055 0.09865| id_cap: 0.11068 0.10873| id_img_cap: 0.04147 0.04313| d_pos: 0.65570 0.67520| s_neg: 0.03364 0.03421| total: 2.27422 2.29541| lr: 0.000996
2025-04-07 09:30:17,824-INFO:Saving checkpoint to logs/expModel-backbone.mf3.ANet2-224x112-512/20/20000.pt
... ...

```

### 5.3 评估模型

使用 evaluation_model.py 脚本评估模型性能。

```bash
python evaluation_model.py <checkpoint>
```

该脚本会使用新训练的模型，分别提取图像和文本的特征，然后进行交叉比对。成对的特征进行比对为正例样本，图像特征和其他不同id的文本特征进行比对为负例样本。最后计算ROC曲线，输出EER、FAR等指标。

*评估示例*

```bash
richard@ubuntu02:/cache/richard/work/person$ python evaluation_model.py logs/expModel-backbone.mf3.ANet2-224x112-512/8/40000.pt
... ...
5000: Part1/362/13.jpg
-194 634
        EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    12.4032%  89.5400%  97.9400%  99.7400%  99.9200%  99.9600%
SCORE   0.213366 0.523672  0.601249  0.656199  0.694987  0.704684
Verify Count=12466498. Score in [ 0 - 0 ]
Top-15 False Rejected:
         361/Part1/362_9.jpg 361/Part1/362_9.jpg_caption -0.2114231288433075
         361/Part1/362_0.jpg 361/Part1/362_0.jpg_caption -0.16850851476192474
         361/Part1/362_8.jpg 361/Part1/362_8.jpg_caption -0.14855709671974182
         15/Part1/16_10.jpg 15/Part1/16_10.jpg_caption -0.119660384953022
         361/Part1/362_10.jpg 361/Part1/362_10.jpg_caption -0.09391786903142929
         346/Part1/347_4.jpg 346/Part1/347_4.jpg_caption -0.0865994468331337
         361/Part1/362_6.jpg 361/Part1/362_6.jpg_caption -0.05766158923506737
         15/Part1/16_9.jpg 15/Part1/16_9.jpg_caption -0.05314112827181816
         357/Part1/358_9.jpg 357/Part1/358_9.jpg_caption -0.04702446982264519
         3/Part1/4_2.jpg 3/Part1/4_2.jpg_caption -0.04456719011068344
         310/Part1/311_2.jpg 310/Part1/311_2.jpg_caption -0.04227234050631523
         37/Part1/38_10.jpg 37/Part1/38_10.jpg_caption -0.04224896430969238
         351/Part1/352_2.jpg 351/Part1/352_2.jpg_caption -0.03817720711231232
         357/Part1/358_17.jpg 357/Part1/358_17.jpg_caption -0.03372059017419815
         120/Part1/121_3.jpg 120/Part1/121_3.jpg_caption -0.030892256647348404
Top-15 False Accepted:
         263/Part1/264_0.jpg 142/Part1/143_15.jpg_caption 0.7036939263343811
         286/Part1/287_9.jpg 166/Part1/167_4.jpg_caption 0.7034932374954224
         286/Part1/287_9.jpg 166/Part1/167_6.jpg_caption 0.7034932374954224
         286/Part1/287_9.jpg 166/Part1/167_7.jpg_caption 0.7034932374954224
         286/Part1/287_9.jpg 166/Part1/167_8.jpg_caption 0.7034932374954224
         286/Part1/287_9.jpg 166/Part1/167_9.jpg_caption 0.7034932374954224
         132/Part1/133_16.jpg 126/Part1/127_0.jpg_caption 0.6986109018325806
         106/Part1/107_8.jpg 101/Part1/102_15.jpg_caption 0.6963139176368713
         263/Part1/264_2.jpg 132/Part1/133_2.jpg_caption 0.6959272623062134
         132/Part1/133_16.jpg 65/Part1/66_18.jpg_caption 0.6955647468566895
         132/Part1/133_16.jpg 126/Part1/127_5.jpg_caption 0.6939175128936768
         132/Part1/133_16.jpg 126/Part1/127_7.jpg_caption 0.6939175128936768
         132/Part1/133_16.jpg 126/Part1/127_11.jpg_caption 0.6939175128936768
         132/Part1/133_5.jpg 44/Part1/45_2.jpg_caption 0.6919234395027161
         263/Part1/264_1.jpg 142/Part1/143_15.jpg_caption 0.6899242401123047
richard@ubuntu02:/cache/richard/work/person$ python evaluation_model.py logs/expModel-backbone.mf3.ANet2-224x112-512/8/135000.pt          
       EER      FAR=1e-3  FAR=1e-4  FAR=1e-5  FAR=1e-6  FAR=0
----------------------------------------------------------------
FRR    11.5197%  88.6400%  98.2400%  99.6400%  99.9400%  99.9800%
SCORE   0.224563 0.536564  0.625238  0.676144  0.723765  0.74347          
```

EER = 12.4032% 相当于实现了 87.5968% 的召回率和精确率。相当于基本实现了siglip2的行人检索能力。当然，由于数据集标注caption的局限性，实际性能会远低于siglip2，更没有多语言的能力。

### 5.4 导出模型

使用 export_model.py 脚本导出模型。

```bash
python export_model.py <checkpoint> <output_dir>
```

*导出示例*

```bash
richard@ubuntu02:/cache/richard/work/person$ python export_model.py logs/expModel-backbone.mf3.ANet2-224x112-512/8/35000.pt logs/expModel-backbone.mf3.ANet2-224x112-512/8/
... ...
export to "logs/expModel-backbone.mf3.ANet2-224x112-512/8/model_text.onnx" OK.
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add        │ 67             │ 67               │
│ Constant   │ 264            │ 176              │
│ Conv       │ 14             │ 14               │
│ MatMul     │ 49             │ 49               │
│ Max        │ 1              │ 1                │
│ Mul        │ 61             │ 61               │
│ ReduceMax  │ 1              │ 1                │
│ Relu       │ 7              │ 7                │
│ Reshape    │ 39             │ 39               │
│ Slice      │ 8              │ 8                │
│ Softmax    │ 6              │ 6                │
│ Sub        │ 6              │ 0                │
│ Tanh       │ 18             │ 18               │
│ Transpose  │ 37             │ 37               │
│ Trilu      │ 6              │ 0                │
│ Unsqueeze  │ 12             │ 0                │
│ Model Size │ 27.8MiB        │ 15.6MiB          │
└────────────┴────────────────┴──────────────────┘
test onnx model ... ...
output[1, 512]: min=-4.49576, max=41.99710
```

实际会在 `output_dir` 导出三个文件：

- `model_text.onnx` 文本编码器
- `model_image.onnx` 图像编码器
- `input_embeddings.pt.npy` 输入字符嵌入映射表，它把4字节utf8字符映射到256维的浮点数向量。这个部分从文本编码器中分开的目的是为了增加`model_text.onnx`的可移植性（很多`NPU`推理框架不支持`nn.Embedding`）。

## 6. Search 测试

在导出模型为 onnx 的两个文件 `model_text.onnx` 和 `model_image.onnx` 的基础上，进入到 search 目录，在此目录下运行程序。

### 6.1 注册图像

先准备好数据集，例如 `xm_images` 目录下有若干张图片，然后运行：

```bash
python enroll_by_onnx.py --image-dir xm_images --output-prefix collection/person_features
```

会生成两个文件：
  - `collection/person_features_features.npy`：特征向量
  - `collection/person_features_metadata.json`：元数据

注，脚本 `enroll_by_siglip.py` 用于使用`mexma-siglip2`或其他 huggingface 的模型注册图像，

### 6.2 Search  

运行：

```bash
cd search
python api.py --image-root xm_images --image_feature collection/person_features_features.npy --image_metadata collection/person_features_metadata.json\ 
--queries "a girl wearing red coat" "a man wearing a gray pands" --top-n 2
  ```

会输出类似以下结果：

```bash
----------
Query: 'a girl wearing red coat'
        Search completed in 0.1603 seconds.
1. Score: 0.3819
   Image: xm_images/20250307172329_05dc54fb0692482ba97870d499661a20_3989585.png
   BBox: [393, 186, 477, 471]
2. Score: 0.3609
   Image: xm_images/20250313071719_e84613a6ea05428ea99cfa9522092ffd_8464113.png
   BBox: [302, 262, 384, 470]

----------
Query: 'a man wearing a gray pands'
        Search completed in 0.2119 seconds.
1. Score: 0.3970
   Image: xm_images/20250313082037_32c91c076e6546b1ad71fca685999a21_3662016.png
   BBox: [306, 114, 342, 240]
2. Score: 0.3854
   Image: xm_images/20250311115944_38ebefe6149b46f5a52467c0115436c3_6291011_W640_H480_I5_O4_T20.png
   BBox: [415, 165, 495, 420]  
```

### 6.3 Web UI

`app_onnx.py` 用于使用 onnx 模型做文本向量化进行搜索；`app_siglip.py` 用于使用 `mexma-siglip2`或其他 huggingface 的模型做文本向量化进行搜索。

```bash
python app_onnx.py --image-root xm_images --image_feature collection/person_features_features.npy \
   --image_metadata collection/person_features_metadata.json --port 5005 \
   --text-model ../../text_dist/logs/expModel-tbackbone.minimind.L3-224x112-1152/15/model_text.onnx 
```

注意，可以使用`mexma-siglip2`或其直接蒸馏的图像模型注册图像，然后使用蒸馏自 `mexma-siglip2` 的文本模型进行检索。

```
richard@ubuntu02:/cache/richard/work/person/search$ python app_onnx.py  \
  --text-model ../../text_dist/logs/expModel-tbackbone.minimind.L3-224x112-1152/15/model_text.onnx \
  --image-root xm_images \
  --port 5006
```

## 7. TODO

- [x] 实现文本-图像检索界面 

- 生成丰富的文本数据。目前采用 PLIP 提供的数据集，标注文件 SYNTH-PEDES/synthpedes-dataset.json 中 caption 过于单一化，比如对人的描述忽略了很多细节，表述顺序也过于单一，因此导致查询时候的文本与该顺序不一致时，召回率会很低。

- 中文训练数据，实现中文的描述数据集

- 更可靠的模型性能评估方法

- [x] 采用bert类模型，实现更通用的文本编码器
