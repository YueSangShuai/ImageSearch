本项目实现基于行人的语言嵌入模型的蒸馏
------------------------------------------

教师模型：

https://huggingface.co/facebook/MEXMA


## 1. 数据准备

本项目搜集行人的语言数据，来自项目：
- https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-10M
- https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-15M


```python
python data.py
```

该脚本用于HumanCaption-10M的 parquet 格式数据，转换为 jsonl 格式数据，并翻译成中文，保存到 `d.jsonl` 文件中，`d.jsonl` 文件将用于后续的蒸馏训练。

## 2. 训练

```python
python main.py -b 200 --max_seq 128 --teachor-info-weight 0 --data d.jsonl
```

各种损失参数说明：

  - `--teachor-info-weight` 权重系数: InfoNCE loss
  - `--temperature` Temperature for InfoNCE loss
  - `--teachor-rel-weight`  权重系数: 师生关系损失
  - `--teachor-sim-weight`  权重系数: 自相似度损失
  - `--teachor-l2-weight`  权重系数: mse loss
  - `--teachor-l1-weight`  权重系数: smooth l1 loss

训练日志：

```bash
richard@ubuntu02:/cache/richard/work/text_dist$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,5 torchrun --nproc_per_node=7 --master_port 29501 main.py -b 240 --text-b tbackbone.minimind.L3 --max_seq_length 128 --lr 1e-3 --lr-cos 40100 --lr-warm 100
... ...
2025-04-14 11:44:02,471-INFO:Iter:39940| sim: 0.0985 0.10| Csim: 0.1039 0.10| rel: 10.3675 10.19| Crel: 10.3891 10.24| l1: 0.9978 1.02| Cl1: 1.1009 1.13| info: 0.0455 0.05| Cinfo: 0.0468 0.05| total: 23.150 22.87| lr: 0.000020
2025-04-14 11:44:24,527-INFO:Iter:39960| sim: 0.1064 0.10| Csim: 0.1097 0.10| rel: 10.7738 10.19| Crel: 10.8545 10.24| l1: 0.9779 1.02| Cl1: 1.0908 1.13| info: 0.0668 0.05| Cinfo: 0.0741 0.05| total: 24.054 22.87| lr: 0.000020
2025-04-14 11:44:46,522-INFO:Iter:39980| sim: 0.1052 0.10| Csim: 0.1076 0.10| rel: 10.5884 10.19| Crel: 10.6177 10.24| l1: 0.9765 1.02| Cl1: 1.0633 1.13| info: 0.0528 0.05| Cinfo: 0.0544 0.05| total: 23.566 22.87| lr: 0.000020
2025-04-14 11:45:08,516-INFO:Iter:40000| sim: 0.0928 0.10| Csim: 0.1006 0.10| rel: 9.8265 10.19| Crel: 9.9143 10.24| l1: 1.0092 1.02| Cl1: 1.1621 1.13| info: 0.0297 0.05| Cinfo: 0.0307 0.05| total: 22.166 22.87| lr: 0.000020
2025-04-14 11:45:08,516-INFO:Saving checkpoint to logs/expModel-tbackbone.minimind.L3-224x112-1152/15/40000.pt
2025-04-14 11:45:30,621-INFO:Iter:40020| sim: 0.0897 0.10| Csim: 0.0921 0.10| rel: 9.5677 10.12| Crel: 9.6170 10.17| l1: 1.0109 1.02| Cl1: 1.1086 1.13| info: 0.0406 0.05| Cinfo: 0.0421 0.05| total: 21.569 22.73| lr: 0.000020
2025-04-14 11:45:52,655-INFO:Iter:40040| sim: 0.0952 0.10| Csim: 0.1000 0.10| rel: 9.9753 10.22| Crel: 9.9904 10.27| l1: 1.0316 1.02| Cl1: 1.1427 1.13| info: 0.0412 0.05| Cinfo: 0.0425 0.05| total: 22.419 22.95| lr: 0.000020
2025-04-14 11:46:14,661-INFO:Iter:40060| sim: 0.1050 0.10| Csim: 0.1107 0.10| rel: 10.2602 10.22| Crel: 10.3078 10.27| l1: 1.0282 1.02| Cl1: 1.1408 1.13| info: 0.0585 0.05| Cinfo: 0.0619 0.05| total: 23.073 22.95| lr: 0.000020
2025-04-14 11:46:36,666-INFO:Iter:40080| sim: 0.1080 0.10| Csim: 0.1116 0.10| rel: 10.2450 10.23| Crel: 10.2807 10.28| l1: 1.0208 1.02| Cl1: 1.1134 1.13| info: 0.0579 0.05| Cinfo: 0.0603 0.05| total: 22.998 22.96| lr: 0.000020
```

注意，最终的 smooth-l1 损失: 1.0208（英文），1.1134（中文）。这个值基本达到了一定的理解水平。



## 3. 导出模型

```python
python export_model.py <model_path> <output> [<device>]
```

`device` 可以是 `cpu` 或 `npu`, 默认为 `cpu`. `npu` 模式用于兼容旧的NPU硬件（ 输入 Nx1xHxW 的图片）


```
richard@ubuntu02:/cache/richard/work/text_dist$ python export_model.py \
  logs/expModel-tbackbone.minimind.L3-224x112-1152/15/40000.pt \
  logs/expModel-tbackbone.minimind.L3-224x112-1152/15/
```

将在 `logs/expModel-tbackbone.minimind.L3-224x112-1152/15/` 目录下生成 `model_text.onnx`，可用于后续的任务。

## 4. 评估

在项目 `git@192.168.59.234:Richard/image_text_embedding.git` 中，行人搜索脚本 `app_onnx.py` 可以用于体验模型的性能。

```
richard@ubuntu02:/cache/richard/work/person/search$ python app_onnx.py \
  --text-model ../../text_dist/logs/expModel-tbackbone.minimind.L3-224x112-1152/15/model_text.onnx

```

