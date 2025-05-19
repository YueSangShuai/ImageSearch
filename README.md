# 火星慧知矢量发动机(Vector Engine) 行人搜索系统

## 项目概述

Vector Engine 是一个基于深度学习的图像搜索系统，允许用户通过自然语言描述搜索相关的人物图像。系统使用先进的多模态模型（MexMA-SigLIP2）将文本查询和图像转换为向量表示，并通过向量相似度匹配来检索最相关的图像。

## 主要功能

1. **自然语言图像搜索**：通过文字描述查找相关的人物图像
2. **以图搜图**：上传包含行人的图片，自动检测并搜索相似行人
3. **行人检测**：使用高效的ONNX模型自动识别图像中的行人
4. **图像注册**：将新图像添加到向量数据库中
5. **图像删除**：从系统中移除图像
6. **多语言支持**：支持中英文查询

## 技术架构

### 核心组件

- **多模态模型**：使用 MexMA-SigLIP2 模型进行文本和图像的向量编码
- **行人检测模型**：使用ONNX优化的目标检测模型快速识别图像中的行人
- **向量数据库**：使用 Qdrant 进行高效的向量存储和检索
- **Web 界面**：基于 Gradio 构建的用户友好界面

### 依赖项

- PyTorch 和 TorchVision：深度学习框架
- Transformers：用于加载和使用预训练模型
- ONNX Runtime：用于高效运行行人检测模型
- Qdrant Client：向量数据库客户端
- Gradio：Web 界面构建
- PIL：图像处理
- NumPy：数学计算
- OpenCV：视频处理和图像操作
- Flash Attention (可选)：用于优化注意力机制计算

## 使用指南

### 安装

1. 确保已安装 Python 3.8 或更高版本
2. 安装所需依赖项
3. 确保 Qdrant 服务器已在本地端口 6333 上运行
4. 下载 MexMA-SigLIP2 模型和行人检测ONNX模型并放置在适当位置


### 数据集注册

```bash
python enroll_images.py --collect zkteco_xm --enroll /data/zkteco/zkteco_xm_export_zfs/snap/ --detection
```

### 启动服务

```bash
python app.py --collect zkteco_xm --port 7862
```

### 参数说明

- `--model_path`：模型文件路径
- `--enroll_path`：要注册的图像目录路径
- `--category`：图像类别（默认为 "person"）
- `--collection_name`：Qdrant 集合名称
- `--threshold`：相似度阈值（默认为 0.2）
- `--device`：计算设备（默认为 CUDA，如果可用）
- `--port`：Web 服务器端口（默认为 7868）

### Web 界面使用

#### 文本搜索
1. 切换到"文本搜索"标签页
2. 在文本框中输入人物描述（如 "穿高跟鞋的女子"）
3. 调整 Top K（返回结果数量）和 Threshold（相似度阈值）参数
4. 点击 "查询" 按钮或使用快捷查询按钮
5. 查看图库中显示的搜索结果

#### 以图搜图
1. 切换到"以图搜图"标签页
2. 上传一张包含行人的图片
3. 调整 Top K 和 Threshold 参数
4. 点击 "检测并搜索" 按钮
5. 系统会自动检测图像中的行人并搜索相似的图像
6. 查看图库中显示的搜索结果，包括检测到的行人和相似图像

## API 参考

### ImageSearchAPI 类

主要方法：

- `register_new_images(image_dir, category, check_exist)`: 注册新图像到向量数据库
- `search(query_text, top_k, threshold)`: 通过文本查询搜索图像
- `search_by_image(image, top_k, threshold)`: 通过图像查询搜索相似图像
- `delete_image(img_path)`: 删除指定图像
- `count_records()`: 获取向量数据库中的记录数量

### PedestrianDetector 类

主要方法：

- `detect_image(image_path, image_pil, output_path)`: 检测图像中的行人
- `process_video(video_path, output_path)`: 处理视频中的行人检测

## 性能优化

- 支持 Flash Attention（如果可用）以加速计算
- 批量处理图像注册以提高效率
- 使用 bfloat16 精度以平衡性能和准确性
- 使用ONNX优化的检测模型提高行人检测速度

## 注意事项

- 确保图像路径可访问
- 向量数据库需要单独安装和配置
- 模型文件较大，请确保有足够的存储空间和内存
- 行人检测模型需要ONNX Runtime支持

## 示例查询

系统预设了以下快捷查询：
- "little girl"
- "a person wearing a red t-shirt"
- "an old man wearing blue hat"
- "穿高跟鞋的女子"
- "背双肩包的男孩"