# c++版本nomic-text-embding推理
本项目依赖onnxruntime包，相关的分词器本来由tokenizer-cpp实现后面由纯cpp实现减少依赖

具体的使用方法在main.cpp中
采用的是models/nomic/nomic-embed-text-v1.5/onnx下的model.onnx文件
vocab也是对应的vovab.txt文件
