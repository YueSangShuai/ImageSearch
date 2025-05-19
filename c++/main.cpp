#include <iostream>
#include <fstream>
#include "Tokenizer/Tokenizer.h"
#include "onnx_inference/onnx_inference.h"


int main() {

    ONNXModelRunner runner("../model.onnx");
    Tokenizer tokenizer("../vocab.txt");
    std::vector<std::string> sentences = {
        "Hello"
    };

    auto [input_ids, attention_mask, token_type_ids] = tokenizer.EncodeBatch(sentences);
    auto output_tensors = runner(input_ids, attention_mask, token_type_ids, sentences.size(),  input_ids.size() / sentences.size());
    // 打印输出结果
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        std::cout << "Output " << i << ":" << std::endl;
        runner.PrintTensorInfo(output_tensors[i]);
    }

    const float* out_data = output_tensors[0].GetTensorData<float>();
    Ort::TensorTypeAndShapeInfo out_shape = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = out_shape.GetShape();
    size_t total_len = out_shape.GetElementCount();

    // 保存为 txt 文件，每个数用空格隔开
    std::ofstream outfile("output.txt");
    if (!outfile.is_open()) {
    std::cerr << "Failed to open or create output.txt" << std::endl;
    return 1;  // 或抛出异常
    }
    for (size_t i = 0; i < total_len; ++i) {
        outfile << out_data[i];
        if (i != total_len - 1) outfile << " ";
    }
    outfile << std::endl;
    outfile.close();


    return 0;
}

