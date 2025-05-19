#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <iostream>

class ONNXModelRunner {
public:
    explicit ONNXModelRunner(const std::string& model_path);
    std::vector<Ort::Value> operator()(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        const std::vector<int64_t>& token_type_ids,
        size_t batch_size,
        size_t seq_len
    );

    static void PrintTensorInfo(const Ort::Value& tensor);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    Ort::Value CreateTensor(const std::vector<int64_t>& flat_data, size_t batch_size, size_t seq_len);
    std::vector<const char*> GetNamePointers(const std::vector<std::string>& names);

    template<typename T>
    static void PrintTensorValues(const T* data_ptr, size_t len);
};
