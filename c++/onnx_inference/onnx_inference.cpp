#include "onnx_inference.h"

ONNXModelRunner::ONNXModelRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModel"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    size_t num_inputs = session_.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        input_names_.emplace_back(session_.GetInputName(i, Ort::AllocatorWithDefaultOptions()));
    }

    size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        output_names_.emplace_back(session_.GetOutputName(i, Ort::AllocatorWithDefaultOptions()));
    }
}

std::vector<Ort::Value> ONNXModelRunner::operator()(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    const std::vector<int64_t>& token_type_ids,
    size_t batch_size,
    size_t seq_len
) {
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(CreateTensor(input_ids, batch_size, seq_len));
    input_tensors.emplace_back(CreateTensor(token_type_ids, batch_size, seq_len));
    input_tensors.emplace_back(CreateTensor(attention_mask, batch_size, seq_len));

    std::vector<const char*> input_name_ptrs = GetNamePointers(input_names_);
    std::vector<const char*> output_name_ptrs = GetNamePointers(output_names_);

    return session_.Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_name_ptrs.data(),
        output_name_ptrs.size()
    );
}

Ort::Value ONNXModelRunner::CreateTensor(const std::vector<int64_t>& flat_data, size_t batch_size, size_t seq_len) {
    std::array<int64_t, 2> shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len)};
    return Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        const_cast<int64_t*>(flat_data.data()),
        flat_data.size(),
        shape.data(),
        shape.size()
    );
}

std::vector<const char*> ONNXModelRunner::GetNamePointers(const std::vector<std::string>& names) {
    std::vector<const char*> ptrs;
    for (const auto& name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

void ONNXModelRunner::PrintTensorInfo(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        std::cerr << "The given value is not a tensor." << std::endl;
        return;
    }

    Ort::TensorTypeAndShapeInfo shape_info = tensor.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType data_type = shape_info.GetElementType();
    std::vector<int64_t> shape = shape_info.GetShape();
    size_t total_len = shape_info.GetElementCount();

    std::cout << "Tensor data type: ";
    switch (data_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            std::cout << "FLOAT\n";
            PrintTensorValues(tensor.GetTensorData<float>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            std::cout << "UINT8\n";
            PrintTensorValues(tensor.GetTensorData<uint8_t>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            std::cout << "INT8\n";
            PrintTensorValues(tensor.GetTensorData<int8_t>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            std::cout << "INT32\n";
            PrintTensorValues(tensor.GetTensorData<int32_t>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            std::cout << "INT64\n";
            PrintTensorValues(tensor.GetTensorData<int64_t>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            std::cout << "FLOAT16\n";
            // For FLOAT16, we use a uint16_t pointer
            PrintTensorValues(reinterpret_cast<const uint16_t*>(tensor.GetTensorData<void>()), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            std::cout << "DOUBLE\n";
            PrintTensorValues(tensor.GetTensorData<double>(), total_len);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            std::cout << "BOOL\n";
            PrintTensorValues(tensor.GetTensorData<bool>(), total_len);
            break;
        default:
            std::cout << "Unknown or unsupported tensor type.\n";
            break;
    }

    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

template<typename T>
void ONNXModelRunner::PrintTensorValues(const T* data_ptr, size_t len) {
    if (!data_ptr) return;
    std::cout << "Tensor values: ";
    for (size_t i = 0; i < len; ++i) {
        std::cout << static_cast<float>(data_ptr[i]) << " ";
    }
    std::cout << std::endl;
}

// 显式模板实例化（避免链接错误）
template void ONNXModelRunner::PrintTensorValues<float>(const float*, size_t);
template void ONNXModelRunner::PrintTensorValues<uint8_t>(const uint8_t*, size_t);
template void ONNXModelRunner::PrintTensorValues<int8_t>(const int8_t*, size_t);
template void ONNXModelRunner::PrintTensorValues<int32_t>(const int32_t*, size_t);
template void ONNXModelRunner::PrintTensorValues<int64_t>(const int64_t*, size_t);
template void ONNXModelRunner::PrintTensorValues<uint16_t>(const uint16_t*, size_t);
template void ONNXModelRunner::PrintTensorValues<double>(const double*, size_t);
template void ONNXModelRunner::PrintTensorValues<bool>(const bool*, size_t);
