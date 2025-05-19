cmake_minimum_required(VERSION 3.14)
project(test_tokenizer C CXX)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 17)


set(ONNXRUNTIME_DIR ./3dpart/onnx)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/common)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/framework)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/graph)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/optimizer)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/providers/cpu)
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(test_tokenizer   main.cpp 
                                Tokenizer/Tokenizer.h 
                                Tokenizer/Tokenizer.cpp 
                                onnx_inference/onnx_inference.h
                                onnx_inference/onnx_inference.cpp
                                )

target_link_libraries(test_tokenizer 

    onnxruntime
)
