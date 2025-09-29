cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(attention_xymc CUDA CXX)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 86)  # 根据你的GPU调整

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加实验性标志以支持 constexpr 在 __host__ __device__ 函数中的使用
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# 限制每个线程使用的寄存器数量为128
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 设置调试标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

# 添加头文件目录
include_directories(include)

# 添加源文件
set(SOURCES
    src/attention.cpp
    src/kernels/naive_attention.cu
    src/kernels/flash_attention.cu
    src/kernels/standard_attention.cu
)

# 创建主程序
add_executable(bench_attention apps/bench_attention.cu ${SOURCES})

# 链接CUDA库
target_link_libraries(bench_attention cublas cudart)


