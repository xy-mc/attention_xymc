cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(attention_xymc CUDA CXX)

# 默认使用 Release 构建（当未显式指定时）
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

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
    # -G 已经包含了 -lineinfo 的功能，同时使用会冲突
    # -G: 生成设备代码调试信息（包含行号信息）
    # -g: 生成主机代码调试信息
    # -O0: 禁用优化
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -arch=sm_86")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    # 确保在Debug模式下禁用优化和内联
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-O0")
    # 保留源代码信息以便调试时行号对应（在PTX中嵌入源代码）
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --source-in-ptx")
endif()

# 添加头文件目录
include_directories(include)
include_directories(../cutlass/include ../cutlass/tools/util/include)
# 添加源文件
set(SOURCES
    src/attention.cpp
    src/kernels/naive_attention.cu
    src/kernels/flash_attention_v1.cu
    src/kernels/flash_attention_v1_optimize.cu
    src/kernels/flash_attention_v2.cu
    src/kernels/flash_attention_v2_optimize.cu
    src/kernels/flash_attention_mma.cu
    src/kernels/flash_attention_mma_optimize.cu
    src/kernels/flash_attention_mma_Kstage.cu
    src/kernels/flash_attention_mma_cutlass.cu
    src/kernels/flash_attention_target.cu
    src/kernels/standard_attention.cu
)

# 创建主程序
add_executable(bench_attention apps/bench_attention.cu ${SOURCES})

# 链接CUDA库
target_link_libraries(bench_attention cublas cudart)


# 测试程序：test_swizzle
# add_executable(test_swizzle apps/test_swizzle.cu)
# target_link_libraries(test_swizzle cudart)

# 仅主机端测试：test_swizzle_host（打印 64x64，x 步长 4）
add_executable(test_swizzle_host apps/test_swizzle_host.cpp)


