#!/bin/bash

# 注意力机制性能测试脚本

echo "=========================================="
echo "注意力机制性能测试"
echo "=========================================="

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到CUDA编译器 (nvcc)"
    echo "请确保CUDA已正确安装并添加到PATH"
    exit 1
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build
cd build

# 配置CMake
echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "编译项目..."
make -j$(nproc)

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    echo "运行性能测试..."
    echo "=========================================="
    
    # 运行测试
    ./bench_attention
    echo ""
    echo "运行 PyTorch SDPA 基线 (flash/mem_efficient/math)..."
    echo "=========================================="
    # 与 apps/bench_attention.cu 中默认测试尺寸保持一致
    python3 ../apps/bench_sdpa.py --B 64 --H 8 --Nq 1024 --D 64 --dtype float32 --iters 50 --warmup 10 || true
    
    echo ""
    echo "=========================================="
    echo "测试完成！"
    echo "结果文件已保存到当前目录"
    echo "=========================================="
else
    echo "编译失败！"
    exit 1
fi
