# CUDA 调试指南

## 问题：无法进入 CUDA 内核调试

### 已完成的修复

1. **构建配置已更新为 Debug 模式**
   - `.vscode/settings.json` 中的 `cmake.buildType` 已设置为 `Debug`
   - CMakeLists.txt 中 Debug 模式使用 `-G -g -O0 -lineinfo` 标志

2. **调试器配置已优化**
   - `launch.json` 中添加了 CUDA 调试器设置
   - 创建了 `tasks.json` 用于构建任务
   - 创建了 `nsight_visualizer.json` 配置文件

### 使用步骤

#### 1. 重新编译项目（重要！）

由于之前是 Release 模式编译，**必须重新编译**才能生成调试符号：

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
```

或者使用 VS Code 的构建任务：
- 按 `Ctrl+Shift+P` (或 `Cmd+Shift+P` on Mac)
- 输入 "Tasks: Run Build Task"
- 选择 "rebuild" 任务

#### 2. 设置断点

在 CUDA 内核函数中设置断点，例如在 `flash_attention_mma_cutlass_forward_kernel` 函数中：

```cpp
__global__ void flash_attention_mma_cutlass_forward_kernel(...) {
    // 在这里设置断点
    const int i = blockIdx.x;
    // ...
}
```

#### 3. 启动调试

- 按 `F5` 或点击调试按钮
- 选择 "CUDA Nsight Debug" 配置

#### 4. 调试技巧

**如果仍然无法进入内核：**

1. **检查编译标志**
   ```bash
   cd build
   cat CMakeCache.txt | grep CMAKE_CUDA_FLAGS
   ```
   应该看到 `-G -g -O0` 标志

2. **手动验证调试符号**
   ```bash
   cuda-gdb build/bench_attention
   (cuda-gdb) info functions flash_attention
   ```
   应该能看到内核函数列表

3. **在 cuda-gdb 中手动设置断点**
   ```bash
   (cuda-gdb) break flash_attention_mma_cutlass_forward_kernel
   (cuda-gdb) run
   ```

4. **检查 GPU 计算能力**
   - 确保 CMakeLists.txt 中的 `CMAKE_CUDA_ARCHITECTURES` 与你的 GPU 匹配
   - 当前设置为 `86` (对应 A100/H100)
   - 如果是其他 GPU，需要修改：
     - RTX 3090/4090: `86`
     - RTX 3080/4080: `86`
     - V100: `70`
     - T4: `75`

5. **使用 CUDA 调试命令**
   在调试控制台中，可以使用：
   ```
   (cuda-gdb) set cuda break_on_launch application
   (cuda-gdb) cuda kernel
   (cuda-gdb) info cuda kernels
   ```

### 常见问题

**Q: 编译后仍然无法进入内核？**
A: 确保完全清理并重新编译：
```bash
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
```

**Q: 调试器启动但立即退出？**
A: 检查程序是否有参数要求，可能需要修改 `launch.json` 中的 `args` 字段

**Q: 能看到主机代码但看不到设备代码？**
A: 确保使用了 `-G` 标志（生成设备代码调试信息）

**Q: CUTLASS 内核调试时行号不对应？**
A: 这是 CUTLASS 模板元编程的常见问题。已添加以下修复：
- 添加了 `--source-in-ptx` 标志，在 PTX 中嵌入源代码信息
- 添加了 `__noinline__` 属性到 cutlass 内核函数，防止过度内联
- 使用 `-O0` 和 `--ptxas-options=-O0` 禁用所有优化

**如果行号仍然不对应，可以尝试：**

1. **启用中间文件保留**（在 CMakeLists.txt 中取消注释）：
   ```cmake
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep --keep-dir=${CMAKE_BINARY_DIR}/cuda_intermediate")
   ```
   这会保留所有中间编译文件，有助于调试模板实例化

2. **使用 cuda-gdb 的行号映射命令**：
   ```
   (cuda-gdb) info line *0x<address>
   (cuda-gdb) list *0x<address>
   ```

3. **在关键位置添加显式断点**：
   由于模板代码可能被展开，可以在函数开始处和关键逻辑处设置多个断点：
   ```cpp
   __global__ __noinline__ void flash_attention_mma_cutlass_forward_kernel(...) {
       // 断点1：函数入口
       const int i = blockIdx.x;
       // 断点2：关键逻辑
       // ...
   }
   ```

4. **使用 printf 调试**：
   在关键位置添加 `printf` 来验证执行流程，这在模板代码中更可靠：
   ```cpp
   if (threadIdx.x == 0 && blockIdx.x == 0) {
       printf("Debug: Line %d executed\n", __LINE__);
   }
   ```

### 验证调试配置

运行以下命令验证配置：

```bash
# 检查可执行文件是否包含调试符号
file build/bench_attention
objdump -h build/bench_attention | grep debug

# 检查 CUDA 调试器是否可用
/usr/local/cuda/bin/cuda-gdb --version
```

### 参考

- [CUDA-GDB 用户指南](https://docs.nvidia.com/cuda/cuda-gdb/)
- [Nsight Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code)

