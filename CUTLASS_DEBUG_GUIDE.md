# CUTLASS 内核调试指南

## 问题：无法进入 CUTLASS 内核调试

### 原因分析

CUTLASS 内核使用函数指针和复杂的模板实例化，导致：
1. 内核函数名在编译后被混淆
2. 调试器无法自动识别内核函数
3. 需要手动设置断点或使用特殊方法

### 解决方案

#### 方法 1：在内核启动处设置断点（推荐）

在内核启动的代码行设置断点（第 472 行）：

```cpp
// src/kernels/flash_attention_mma_cutlass.cu:472
kernel_fptr<<<grid, block, smem_size, stream>>>(...);
```

**步骤：**
1. 在 VS Code 中打开 `src/kernels/flash_attention_mma_cutlass.cu`
2. 在第 472 行（`kernel_fptr<<<...>>>`）设置断点
3. 启动调试（F5）
4. 当程序停在断点时，使用 `step` 命令进入内核

#### 方法 2：在内核函数入口设置断点

在内核函数的第一行设置断点（第 84 行）：

```cpp
// src/kernels/flash_attention_mma_cutlass.cu:84
const int i = blockIdx.x;
```

**步骤：**
1. 在 VS Code 中打开 `src/kernels/flash_attention_mma_cutlass.cu`
2. 在第 84 行设置断点
3. 启动调试（F5）
4. 程序会在内核执行时停在这里

#### 方法 3：使用 cuda-gdb 命令手动设置断点

如果自动断点不工作，可以在调试控制台中使用命令：

```bash
# 1. 启动调试后，在调试控制台输入：

# 列出所有 CUDA 内核函数
(cuda-gdb) info cuda kernels

# 使用正则表达式在所有匹配的内核上设置断点
(cuda-gdb) rbreak flash_attention_mma_cutlass_forward_kernel

# 或者使用函数名模式
(cuda-gdb) rbreak .*cutlass.*forward.*kernel

# 继续执行
(cuda-gdb) continue

# 当内核启动时，使用以下命令切换到设备代码
(cuda-gdb) cuda kernel
(cuda-gdb) cuda thread
```

#### 方法 4：使用 `cuda break_on_launch`（已在配置中启用）

`launch.json` 中已设置：
```
set cuda break_on_launch application
```

这会在所有内核启动时自动暂停。然后：

```bash
# 查看当前内核
(cuda-gdb) info cuda kernels

# 切换到内核
(cuda-gdb) cuda kernel <kernel_id>

# 切换到线程
(cuda-gdb) cuda thread (0,0,0)

# 设置断点
(cuda-gdb) break flash_attention_mma_cutlass_forward_kernel
```

#### 方法 5：使用 printf 调试（最可靠）

如果断点仍然不工作，可以在内核中添加 printf：

```cpp
__global__ __noinline__ void flash_attention_mma_cutlass_forward_kernel(...) {
    // 在内核开始处添加
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("Kernel entered! Line %d\n", __LINE__);
    }
    
    const int i = blockIdx.x;
    // ...
}
```

### 验证调试配置

1. **检查编译标志**：
   ```bash
   cd build
   cat CMakeCache.txt | grep CMAKE_CUDA_FLAGS
   ```
   应该看到：`-G -g -O0 -lineinfo`

2. **检查内核符号**：
   ```bash
   cuda-gdb build/bench_attention
   (cuda-gdb) info functions flash_attention
   ```
   应该能看到内核函数列表

3. **列出所有 CUDA 内核**：
   ```bash
   (cuda-gdb) run
   # 程序运行后
   (cuda-gdb) info cuda kernels
   ```

### 常见问题

**Q: 断点设置了但程序不停止？**
A: 
- 确保使用 Debug 模式编译
- 检查断点是否设置在内核实际执行的代码路径上
- 尝试在内核启动处（第 472 行）设置断点

**Q: 能看到主机代码但无法进入设备代码？**
A:
- 使用 `cuda kernel` 命令切换到设备上下文
- 使用 `cuda thread (0,0,0)` 切换到第一个线程
- 然后使用 `step` 或 `next` 命令

**Q: 内核函数名找不到？**
A:
- 使用 `rbreak` 命令和正则表达式：`rbreak .*cutlass.*`
- 或者使用 `info cuda kernels` 查看实际的内核名称
- 在内核启动处（第 472 行）设置断点，然后单步进入

### 推荐的调试流程

1. **在 VS Code 中设置断点**：
   - 第 472 行：`kernel_fptr<<<...>>>`
   - 第 84 行：`const int i = blockIdx.x;`

2. **启动调试**（F5）

3. **当停在第 472 行时**：
   - 在调试控制台输入：`step` 或 `s`
   - 这会进入内核函数

4. **如果无法进入**：
   - 在调试控制台输入：`cuda kernel`
   - 然后输入：`cuda thread (0,0,0)`
   - 再输入：`break flash_attention_mma_cutlass_forward_kernel`
   - 最后输入：`continue`

### 参考

- [CUDA-GDB 用户指南](https://docs.nvidia.com/cuda/cuda-gdb/)
- [Nsight Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code)


