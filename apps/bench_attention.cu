#include "../include/attention/attention.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <functional>
#include <cmath> // Added for std::abs
#include <string>

/*
smem 是max(smem_qk,smem_v)
中间计算结果可以放在寄存器里面
两次矩阵乘法  
1.Br Bc D
2.Br D Bc

常见 B H N D尺寸
B 1-64 1 2 4 8 16 32 64
H 12 16 32 64 96
N 128 256 512 1024 2048 4096
D 64 128
*/

#define num_test 1

// 性能测试结果结构
struct PerformanceResult {
    std::string name;
    double avg_time_ms;
    double gflops;
    double bandwidth_gb_s;
    bool success;
};

// 误差测试结果结构
struct ErrorResult {
    std::string name;
    double max_error;            // 绝对误差最大值（仅作参考）
    double mean_error;           // 绝对误差均值（仅作参考）
    double relative_error;       // L1 相对误差（|Δ|/|ref| 的平均，受近零影响）
    // 更适合注意力评估的指标：
    double rel_l2;              // 全局相对 L2：||O-Oref||2 / ||Oref||2（整体幅值偏差）
    double max_rel;             // 元素级最大相对误差：max |Δ| / (|ref|+eps)（易受近零 ref 影响）
    double cosine_sim;          // 全局余弦相似度：展平后的方向一致性（越接近 1 越好）
    double row_rel_l2_mean;     // 行级相对 L2 的平均：逐 (b,h,n) 向量评估后再平均
    double row_rel_l2_max;      // 行级相对 L2 的最大值：最差一行的相对偏差
    double row_cos_mean;        // 行级余弦相似度均值：平均方向一致性
    double row_cos_min;         // 行级余弦相似度最小值：最差一行的方向一致性
    bool success;
};

/*
合理误差参考（FP32，经验值）：
Rel L2: ≤1e-3 优秀；≤1e-2 可接受；>1e-2 需检查
Cosine: ≥9.999e-01 优秀；≥9.99e-01 可接受；<9.9e-01 需检查
RowRelL2Max: ≤2e-2 可接受；更大说明存在行级严重偏差
RowCosMin: ≥9.9e-01 可接受；过低说明有行方向明显错误
Max Rel: 易被近零参考值放大，仅用于异常排查参考
*/

// 注意力数据管理类
class AttentionData {
public:
    attention::AttentionDims dims;
    size_t size_Q, size_K, size_V, size_O;
    float *h_Q, *h_K, *h_V, *h_O, *h_O_ref;
    float *d_Q, *d_K, *d_V, *d_O;
    
    AttentionData(const attention::AttentionDims& dims) : dims(dims) {
        // 计算数据大小 (BHND格式)
        size_Q = dims.B * dims.H * dims.N * dims.D;
        size_K = dims.B * dims.H * dims.N * dims.D;
        size_V = dims.B * dims.H * dims.N * dims.D;
        size_O = dims.B * dims.H * dims.N * dims.D;
        
        // 分配主机内存
        h_Q = new float[size_Q];
        h_K = new float[size_K];
        h_V = new float[size_V];
        h_O = new float[size_O];
        h_O_ref = new float[size_O];
        
        // 分配设备内存
        cudaMalloc(&d_Q, size_Q * sizeof(float));
        cudaMalloc(&d_K, size_K * sizeof(float));
        cudaMalloc(&d_V, size_V * sizeof(float));
        cudaMalloc(&d_O, size_O * sizeof(float));
    }
    
    ~AttentionData() {
        delete[] h_Q;
        delete[] h_K;
        delete[] h_V;
        delete[] h_O;
        delete[] h_O_ref;
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
    }
    
    void initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t i = 0; i < size_Q; i++) h_Q[i] = dis(gen);
        for (size_t i = 0; i < size_K; i++) h_K[i] = dis(gen);
        for (size_t i = 0; i < size_V; i++) h_V[i] = dis(gen);
    }

    // 生成 N(mean, std^2) 的 Q/K/V（可复现）
    void initialize_normal(float mean = 0.0f, float std = 1.0f, uint64_t seed = 42) {
        std::mt19937 gen(static_cast<uint32_t>(seed));
        std::normal_distribution<float> dis(mean, std);
        for (size_t i = 0; i < size_Q; i++) h_Q[i] = dis(gen);
        for (size_t i = 0; i < size_K; i++) h_K[i] = dis(gen);
        for (size_t i = 0; i < size_V; i++) h_V[i] = dis(gen);
    }
    
    void copyToDevice() {
        cudaMemcpy(d_Q, h_Q, size_Q * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, size_K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, size_V * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void copyToHost() {
        cudaMemcpy(h_O, d_O, size_O * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    void copyRefToHost() {
        cudaMemcpy(h_O_ref, d_O, size_O * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

// 计算注意力机制的FLOPs
double compute_attention_flops(const attention::AttentionDims& dims) {
    int B = dims.B;
    int H = dims.H;
    int N = dims.N;
    int D = dims.D;
    
    // QK^T: B*H*N*N*D
    double qk_flops = (double)B * H * N * N * D;
    // Softmax: B*H*N*N (exp + sum)
    double softmax_flops = (double)B * H * N * N * 2;
    // Attention*V: B*H*N*N*D
    double av_flops = (double)B * H * N * N * D;
    
    return qk_flops + softmax_flops + av_flops;
}

// 计算内存带宽
double compute_attention_bandwidth(const attention::AttentionDims& dims) {
    int B = dims.B;
    int H = dims.H;
    int N = dims.N;
    int D = dims.D;
    
    // 读取: Q, K, V
    double read_bytes = (double)(B * H * N * D + B * H * N * D + B * H * N * D) * sizeof(float);
    // 写入: O
    double write_bytes = (double)(B * H * N * D) * sizeof(float);
    
    return read_bytes + write_bytes;
}

// 运行性能测试
PerformanceResult runPerformanceTest(
    std::function<void(const float*, const float*, const float*, float*, const attention::AttentionDims&, cudaStream_t)> func,
    AttentionData& data,
    int num_runs,
    const std::string& name) {
    
    PerformanceResult result;
    result.name = name;
    result.success = false;
    
    // 预热
    func(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
    cudaDeviceSynchronize();
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        func(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.avg_time_ms = total_time_ms / num_runs;
    
    // 计算性能指标
    double flops = compute_attention_flops(data.dims);
    double bandwidth = compute_attention_bandwidth(data.dims);
    
    result.gflops = (flops / 1e9) / (result.avg_time_ms / 1000.0);
    result.bandwidth_gb_s = (bandwidth / 1e9) / (result.avg_time_ms / 1000.0);
    result.success = true;
    
    return result;
}

// 运行误差测试
ErrorResult runErrorTest(
    std::function<void(const float*, const float*, const float*, float*, const attention::AttentionDims&, cudaStream_t)> func,
    AttentionData& data,
    const std::string& name) {
    
    ErrorResult result;
    result.name = name;
    result.success = false;
    
    // 运行测试函数
    func(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
    cudaDeviceSynchronize();
    
    // 复制结果到主机
    data.copyToHost();
    
    // 计算误差（全局与行级）
    const double eps = 1e-6; // stability for near-zero denominators
    double max_error = 0.0;
    double sum_error = 0.0;
    double sum_ref = 0.0;

    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    double max_rel = 0.0;

    double global_dot = 0.0;
    double global_norm_o_sq = 0.0;
    double global_norm_ref_sq = 0.0;

    // 行级指标（对每个 (b,h,n) 的长度为 D 的向量）
    const int B = data.dims.B;
    const int H = data.dims.H;
    const int N = data.dims.N;
    const int D = data.dims.D;
    const size_t row_stride = static_cast<size_t>(D);
    const size_t rows = static_cast<size_t>(B) * H * N;

    double row_rel_l2_sum = 0.0;
    double row_rel_l2_max = 0.0;
    double row_cos_sum = 0.0;
    double row_cos_min = 1.0; // cosine ∈ [-1,1]

    // 全局逐元素聚合
    for (size_t i = 0; i < data.size_O; i++) {
        const double o = data.h_O[i];
        const double r = data.h_O_ref[i];
        const double diff = o - r;
        const double abs_diff = std::abs(diff);
        max_error = std::max(max_error, abs_diff);
        sum_error += abs_diff;
        sum_ref += std::abs(r);

        sum_sq_diff += diff * diff;
        sum_sq_ref += r * r;
        const double rel = abs_diff / std::max(std::abs(r), eps);
        if (rel > max_rel) max_rel = rel;

        global_dot += o * r;
        global_norm_o_sq += o * o;
        global_norm_ref_sq += r * r;
    }

    // 行级循环
    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * row_stride;
        double row_diff_sq = 0.0;
        double row_ref_sq = 0.0;
        double row_dot = 0.0;
        double row_o_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            const double o = data.h_O[base + d];
            const double r = data.h_O_ref[base + d];
            const double diff = o - r;
            row_diff_sq += diff * diff;
            row_ref_sq += r * r;
            row_dot += o * r;
            row_o_sq += o * o;
        }
        const double tau_row = 1e-2 * std::sqrt(static_cast<double>(D));
        const double row_rel_l2 = std::sqrt(row_diff_sq) / std::max(std::sqrt(row_ref_sq), tau_row);
        row_rel_l2_sum += row_rel_l2;
        if (row_rel_l2 > row_rel_l2_max) row_rel_l2_max = row_rel_l2;

        const double row_cos = row_dot / (std::sqrt(row_o_sq) * std::sqrt(row_ref_sq) + eps);
        row_cos_sum += row_cos;
        if (row_cos < row_cos_min) row_cos_min = row_cos;
    }

    // 写入结果
    result.max_error = max_error;
    result.mean_error = sum_error / data.size_O;
    result.relative_error = (sum_ref > 0) ? (sum_error / sum_ref) : 0.0;
    result.rel_l2 = std::sqrt(sum_sq_diff) / std::max(std::sqrt(sum_sq_ref), eps);
    result.max_rel = max_rel;
    result.cosine_sim = global_dot / (std::sqrt(global_norm_o_sq) * std::sqrt(global_norm_ref_sq) + eps);
    result.row_rel_l2_mean = row_rel_l2_sum / static_cast<double>(rows);
    result.row_rel_l2_max = row_rel_l2_max;
    result.row_cos_mean = row_cos_sum / static_cast<double>(rows);
    result.row_cos_min = row_cos_min;
    result.success = true;
    
    return result;
}

// 将矩阵结果写入文件
void write_matrix_to_file(const std::string& filename, const float* matrix, int B, int H, int N, int D) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(30);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            outfile << "Batch " << b << ", Head " << h << ":\n";
            for (int n = 0; n < N; n++) {
                for (int d = 0; d < D; d++) {
                    int idx = ((b * H + h) * N + n) * D + d;
                    outfile << matrix[idx] << " ";
                }
                outfile << "\n";
            }
            outfile << "\n";
        }
    }
    outfile.close();
}

// 将矩阵的一个 (b,h) 在 N 维从 n0 开始、大小为 Br 的切片，以 Br×D 形式写入
void write_matrix_tile_BrD(const std::string& filename,
                           const float* matrix,
                           int B, int H, int N, int D,
                           int b, int h, int n0, int Br) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    const int n_end = std::min(n0 + Br, N);
    outfile << std::fixed << std::setprecision(30);
    outfile << "Slice (b=" << b << ", h=" << h << ", n in [" << n0 << ", " << (n_end - 1) << "]) as BrxD\n";
    for (int n = n0; n < n_end; ++n) {
        for (int d = 0; d < D; ++d) {
            const int idx = ((b * H + h) * N + n) * D + d; // BHND
            outfile << matrix[idx] << (d + 1 == D ? '\n' : ' ');
        }
    }
    outfile.close();
}

// 运行一个实现并把输出写入文件
using RunFunc = std::function<void(const float*, const float*, const float*, float*, const attention::AttentionDims&, cudaStream_t)>;

void run_and_write(const RunFunc& func, AttentionData& data, const std::string& filename) {
    func(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
    cudaDeviceSynchronize();
    data.copyToHost();
    // write_matrix_to_file(filename, data.h_O, data.dims.B, data.dims.H, data.dims.N, data.dims.D);
    write_matrix_tile_BrD(filename, data.h_O, data.dims.B, data.dims.H, data.dims.N, data.dims.D, 0, 0, 0, 64);
    std::cout << "Results written to: " << filename << std::endl;
}

// 批量输出指定实现的结果
void dump_selected_results(AttentionData& data,
                           const std::vector<std::pair<RunFunc, std::string>>& items,
                           const std::string& base_filename) {
    for (const auto& it : items) {
        const auto& fn = it.first;
        const auto& tag = it.second; // 例如 "v2_optimize" / "mma"
        std::string filename = base_filename + "_" + tag + ".txt";
        run_and_write(fn, data, filename);
    }
}

// 打印性能结果
void printPerformanceResult(const PerformanceResult& result) {
    if (result.success) {
        std::cout << std::left << std::setw(25) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_time_ms << " ms"
                  << std::setw(15) << std::setprecision(2) << result.gflops << " GFLOPS"
                  << std::setw(15) << std::setprecision(2) << result.bandwidth_gb_s << " GB/s"
                  << std::endl;
    } else {
        std::cout << std::left << std::setw(25) << result.name << "FAILED" << std::endl;
    }
}

// 打印误差结果
void printErrorResult(const ErrorResult& result) {
    if (result.success) {
        std::cout << std::left << std::setw(25) << result.name
                  << std::right << std::setw(14) << std::scientific << std::setprecision(2) << result.rel_l2
                  << std::setw(14) << std::setprecision(2) << result.max_rel
                  << std::setw(14) << std::setprecision(2) << result.cosine_sim
                  << std::setw(14) << std::setprecision(2) << result.row_rel_l2_max
                  << std::setw(14) << std::setprecision(2) << result.row_cos_min
                  << std::endl;
    } else {
        std::cout << std::left << std::setw(25) << result.name << "FAILED" << std::endl;
    }
}

// 运行参考实现
void run_reference(AttentionData& data) {
    attention::standard_attention_forward(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
    cudaDeviceSynchronize();
    data.copyRefToHost();
}

int main() {
    // 设置注意力维度 (BHND格式)
    std::vector<attention::AttentionDims> test_cases = {
        // {1, 8, 128, 64},      // 小规模
        // {1, 8, 512, 64},      // 中等规模
        {1, 8, 1024, 64},     // 大规模
        // {1, 8, 2048, 64},     // 超大规模
        // {2, 8, 1024, 64},     // 多批次
        // {1, 16, 1024, 64},    // 多头
    };

    // 测试每个维度
    for (const auto& dims : test_cases) {
        std::cout << "\nTesting attention dimensions: "
                  << "B=" << dims.B
                  << ", H=" << dims.H
                  << ", N=" << dims.N
                  << ", D=" << dims.D << "\n"
                  << "========================================\n";

        // 创建并初始化数据
        AttentionData data(dims);
        // 生成可复现的 N(0,1) 测试数据
        data.initialize_normal(0.0f, 1.0f, /*seed=*/1234);
        // data.initialize();
        data.copyToDevice();

        // 运行参考实现
        run_reference(data);

        // 运行所有版本的注意力
        std::vector<PerformanceResult> results;
        
        // results.push_back(runPerformanceTest(
        //     attention::attention_naive_forward, data, num_test, "Naive Attention"));
        // results.push_back(runPerformanceTest(
        //     attention::standard_attention_forward, data, num_test, "Standard Attention"));
        // results.push_back(runPerformanceTest(
        //     attention::flash_attention_v1_forward, data, num_test, "Flash Attention_v1"));
        // results.push_back(runPerformanceTest(
        //     attention::flash_attention_v2_forward, data, num_test, "Flash Attention_v2"));
        // results.push_back(runPerformanceTest(
        //     attention::flash_attention_v1_optimize_forward, data, num_test, "Flash Attention_v1_optimize"));
        // results.push_back(runPerformanceTest(
        //     attention::flash_attention_v2_optimize_forward, data, num_test, "Flash Attention_v2_optimize"));
        results.push_back(runPerformanceTest(
            attention::flash_attention_mma_forward, data, num_test, "Flash Attention_mma"));
        // results.push_back(runPerformanceTest(
        //     attention::flash_attention_target_forward, data, num_test, "Flash Attention_target_half"));

        // 打印性能结果
        std::cout << "\nPerformance Results:\n"
                  << "========================================\n";
        std::cout << std::left << std::setw(25) << "Implementation"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "GFLOPS"
                  << std::setw(15) << "Bandwidth (GB/s)"
                  << std::endl;
        std::cout << "----------------------------------------\n";
        
        for (const auto& result : results) {
            printPerformanceResult(result);
        }

        // 进行误差测试
        std::cout << "\nError Analysis (compared with reference):\n"
                  << "========================================\n";
        std::cout << std::left << std::setw(25) << "Implementation"
                  << std::right << std::setw(14) << "Rel L2"
                  << std::setw(14) << "Max Rel"
                  << std::setw(14) << "Cosine"
                  << std::setw(14) << "RowRelL2Max"
                  << std::setw(14) << "RowCosMin"
                  << std::endl;
        std::cout << "----------------------------------------\n";
        
        std::vector<ErrorResult> error_results;
        // error_results.push_back(runErrorTest(
        //     attention::attention_naive_forward, data, "Naive Attention"));
        // error_results.push_back(runErrorTest(
        //     attention::standard_attention_forward, data, "Standard Attention"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_v1_forward, data, "Flash Attention_v1"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_v2_forward, data, "Flash Attention_v2"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_v1_optimize_forward, data, "Flash Attention_v1_optimize"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_v2_optimize_forward, data, "Flash Attention_v2_optimize"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_mma_forward, data, "Flash Attention_mma"));
        // error_results.push_back(runErrorTest(
        //     attention::flash_attention_target_forward, data, "Flash Attention_target_half"));

        for (const auto& result : error_results) {
            printErrorResult(result);
        }

        // 输出结果到文件（选择要输出的实现）
        // std::string base_filename = std::string("attention_result_") +
        //     std::to_string(dims.B) + "x" +
        //     std::to_string(dims.H) + "x" +
        //     std::to_string(dims.N) + "x" +
        //     std::to_string(dims.D);

        // // 参考结果
        // std::string ref_filename = base_filename + "_reference.txt";
        // // write_matrix_to_file(ref_filename, data.h_O_ref, dims.B, dims.H, dims.N, dims.D);
        // write_matrix_tile_BrD(ref_filename, data.h_O_ref, dims.B, dims.H, dims.N, dims.D, 0, 0, 0, 64);
        // std::cout << "\nReference results written to: " << ref_filename << std::endl;

        // // 根据需要在此列表中添加/移除要输出的实现
        // dump_selected_results(
        //     data,
        //     {
        //         {attention::flash_attention_v2_forward, "v2"},
        //         {attention::flash_attention_v2_optimize_forward, "v2_optimize"},
        //         {attention::flash_attention_mma_forward, "mma"}
        //     },
        //     base_filename);
    }

    return 0;
}