#include "../include/attention/attention.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <functional>

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
    double max_error;
    double mean_error;
    double relative_error;
    bool success;
};

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
        std::normal_distribution<float> dis(0.0f, 0.1f);
        
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
    
    // 计算误差
    double max_error = 0.0;
    double sum_error = 0.0;
    double sum_ref = 0.0;
    
    for (size_t i = 0; i < data.size_O; i++) {
        double error = std::abs(data.h_O[i] - data.h_O_ref[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
        sum_ref += std::abs(data.h_O_ref[i]);
    }
    
    result.max_error = max_error;
    result.mean_error = sum_error / data.size_O;
    result.relative_error = (sum_ref > 0) ? (sum_error / sum_ref) : 0.0;
    result.success = true;
    
    return result;
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
                  << std::right << std::setw(15) << std::scientific << std::setprecision(2) << result.max_error
                  << std::setw(15) << std::setprecision(2) << result.mean_error
                  << std::setw(15) << std::setprecision(2) << result.relative_error
                  << std::endl;
    } else {
        std::cout << std::left << std::setw(25) << result.name << "FAILED" << std::endl;
    }
}

// 将矩阵结果写入文件
void write_matrix_to_file(const std::string& filename, const float* matrix, int B, int H, int N, int D) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(6);
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

// 运行参考实现
void run_reference(AttentionData& data) {
    attention::attention_naive_forward(data.d_Q, data.d_K, data.d_V, data.d_O, data.dims, 0);
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
        data.initialize();
        data.copyToDevice();

        // 运行参考实现
        run_reference(data);

        // 运行所有版本的注意力
        std::vector<PerformanceResult> results;
        
        results.push_back(runPerformanceTest(
            attention::attention_naive_forward, data, num_test, "Naive Attention"));
        results.push_back(runPerformanceTest(
            attention::flash_attention_forward, data, num_test, "Flash Attention"));
        results.push_back(runPerformanceTest(
            attention::standard_attention_forward, data, num_test, "Standard Attention"));

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
                  << std::right << std::setw(15) << "Max Error"
                  << std::setw(15) << "Mean Error"
                  << std::setw(15) << "Relative Error"
                  << std::endl;
        std::cout << "----------------------------------------\n";
        
        std::vector<ErrorResult> error_results;
        error_results.push_back(runErrorTest(
            attention::attention_naive_forward, data, "Naive Attention"));
        error_results.push_back(runErrorTest(
            attention::flash_attention_forward, data, "Flash Attention"));
        error_results.push_back(runErrorTest(
            attention::standard_attention_forward, data, "Standard Attention"));
        
        for (const auto& result : error_results) {
            printErrorResult(result);
        }

        // 输出结果到文件
        data.copyToHost();
        std::string filename = "attention_result_" + 
            std::to_string(dims.B) + "x" + 
            std::to_string(dims.H) + "x" + 
            std::to_string(dims.N) + "x" + 
            std::to_string(dims.D) + ".txt";
        write_matrix_to_file(filename, data.h_O, dims.B, dims.H, dims.N, dims.D);
        std::cout << "\nResults written to: " << filename << std::endl;
    }

    return 0;
}