#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <algorithm>  // for std::swap
#include "../include/attention/PTX.h"
#include "../include/attention/utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cute/tensor.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#ifndef DEBUG
#define DEBUG 0
#endif

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

// static __host__ __device__ __forceinline__
// int swizzle_QK(const int y, const int x) {
//     // x >>= 2;
//     // return ((y & 7) ^ x) << 2;
//     return ((y & 7) ^ (x >> 2)) << 2 | (x & 3);
//     // return x;
// }

// static __host__ __device__ __forceinline__
// int swizzle_V(const int y, const int x) {
//     return (((y & 7) >> 1) ^ (x >> 3)) << 3 | (x & 7);
//     // return x;
// }

__device__ __forceinline__ void spin_cycles(unsigned long long cycles) {
    unsigned long long s = clock64();
    while (clock64() - s < cycles) { /* busy wait */ }
  }

template<
    const int NumThreads,
    const int KMmaAtomM,  // 16
    const int KMmaAtomN,  // 8
    const int KMmaAtomK,  // 8
    const int kMmaTileSeqLenQ,  // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK,  // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenP,  // 4, more MMA(warp), M=16*4=64, P@V
                                // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kMmaTileHeadDimV, // 1, more MMA(warp), N=8*1 =8,  P@V
                                // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kWarpTileSeqLenQ, // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK, // 8, more values, N, Bc=8*8 =64, matmul N
    const int kWarpTileSeqLenP, // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileHeadDimV, // 8, more values, N,
                                // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
    const int Kstage,
    const int D,
    const int Br,
    const int Bc,
    class Tiledcopy_Q,
    class Tiledcopy_KV,
    class TiledMMA_QK,
    class TiledMMA_PV,
    class S2RAtomQ,
    class S2RAtomKV, 
    class R2GAtomO,
    class T_QKV
>
__global__ __noinline__ void flash_attention_mma_cutlass_forward_kernel(
    const float* __restrict__ DQ,
    const float* __restrict__ DK,
    const float* __restrict__ DV,
    float* __restrict__ DO,   // restrict 表示通过该指针访问的内存区域不会被其他指针别名（alias）访问
    const int B, const int H, const int N,
    const float scale, 
    const int Tr, const int Tc,
    Tiledcopy_Q global_smem_q, Tiledcopy_KV global_smem_kv, TiledMMA_QK mma_qk, TiledMMA_PV mma_pv, 
    S2RAtomQ s2r_atom_Q, S2RAtomKV s2r_atom_KV, R2GAtomO r2g_atom_O) {

    // DEBUG BREAKPOINT: Set breakpoint here to enter the kernel
    // This is the first executable line in the kernel
    
    using namespace cute;
    
    const int i = blockIdx.x;
    const int b = blockIdx.y;
    const int h = blockIdx.z;
    const int tx = threadIdx.x;
    
    auto Q = make_tensor(make_gmem_ptr(DQ), make_layout(make_shape(B, H, N, D), GenRowMajor{}));

    auto K = make_tensor(make_gmem_ptr(DK), make_layout(make_shape(B, H, N, D), GenRowMajor{}));

    auto V = make_tensor(make_gmem_ptr(DV), make_layout(make_shape(B, H, N, D), GenRowMajor{}));

    auto O = make_tensor(make_gmem_ptr(DO), make_layout(make_shape(B, H, N, D), GenRowMajor{}));
    
    auto gQ = local_tile(Q, make_shape(_1{}, _1{}, Br, D), make_coord(b, h, i, 0))(0, 0, _, _); // (Br, D)
    
    auto gK = local_tile(K, make_shape(_1{}, _1{}, Bc, D), make_coord(b, h, _, 0))(0, 0, _, _, _); // (Bc, D, RestKV)

    auto gV = local_tile(V, make_shape(_1{}, _1{}, Bc, D), make_coord(b, h, _, 0))(0, 0, _, _, _); // (Bc, D, RestKV)

    auto gO = local_tile(O, make_shape(_1{}, _1{}, Br, D), make_coord(b, h, i, 0))(0, 0, _, _); // (Br, D)

    extern __shared__ __align__(16) T_QKV s_data[];
    T_QKV *smem_q = s_data;
    T_QKV *smem_k = smem_q + Br * D;
    T_QKV *smem_v = smem_k + Bc * D * Kstage;
    
    auto sQ = make_tensor(make_smem_ptr(smem_q), make_layout(Shape<Int<Br>, Int<D>>{}, GenRowMajor{}));
    auto sK = make_tensor(make_smem_ptr(smem_k), make_layout(Shape<Int<Bc>, Int<D>, Int<Kstage>>{}, GenRowMajor{}));// 必须使用静态类型
    auto sV = make_tensor(make_smem_ptr(smem_v), make_layout(Shape<Int<Bc>, Int<D>>{}, GenRowMajor{}));
    auto sV_T = make_tensor(make_smem_ptr(smem_v), make_layout(Shape<Int<D>, Int<Bc>>{}, GenColMajor{}));

    auto thr_copy_q = global_smem_q.get_slice(tx); // 
    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tQsQ = thr_copy_q.partition_D(sQ);
    
    #if DEBUG
    if (thread0()) {
        print("thr_copy_q: "); print(thr_copy_q); printf("\n");
        print("tQgQ: "); print(tQgQ); printf("\n");
        print("tQsQ: "); print(tQsQ); printf("\n");
    }
    #endif

    for (int k = 0; k < size<1>(tQgQ); k++) {
        copy(global_smem_q, tQgQ(_, k, _), tQsQ(_, k, _));
    }
    // copy(thr_copy_q, tQgQ, tQsQ);
    cp_async_fence();

    auto thr_copy_kv = global_smem_kv.get_slice(tx);
    auto tKgK = thr_copy_kv.partition_S(gK); // (CPY, CPY_Br, CPY_D, k)
    auto tKsK = thr_copy_kv.partition_D(sK); // (CPY, CPY_Bc, CPY_D, Kstage)
    auto tVgV = thr_copy_kv.partition_S(gV); // (CPY, CPY_Bc, CPY_D, k)
    auto tVsV = thr_copy_kv.partition_D(sV); // (CPY, CPY_Bc, CPY_D)
    
    if constexpr (Kstage > 1) {
        for (int stage = 0; stage < (Kstage - 1); stage++) {
            for (int k = 0; k < size<1>(tKgK); k++) {
                copy(global_smem_kv, tKgK(_, k, _, stage), tKsK(_, k, _, stage));
            }
            cp_async_fence();
        }
    }

    if constexpr (Kstage > 1) {
        cp_async_wait<Kstage - 2>();
        __syncthreads();
    }

    auto thr_mma_qk = mma_qk.get_slice(tx);
    auto tSrQ = thr_mma_qk.partition_fragment_A(sQ); // (MMA, MMA_Q, MMA_D)
    auto tSrK = thr_mma_qk.partition_fragment_B(sK(_, _, 0)); // (MMA, MMA_K, MMA_D)
    auto tSrS = partition_fragment_C(mma_qk, Shape<Int<Br>, Int<Bc>>{}); // (MMA, MMA_Q, MMA_K)

    TiledCopy s2r_copy_q = make_tiled_copy_A(s2r_atom_Q, mma_qk);
    ThrCopy s2r_thr_copy_q = s2r_copy_q.get_slice(tx);
    Tensor tXsQ = s2r_thr_copy_q.partition_S(sQ); // (CPY, MMA_Q, MMA_D)
    Tensor tXrQ = s2r_thr_copy_q.retile_D(tSrQ);  // (CPY, MMA_Q, MMA_D)

    TiledCopy s2r_copy_k = make_tiled_copy_B(s2r_atom_KV, mma_qk);
    ThrCopy s2r_thr_copy_k = s2r_copy_k.get_slice(tx);
    Tensor tXsK = s2r_thr_copy_k.partition_S(sK); // (CPY, MMA_K, MMA_D, K_stage)
    Tensor tXrK = s2r_thr_copy_k.retile_D(tSrK);  // (CPY, MMA_K, MMA_D)

    auto thr_mma_pv = mma_pv.get_slice(tx);
    auto tDrV = thr_mma_pv.partition_fragment_B(sV_T); // (MMA, MMA_D, MMA_K)
    auto tDrD = partition_fragment_C(mma_pv, Shape<Int<Br>, Int<D>>{}); // (MMA, MMA_Q, MMA_D)

    auto tOrO = partition_fragment_C(mma_pv, Shape<Int<Br>, Int<D>>{}); // (MMA, MMA_Q, MMA_D)
    fill(tOrO, 0.0f);

    TiledCopy s2r_copy_v = make_tiled_copy_B(s2r_atom_KV, mma_pv);
    ThrCopy s2r_thr_copy_v = s2r_copy_v.get_slice(tx);
    Tensor tXsV = s2r_thr_copy_v.partition_S(sV_T); // (CPY, MMA_D, MMA_K)
    Tensor tXrV = s2r_thr_copy_v.retile_D(tDrV);  // (CPY, MMA_D, MMA_K)

    #if DEBUG
    if (thread0()) {
        print("tSrQ: "); print(tSrQ); printf("\n");
        print("tSrK: "); print(tSrK); printf("\n");
        print("tSrS: "); print(tSrS); printf("\n");
        print("tXsQ: "); print(tXsQ); printf("\n");
        print("tXrQ: "); print(tXrQ); printf("\n");
        print("tXsK: "); print(tXsK); printf("\n");
        print("tXrK: "); print(tXrK); printf("\n");

        print("tDrV: "); print(tDrV); printf("\n");
        print("tDrD: "); print(tDrD); printf("\n");
        print("tOrO: "); print(tOrO); printf("\n");
        print("tXsV: "); print(tXsV); printf("\n");
        print("tXrV: "); print(tXrV); printf("\n");
    }
    #endif

    auto lane_row_max_new = make_tensor<float>(make_shape(Int<kWarpTileSeqLenQ>{}, Int<2>{}));  // make_tensor需要静态尺寸
    auto lane_row_sum_new = make_tensor<float>(make_shape(Int<kWarpTileSeqLenQ>{}, Int<2>{}));
    fill(lane_row_max_new, -INFINITY);
    fill(lane_row_sum_new, 0.0f);

    for(int j = 0; j < size<2>(gK); j++) {
        if constexpr (Kstage > 1) {
            for (int k = 0; k < size<1>(tVsV); k++) {
                copy(global_smem_kv, tVgV(_, k, _, j), tVsV(_, k, _));
            }

            cp_async_fence();

            if (j + Kstage - 1 < size<2>(gK)) {
                for (int k = 0; k < size<1>(tKsK); k++) {
                    copy(global_smem_kv, tKgK(_, k, _, j + Kstage - 1), tKsK(_, k, _, j + Kstage - 1));
                }
            }

            cp_async_fence();

        }
        else {
            for (int k = 0; k < size<1>(tKsK); k++) {
                copy(global_smem_kv, tKgK(_, k, _, j), tKsK(_, k, _, j));
            }

            cp_async_fence();

            for (int k = 0; k < size<1>(tVsV); k++) {
                copy(global_smem_kv, tVgV(_, k, _, j), tVsV(_, k, _));
            }

            cp_async_fence();

            cp_async_wait<1>();
            __syncthreads();
        }
        
        clear(tSrS);

        auto lane_row_max_old = make_tensor<float>(make_shape(Int<kWarpTileSeqLenQ>{}, Int<2>{}));
        for (int q = 0; q < size<0>(lane_row_max_old); q++) {
            lane_row_max_old(q, 0) = lane_row_max_new(q, 0);
            lane_row_max_old(q, 1) = lane_row_max_new(q, 1);
        }

        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: tSrQ values:\n");
        //     for (int q = 0; q < size<1>(tQsQ); q++) {
        //         for (int d = 0; d < size<2>(tQsQ); d++) {
        //             printf("  tQsQ(0, %d, %d) = %f\n", q, d, (float)tQsQ(0, q, d));
        //             printf("  tQsQ(1, %d, %d) = %f\n", q, d, (float)tQsQ(1, q, d));
        //             printf("  tQsQ(2, %d, %d) = %f\n", q, d, (float)tQsQ(2, q, d));
        //             printf("  tQsQ(3, %d, %d) = %f\n", q, d, (float)tQsQ(3, q, d));
        //         }
        //     }
        // }
        // Copy from shared memory to register (retiled to match MMA layout)
        // 先检查 shared memory 的值
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: Before copy - tXsQ (shared memory) values:\n");
        //     for (int cpy = 0; cpy < size<0>(tXsQ); cpy++) {
        //         for (int q = 0; q < size<1>(tXsQ); q++) {
        //             for (int d = 0; d < size<2>(tXsQ); d++) {
        //                 printf("  tXsQ(%d, %d, %d) = %f\n", cpy, q, d, (float)tXsQ(cpy, q, d));
        //             }
        //         }
        //     }
        // }
        
        copy(s2r_copy_q, tXsQ, tXrQ);

        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: After copy - tXrQ (register) values:\n");
        //     for (int cpy = 0; cpy < size<0>(tXrQ); cpy++) {
        //         for (int q = 0; q < size<1>(tXrQ); q++) {
        //             for (int d = 0; d < size<2>(tXrQ); d++) {
        //                 printf("  tXrQ(%d, %d, %d) = %f\n", cpy, q, d, (float)tXrQ(cpy, q, d));
        //             }
        //         }
        //     }
        // }

        copy(s2r_copy_k, tXsK(_, _, _, j % Kstage), tXrK);
        
        // 检查 gemm 输入 - 对比 tXrQ 和 tSrQ 的值（它们指向同一块内存但布局不同）
        // 注意：使用 tSrQ 的实际大小来限制访问，因为实际内存由 tSrQ 决定
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: After copy - tSrQ values (actual memory layout):\n");
        //     // 使用 tSrQ 的布局和大小，这是实际的内存布局
        //     for (int mma = 0; mma < size<0>(tSrQ); mma++) {
        //         for (int q = 0; q < size<1>(tSrQ); q++) {
        //             for (int d = 0; d < size<2>(tSrQ); d++) {
        //                 printf("  tSrQ(%d, %d, %d) = %f\n", mma, q, d, (float)tSrQ(mma, q, d));
        //             }
        //         }
        //     }
            
        //     for (int k = 0; k < size(tXrQ); k++) {
        //         printf("  tXrQ(%d) = %f\n", k, (float)tXrQ(k));
        //     }
        // }
        
        gemm(mma_qk, tSrQ, tSrK, tSrS);
        
        // 确保 MMA 操作的结果在寄存器中可见
        // __threadfence() 确保所有内存操作（包括寄存器到寄存器的操作）完成
        // __threadfence();
        
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: After gemm - tSrS values:\n");
        //     for (int q = 0; q < size<1>(tSrS); q++) {
        //         for (int k = 0; k < size<2>(tSrS); k++) {
        //             printf("  tSrS(0, %d, %d) = %f\n", q, k, (float)tSrS(make_coord(0, 0), q, k));
        //             printf("  tSrS(1, %d, %d) = %f\n", q, k, (float)tSrS(make_coord(1, 0), q, k));
        //             printf("  tSrS(2, %d, %d) = %f\n", q, k, (float)tSrS(make_coord(0, 1), q, k));
        //             printf("  tSrS(3, %d, %d) = %f\n", q, k, (float)tSrS(make_coord(1, 1), q, k));
        //         }
        //     }
        // }

        for (int q = 0; q < size<1>(tSrS); q++) {
            for (int k = 0; k < size<2>(tSrS); k++) {
                float tmp_max_0 = max(tSrS(make_coord(0, 0), q, k), tSrS(make_coord(1, 0), q, k)) * scale;
                float tmp_max_1 = max(tSrS(make_coord(0, 1), q, k), tSrS(make_coord(1, 1), q, k)) * scale;
                // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
                //     printf("mma_cutlass: tmp_max_0: %f, tmp_max_1: %f\n", tmp_max_0, tmp_max_1);
                // }
                lane_row_max_new(q, 0) = max(lane_row_max_new(q, 0), tmp_max_0);
                lane_row_max_new(q, 1) = max(lane_row_max_new(q, 1), tmp_max_1);
            }
        }

        for (int q = 0; q < size<0>(lane_row_max_new); q++) {
            lane_row_max_new(q, 0) = warp_reduce_max<float, 4>(lane_row_max_new(q, 0));
            lane_row_max_new(q, 1) = warp_reduce_max<float, 4>(lane_row_max_new(q, 1));
            // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_cutlass: lane_row_max_new: %f, %f\n", lane_row_max_new(q, 0), lane_row_max_new(q, 1));
            // }
        }

        auto acc = make_tensor<float>(make_shape(Int<kWarpTileSeqLenQ>{}, Int<2>{}));
        fill(acc, 0.0f);

        for (int q = 0; q < size<1>(tSrS); q++) {
            for (int k = 0; k < size<2>(tSrS); k++) {
                tSrS(make_coord(0, 0), q, k) = __expf(__fmaf_rn(tSrS(make_coord(0, 0), q, k), scale, -lane_row_max_new(q, 0)));
                tSrS(make_coord(1, 0), q, k) = __expf(__fmaf_rn(tSrS(make_coord(1, 0), q, k), scale, -lane_row_max_new(q, 0)));
                tSrS(make_coord(0, 1), q, k) = __expf(__fmaf_rn(tSrS(make_coord(0, 1), q, k), scale, -lane_row_max_new(q, 1)));
                tSrS(make_coord(1, 1), q, k) = __expf(__fmaf_rn(tSrS(make_coord(1, 1), q, k), scale, -lane_row_max_new(q, 1)));
                acc(q, 0) += (tSrS(make_coord(0, 0), q, k) + tSrS(make_coord(1, 0), q, k));
                acc(q, 1) += (tSrS(make_coord(0, 1), q, k) + tSrS(make_coord(1, 1), q, k));

                float tmp = tSrS(make_coord(1, 0), q, k);
                tSrS(make_coord(1, 0), q, k) = tSrS(make_coord(0, 1), q, k);
                tSrS(make_coord(0, 1), q, k) = tmp;
            }
        }

        for (int q = 0; q < size<0>(acc); q++) {
            acc(q, 0) = warp_reduce_sum<float, 4>(acc(q, 0));
            acc(q, 1) = warp_reduce_sum<float, 4>(acc(q, 1));
            // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_cutlass: acc: %f, %f\n", acc(q, 0), acc(q, 1));
            // }
        }

        for (int q = 0; q < size<0>(lane_row_max_new); q++) {
            lane_row_sum_new(q, 0) = __fmaf_rn(__expf(lane_row_max_old(q, 0) - lane_row_max_new(q, 0)), 
                                    lane_row_sum_new(q, 0), acc(q, 0));
            lane_row_sum_new(q, 1) = __fmaf_rn(__expf(lane_row_max_old(q, 1) - lane_row_max_new(q, 1)), 
                                    lane_row_sum_new(q, 1), acc(q, 1));

            // if (i == 0 && j == 1 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_cutlass: lane_row_sum_new: %f, %f\n", lane_row_sum_new(q, 0), lane_row_sum_new(q, 1));
            // }
        }
            
        if constexpr (Kstage > 1) {
            if (j + Kstage - 1 < Tc) {
                CP_ASYNC_WAIT_GROUP(0);
            }
            else {
                CP_ASYNC_WAIT_GROUP(0);
            }
        }
        else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        clear(tDrD);

        copy(s2r_copy_v, tXsV, tXrV);

        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: After copy - tDrV values:\n");
        //     print_tensor(tDrV);
        // }
        // print_tensor(tDrV);
        gemm(mma_pv, tSrS, tDrV, tDrD);

        // spin_cycles(2ULL * 1000 * 1000);
        // __syncthreads();
        // __threadfence();
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_cutlass: After gemm - tDrV values:\n");
        //     for (int q = 0; q < size<1>(tDrV); q++) {
        //         for (int k = 0; k < size<2>(tDrV); k++) {
        //             printf("  tDrV(0, %d, %d) = %f\n", q, k, (float)tDrV(0, q, k));
        //             printf("  tDrV(1, %d, %d) = %f\n", q, k, (float)tDrV(1, q, k));
        //         }
        //     }
        // }

        for (int q = 0; q < size<1>(tOrO); q++) {
            for (int k = 0; k < size<2>(tOrO); k++) {
                tOrO(make_coord(0, 0), q, k) = __fmaf_rn(__expf(lane_row_max_old(q, 0) - lane_row_max_new(q, 0)), 
                                               tOrO(make_coord(0, 0), q, k), tDrD(make_coord(0, 0), q, k));
                tOrO(make_coord(1, 0), q, k) = __fmaf_rn(__expf(lane_row_max_old(q, 0) - lane_row_max_new(q, 0)), 
                                               tOrO(make_coord(1, 0), q, k), tDrD(make_coord(1, 0), q, k));

                tOrO(make_coord(0, 1), q, k) = __fmaf_rn(__expf(lane_row_max_old(q, 1) - lane_row_max_new(q, 1)), 
                                               tOrO(make_coord(0, 1), q, k), tDrD(make_coord(0, 1), q, k));
                tOrO(make_coord(1, 1), q, k) = __fmaf_rn(__expf(lane_row_max_old(q, 1) - lane_row_max_new(q, 1)), 
                                               tOrO(make_coord(1, 1), q, k), tDrD(make_coord(1, 1), q, k));

                // if (i == 0 && j == 1 && b == 0 && h == 0 && tx == 0) {
                //     printf("mma_cutlass: tOrO: %f, %f, %f, %f\n", tOrO(make_coord(0, 0), q, k), tOrO(make_coord(1, 0), q, k), tOrO(make_coord(0, 1), q, k), tOrO(make_coord(1, 1), q, k));
                // }
            }
        }

        if constexpr (Kstage > 1) {
            if (j + Kstage - 1 < Tc) {
                cp_async_wait<0>();
            }
        }
    }

    for (int q = 0; q < size<1>(tOrO); q++) {
        for (int k = 0; k < size<2>(tOrO); k++) {
            tOrO(make_coord(0, 0), q, k) = __fdividef(tOrO(make_coord(0, 0), q, k), lane_row_sum_new(q, 0));
            tOrO(make_coord(1, 0), q, k) = __fdividef(tOrO(make_coord(1, 0), q, k), lane_row_sum_new(q, 0));
            tOrO(make_coord(0, 1), q, k) = __fdividef(tOrO(make_coord(0, 1), q, k), lane_row_sum_new(q, 1));
            tOrO(make_coord(1, 1), q, k) = __fdividef(tOrO(make_coord(1, 1), q, k), lane_row_sum_new(q, 1));

            // if (i == 0 && b == 0 && h == 0 && tx == 32) {
            //     printf("mma_cutlass: tOrO: %f, %f, %f, %f\n", tOrO(make_coord(0, 0), q, k), tOrO(make_coord(1, 0), q, k), tOrO(make_coord(0, 1), q, k), tOrO(make_coord(1, 1), q, k));
            // }
        }
    }

    auto r2g_copy_O = make_tiled_copy_C(r2g_atom_O, mma_pv);
    auto r2g_thr_copy_O = r2g_copy_O.get_slice(tx);
    auto tXgO = r2g_thr_copy_O.partition_D(gO);
    auto tXrO = r2g_thr_copy_O.retile_S(tOrO);
    
    #if DEBUG
    {
        print("tXrO: "); print(tXrO); printf("\n");
        print("tOrO: "); print(tOrO); printf("\n");
    }
    #endif

    copy(r2g_copy_O, tXrO, tXgO);
}

template<
    const int D>
void launch_flash_attention_mma_cutlass_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N,
    float scale,
    cudaStream_t stream) {
    
    using namespace cute;
    
    using T_QKV = cutlass::tfloat32_t;
    using TP = cutlass::tfloat32_t;
    using TO = float;
    using TI = cutlass::tfloat32_t;

    TI alpha = static_cast<TI>(1.0f);
    TI beta = static_cast<TI>(0.0f);

    constexpr int KMmaAtomM = 16;
    constexpr int KMmaAtomN = 8;
    constexpr int KMmaAtomK = 8;
    constexpr int kMmaTileSeqLenQ = 4;
    constexpr int kMmaTileSeqLenK = 1;
    constexpr int kMmaTileSeqLenP = 4;
    constexpr int kMmaTileHeadDimV = 1;
    constexpr int kWarpTileSeqLenQ = 2;
    constexpr int kWarpTileSeqLenK = 4;
    constexpr int kWarpTileSeqLenP = 2;
    constexpr int kWarpTileHeadDimV = 4;

    constexpr int NumThreads = WARP_SIZE * kMmaTileSeqLenQ;

    constexpr int Br = KMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = KMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; 
    
    constexpr int NumPerRow = D / 4;
    
    // if D = 64, NumPerRow = 16, NumThreads = 128, then global_smem_q is 16 * 16
    TiledCopy global_smem_q = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T_QKV>{},
                                                Layout<Shape<Int<NumThreads / NumPerRow>, Int<NumPerRow>>, 
                                                Stride<Int<NumPerRow>, _1>>{},
                                                Layout<Shape<_1, _4>, 
                                                Stride<_0, _1>>{}); // 这里面的参数都是类型，并没有实例化


    // print_latex(global_smem_q);

    TiledCopy global_smem_kv = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T_QKV>{},
                                                Layout<Shape<Int<NumThreads / NumPerRow>, Int<NumPerRow>>, 
                                                Stride<Int<NumPerRow>, _1>>{},
                                                Layout<Shape<_1, _4>, 
                                                Stride<_0, _1>>{});
                                                
    // print_latex(global_smem_kv);

    TiledMMA mma_qk = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                                        Layout<Shape<Int<kMmaTileSeqLenQ>, Int<kMmaTileSeqLenK>>>{},
                                        Tile<Int<Br>, Int<Bc>, Int<D>>{});
    
    // print_latex(mma_qk);

    TiledMMA mma_pv = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                                        Layout<Shape<Int<kMmaTileSeqLenP>, Int<kMmaTileHeadDimV>>>{},
                                        Tile<Int<Br>, Int<D>, Layout<Shape<_4, _2, Int<Bc / KMmaAtomK>>, Stride<_2, _1, _8>>>{});

    // print_latex(mma_pv);

    Copy_Atom<SM75_U32x4_LDSM_N, T_QKV> s2r_atom_Q;
    Copy_Atom<UniversalCopy<T_QKV>, T_QKV> s2r_atom_KV;

    Copy_Atom<UniversalCopy<TO>, TO> r2g_atom_O;
    
    const int Tr = ceil((float)N / Br);
    const int Tc = ceil((float)N / Bc);

    dim3 grid(Tr, B, H);
    dim3 block(NumThreads);
    
    constexpr int PAD = 0;
    constexpr int Kstage = 1;
    auto smem_size = (Br * (D + PAD) + Bc * (D + PAD) + Bc * (D + PAD) * Kstage) * sizeof(T_QKV);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    auto kernel_fptr = flash_attention_mma_cutlass_forward_kernel<NumThreads, KMmaAtomM, KMmaAtomN, 
    KMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ, 
    kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, Kstage, D, Br, Bc,
    decltype(global_smem_q), decltype(global_smem_kv), decltype(mma_qk), decltype(mma_pv), decltype(s2r_atom_Q), 
    decltype(s2r_atom_KV), decltype(r2g_atom_O), T_QKV>;
    
    cudaFuncSetCacheConfig(kernel_fptr, cudaFuncCachePreferShared);

    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
    } else {
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }

    // Debug breakpoint marker - set breakpoint here to catch kernel launch
    #ifdef DEBUG
    printf("Launching cutlass kernel: grid=(%d,%d,%d), block=(%d), smem=%zu\n", 
           grid.x, grid.y, grid.z, block.x, smem_size);
    #endif

    kernel_fptr<<<grid, block, smem_size, stream>>>(
        Q, K, V, O, B, H, N, scale, Tr, Tc, global_smem_q, global_smem_kv, mma_qk, mma_pv, s2r_atom_Q, s2r_atom_KV, r2g_atom_O);

    // cudaFree(d_l);
    // cudaFree(d_m);
}

} // namespace attention

template void attention::launch_flash_attention_mma_cutlass_forward<64>(
    const float*, const float*, const float*, float*,
    int, int, int, float, cudaStream_t);

template void attention::launch_flash_attention_mma_cutlass_forward<128>(
    const float*, const float*, const float*, float*,
    int, int, int, float, cudaStream_t);