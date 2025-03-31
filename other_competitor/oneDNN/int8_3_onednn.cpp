/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <chrono>
#include <pthread.h>
#include <sched.h>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

#define NUM 32
#define PEAK_GFLOPS 2.6
static double gtod_ref_time_sec = 0.0;

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)tv.tv_sec;

    norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

int GEMM1[30] = {
    32, 12544, 288, // ID1-10
    144, 3136, 1296,
    192, 3136, 1728,
    192, 784, 4800,
    336, 784, 8400,
    32, 16384, 288,
    96, 4096, 864,
    144, 4096, 1296,
    144, 1024, 1296,
    192, 1024, 1728};

int GEMM2[30] = {
    24, 12544, 32,
    32, 3136, 144,
    32, 3136, 192,
    56, 784, 192,
    56, 784, 336,
    16, 16384, 32,
    24, 4096, 96,
    24, 4096, 144,
    32, 1024, 144,
    32, 1024, 192};

int GEMM3[30] = {
    144, 12544, 24,
    192, 3236, 32,
    192, 3136, 32,
    336, 784, 56,
    336, 784, 56,
    96, 16384, 16,
    144, 4096, 24,
    144, 4096, 24,
    192, 1024, 32,
    192, 1024, 32};

void matmul_example(dnnl::engine::kind engine_kind)
{

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    FILE *fp;
    if ((fp = fopen("../result/int8_3_onednn.txt", "w")) == NULL)
    {
        puts("Fail to open file!");
        exit(0);
    }

    for (int j = 0; j < 10; j++)
    {
        // Tensor dimensions.
        const memory::dim M1 = GEMM1[j * 3], N = GEMM1[j * 3 + 1], K1 = GEMM1[j * 3 + 2];
        const memory::dim M2 = GEMM2[j * 3], K2 = GEMM2[j * 3 + 2];
        const memory::dim M3 = GEMM3[j * 3], K3 = GEMM3[j * 3 + 2];

        // Source (src), weights, and destination (dst) tensors dimensions.
        memory::dims A1_dims = {M1, K1};
        memory::dims A2_dims = {M2, K2};
        memory::dims A3_dims = {M3, K3};
        memory::dims B_dims = {K1, N};
        memory::dims C1_dims = {M1, N};
        memory::dims C2_dims = {M2, N};
        memory::dims C3_dims = {M3, N};

        // Allocate buffers.
        std::vector<int8_t> A1_data(product(A1_dims));
        std::vector<int8_t> A2_data(product(A2_dims));
        std::vector<int8_t> A3_data(product(A3_dims));
        std::vector<int8_t> B_data(product(B_dims));
        std::vector<int8_t> C1_data(product(C1_dims));
        std::vector<int8_t> C2_data(product(C2_dims));
        std::vector<int> C3_data(product(C3_dims));

        // Randomly initialize src and weights with int8_t values between -128 and 127.
        std::generate(A1_data.begin(), A1_data.end(), []()
                      { return static_cast<int8_t>(std::rand() % 256 - 128); });
        std::generate(A2_data.begin(), A2_data.end(), []()
                      { return static_cast<int8_t>(std::rand() % 256 - 128); });
        std::generate(A3_data.begin(), A3_data.end(), []()
                      { return static_cast<int8_t>(std::rand() % 256 - 128); });
        std::generate(B_data.begin(), B_data.end(), []()
                      { return static_cast<uint8_t>(std::rand() % 256 - 128); });

        // Create memory descriptors and memory objects for src, weights, and dst.
        auto A1_md = memory::desc(A1_dims, dt::s8, tag::ab);
        auto A2_md = memory::desc(A2_dims, dt::s8, tag::ab);
        auto A3_md = memory::desc(A3_dims, dt::s8, tag::ab);
        auto B_md = memory::desc(B_dims, dt::s8, tag::ab);
        auto C1_md = memory::desc(C1_dims, dt::s8, tag::ab);
        auto C2_md = memory::desc(C2_dims, dt::s8, tag::ab);
        auto C3_md = memory::desc(C3_dims, dt::s32, tag::ab);

        auto A1_mem = memory(A1_md, engine);
        auto A2_mem = memory(A2_md, engine);
        auto A3_mem = memory(A3_md, engine);
        auto B_mem = memory(B_md, engine);
        auto C1_mem = memory(C1_md, engine);
        auto C2_mem = memory(C2_md, engine);
        auto C3_mem = memory(C3_md, engine);

        // Write data to memory object's handles.
        write_to_dnnl_memory(A1_data.data(), A1_mem);
        write_to_dnnl_memory(A2_data.data(), A2_mem);
        write_to_dnnl_memory(A3_data.data(), A3_mem);
        write_to_dnnl_memory(B_data.data(), B_mem);

        // Correctly define memory descriptors for GEMM inputs and outputs
        auto matmul1_pd = matmul::primitive_desc(engine, A1_md, B_md, C1_md);
        auto matmul2_pd = matmul::primitive_desc(engine, A2_md, C1_md, C2_md);
        auto matmul3_pd = matmul::primitive_desc(engine, A3_md, C2_md, C3_md);

        // Create primitives for three GEMMs.
        auto matmul1_prim = matmul(matmul1_pd);
        auto matmul2_prim = matmul(matmul2_pd);
        auto matmul3_prim = matmul(matmul3_pd);

        // Primitive arguments for three GEMMs.
        std::unordered_map<int, memory> matmul1_args = {
            {DNNL_ARG_SRC, A1_mem},
            {DNNL_ARG_WEIGHTS, B_mem},
            {DNNL_ARG_DST, C1_mem}};
        std::unordered_map<int, memory> matmul2_args = {
            {DNNL_ARG_SRC, A2_mem},
            {DNNL_ARG_WEIGHTS, C1_mem},
            {DNNL_ARG_DST, C2_mem}};
        std::unordered_map<int, memory> matmul3_args = {
            {DNNL_ARG_SRC, A3_mem},
            {DNNL_ARG_WEIGHTS, C2_mem},
            {DNNL_ARG_DST, C3_mem}};

        // Number of iterations for performance testing
        const int num_iterations = 10;

        for (int i = 0; i < 5; ++i)
        {
            // Execute three GEMMs sequentially.
            matmul1_prim.execute(engine_stream, matmul1_args);
            matmul2_prim.execute(engine_stream, matmul2_args);
            matmul3_prim.execute(engine_stream, matmul3_args);

            // Wait for the computation to finalize.
            engine_stream.wait();
        }

        // Start time measurement
        double start = dclock();

        for (int i = 0; i < num_iterations; ++i)
        {
            // Execute three GEMMs sequentially.
            matmul1_prim.execute(engine_stream, matmul1_args);
            matmul2_prim.execute(engine_stream, matmul2_args);
            matmul3_prim.execute(engine_stream, matmul3_args);

            // Wait for the computation to finalize.
            engine_stream.wait();
        }

        // End time measurement
        double cost = (dclock() - start) / num_iterations;
        double ops = (double)(M1 * K1 + M2 * K2 + M3 * K3) * N * 1.0e-09 * 2;

        // Read data from memory object's handle.
        read_from_dnnl_memory(C3_data.data(), C3_mem);

        printf("onednn_int8:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d flops = %-10.3lf effic= %.3lf %\n",
               N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 4 * 2) * 100 / NUM);
        fprintf(fp, "%.3lf \n", ops / cost);
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
