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

#define NUM 24
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

int GEMM[60] = {
    32, 12544, 288, // ID1-10
    144, 3136, 1296,
    192, 3136, 1728,
    192, 784, 4800,
    336, 784, 8400,
    32, 16384, 288,
    96, 4096, 864,
    144, 4096, 1296,
    144, 1024, 1296,
    192, 1024, 1728,
    128, 128, 512, // ID11-20
    256, 256, 512,
    512, 512, 512,
    1024, 1024, 512,
    2048, 2048, 512,
    128, 16384, 512,
    256, 16384, 512,
    512, 16384, 512,
    1024, 16384, 512,
    2048, 16384, 512};

void matmul_example(dnnl::engine::kind engine_kind)
{

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    FILE *fp;
    if ((fp = fopen("../result/int8_onednn.txt", "w")) == NULL)
    {
        puts("Fail to open file!");
        exit(0);
    }

    for (int j = 0; j < 20; j++)
    {
        // Tensor dimensions.
        const memory::dim M = GEMM[j * 3], K = GEMM[j * 3 + 1], N = GEMM[j * 3 + 2];

        // Source (src), weights, and destination (dst) tensors dimensions.
        memory::dims src_dims = {M, K};
        memory::dims weights_dims = {K, N};
        memory::dims dst_dims = {M, N};

        // Allocate buffers.
        std::vector<int8_t> src_data(product(src_dims));
        std::vector<int8_t> weights_data(product(weights_dims));
        std::vector<int> dst_data(product(dst_dims));

        // Randomly initialize src and weights between -1 and 1.
        std::generate(src_data.begin(), src_data.end(), []()
                      { return static_cast<int8_t>(std::rand() % 256 - 128); });

        std::generate(weights_data.begin(), weights_data.end(), []()
                      { return static_cast<int8_t>(std::rand() % 256 - 128); });

        // Create memory descriptors and memory objects for src, weights, and dst.
        auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
        auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab);
        auto dst_md = memory::desc(dst_dims, dt::s32, tag::ab);

        auto src_mem = memory(src_md, engine);
        auto weights_mem = memory(weights_md, engine);
        auto dst_mem = memory(dst_md, engine);

        // Write data to memory object's handles.
        write_to_dnnl_memory(src_data.data(), src_mem);
        write_to_dnnl_memory(weights_data.data(), weights_mem);

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);

        // Create the primitive.
        auto matmul_prim = matmul(matmul_pd);

        // Primitive arguments.
        std::unordered_map<int, memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, src_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        matmul_args.insert({DNNL_ARG_DST, dst_mem});

        // Number of iterations for performance testing
        const int num_iterations = 10;

        for (int i = 0; i < 5; ++i)
        {
            // Primitive execution: matrix multiplication.
            matmul_prim.execute(engine_stream, matmul_args);

            // Wait for the computation to finalize.
            engine_stream.wait();
        }

        // Start time measurement
        double start = dclock();

        for (int i = 0; i < num_iterations; ++i)
        {
            // Primitive execution: matrix multiplication.
            matmul_prim.execute(engine_stream, matmul_args);

            // Wait for the computation to finalize.
            engine_stream.wait();
        }

        // End time measurement
        double cost = (dclock() - start) / num_iterations;
        double ops = (double)M * N * K * 1.0e-09 * 2;

        // Read data from memory object's handle.
        read_from_dnnl_memory(dst_data.data(), dst_mem);

        printf("oneDNN:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf\%\n", M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 4 * 2) * 100 / NUM);
        fprintf(fp, "%.3lf\n", ops / cost);
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
