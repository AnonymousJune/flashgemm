## Introduction

This project optimizes the performance of both single GEMM operations and sequences of multiple consecutive GEMM operations using the x86 instruction set. It includes three folders: the `singlegemm` folder optimizes single FP32 GEMM operations of various shapes, the `multigemm` folder implements arbitrary sequences of consecutive GEMM operations for INT8, BF16, and FP32 data types, and the `AMX` folder contains executable files implemented with the AMX instruction set.

## Prerequisites

- x86 cpu equipped with AVX512
- gcc >=11.3
- binutils >= 2.42

## Usage

```
cd singlegemm/test
sh run.sh

cd multigemm/test
sh run.sh
```

