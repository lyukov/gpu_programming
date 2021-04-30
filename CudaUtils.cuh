#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaTimer.h"
#include <stdio.h>

#define SAFE_CALL(Call) { \
    cudaError_t cuerr = Call; \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error: %s at call \"" #Call "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL(KernelCall) { \
    KernelCall; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCall "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCall "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA kernel execution, aborting..."; \
    } \
}