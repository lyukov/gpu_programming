﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MedianFilter.h"
#include "CudaTimer.h"
#include <stdio.h>
#include <iostream>

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

__global__ void medianKernel(
        uint8* inputImage,
        uint8* outputImage,
        int channels,
        int height,
        int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    int radius = WIN_SIZE / 2;
    uint8 arr[WIN_SIZE * WIN_SIZE];
    int ind = 0;
    int xFrom = max(x - radius, 0);
    int xTo = min(x + radius, width);
    int yFrom = max(y - radius, 0);
    int yTo = min(y + radius, height);
    for (int dx = xFrom; dx <= xTo; ++dx) {
        for (int dy = yFrom; dy <= yTo; ++dy) {
            arr[ind++] = inputImage[dy * width * channels + dx * channels + ch];
        }
    }
    uint8 temp;
    for (int i = 0; i < ind - 1; i++) {
        for (int j = 0; j < ind - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    outputImage[y * width * channels + x * channels + ch] = arr[ind / 2];
}

void MedianFilterCUDA(
        uint8* inputImage,
        uint8* outputImage,
        int height,
        int width,
        int channels,
        double& elapsed,
        double& kernelElapsed
) {
    uint8* devInputImage = NULL;
    uint8* devOutputImage = NULL;
    long imageSizeInBytes = height * width * channels;

    CudaTimer timer = CudaTimer();

    cudaMalloc((void**)&devInputImage, imageSizeInBytes);
    cudaMalloc((void**)&devOutputImage, imageSizeInBytes);

    cudaMemcpy(devInputImage, inputImage, imageSizeInBytes, cudaMemcpyHostToDevice);
    //cudaMemcpy(devOutputImage, devInputImage, imageSizeInBytes, cudaMemcpyDeviceToDevice);

    dim3 gridSize(width, height);
    dim3 blockSize(1, 1, channels);
    CudaTimer kernelTimer = CudaTimer();
    SAFE_KERNEL_CALL((
        medianKernel <<<gridSize, blockSize>>> (devInputImage, devOutputImage, channels, height, width)
    ));
    kernelElapsed = kernelTimer.stop();

    cudaMemcpy(outputImage, devOutputImage, imageSizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(devInputImage);
    cudaFree(devOutputImage);

    elapsed = timer.stop();
}