#include "CudaUtils.cuh"
#include "TotalVariation.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>

__global__ void differenceKernel(
        uint8* image,
        double* diff,
        int height, int width, int channels,
        double lambda1, double lambda2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) { return; }
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = y * width * channels + x * channels + ch;
    double difference = 0;
    double center = (double)image[idx];
    if (x > 0 && x < width - 1) {
        double plusX = (double)image[idx + channels];
        double minusX = (double)image[idx - channels];
        difference += lambda1 * abs(plusX - center)
            + lambda2 * abs(minusX - 2.0 * center + plusX);
    }
    if (y > 0 && y < height - 1) {
        int yShift = width * channels;
        double plusY = (double)image[idx + yShift];
        double minusY = (double)image[idx - yShift];
        difference += lambda1 * abs(plusY - center)
            + lambda2 * abs(minusY - 2.0 * center + plusY);
    }
    diff[idx] = difference;
}

double TotalGeneralizedVariationCUDA(
        uint8* image,
        int height, int width, int channels,
        double lambda1, double lambda2,
        double& elapsed,
        double& kernelElapsed
) {
    double* devDifference = NULL;
    uint8* devInputImage = NULL;
    long imageSizeInBytes = height * width * channels;

    CudaTimer timer = CudaTimer();

    cudaMalloc((void**)&devInputImage, imageSizeInBytes);
    cudaMalloc((void**)&devDifference, imageSizeInBytes * sizeof(double));

    cudaMemcpy(devInputImage, image, imageSizeInBytes, cudaMemcpyHostToDevice);

    int blockSize = min(1024, width);
    dim3 gridSize((width + blockSize - 1) / blockSize, height, channels);
    CudaTimer kernelTimer = CudaTimer();
    SAFE_KERNEL_CALL((
        differenceKernel <<<gridSize, blockSize>>> (
            devInputImage, devDifference, height, width, channels, lambda1, lambda2
        )
    ));
    kernelElapsed = kernelTimer.stop();

    double result = thrust::reduce(
        thrust::device,
        devDifference,
        devDifference + imageSizeInBytes,
        0.0,
        thrust::plus<double>()
    );

    cudaFree(devInputImage);
    cudaFree(devDifference);

    elapsed = timer.stop();
    return result;
}