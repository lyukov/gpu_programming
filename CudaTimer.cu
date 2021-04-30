#include "CudaTimer.h"
#include "device_launch_parameters.h"

CudaTimer::CudaTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
}

double CudaTimer::stop() {
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds / 1000.0;
}