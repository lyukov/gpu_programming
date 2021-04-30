#pragma once
#include "CudaUtils.cuh"

class CudaTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    CudaTimer();
    double stop();
};