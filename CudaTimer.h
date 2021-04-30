#pragma once
#include "cuda_runtime.h"

class CudaTimer {
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;

public:
	CudaTimer();
	double stop();
};