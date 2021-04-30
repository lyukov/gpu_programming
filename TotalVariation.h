#pragma once
#include "util.h"

double TotalGeneralizedVariationCUDA(
    uint8* image,
    int height, int width, int channels,
    double lambda1, double lambda2,
    double& elapsed,
    double& kernelElapsed
);

double TotalGeneralizedVariationCPU(
    uint8* image,
    int height, int width, int channels,
    double lambda1, double lambda2
);

double TotalGeneralizedVariationOMP(
    uint8* image,
    int height, int width, int channels,
    double lambda1, double lambda2
);