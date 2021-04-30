#pragma once
#include "util.h"
#define WIN_SIZE 7
void MedianFilterCUDA(uint8* inputImage, uint8* outputImage, int height, int width, int channels,
    double& elapsed, double& kernelElapsed);
void MedianFilterCPU(uint8* inputImage, uint8* outputImage, int height, int width, int channels);
void MedianFilterOMP(uint8* inputImage, uint8* outputImage, int height, int width, int channels);