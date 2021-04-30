#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MedianFilter.h"

void medianFilterExperiment() {
    cv::Mat image = cv::imread("Test_Image.png");
    cout << "Height: " << image.rows
        << ", Width: " << image.cols
        << ", Channels: " << image.channels()
        << endl;
    cv::Mat outputImage = image.clone();

    double elapsed, kernelElapsed;
    MedianFilterCUDA(image.data, outputImage.data, image.rows, image.cols, image.channels(), elapsed, kernelElapsed);
    cv::imwrite("Median_Image_GPU.png", outputImage);

    clock_t start_s = clock();
    MedianFilterCPU(image.data, outputImage.data, image.rows, image.cols, image.channels());
    clock_t stop_s = clock();
    double cpuElapsed = (stop_s - start_s) / double(CLOCKS_PER_SEC);
    cv::imwrite("Median_Image_CPU.png", outputImage);

    const int NUM_THREADS = 4;
    omp_set_num_threads(NUM_THREADS);

    start_s = clock();
    MedianFilterOMP(image.data, outputImage.data, image.rows, image.cols, image.channels());
    stop_s = clock();
    double ompElapsed = (stop_s - start_s) / double(CLOCKS_PER_SEC);
    cv::imwrite("Median_Image_OMP.png", outputImage);

    cout << "All GPU time: " << elapsed << endl
        << "Kernel time: " << kernelElapsed << endl
        << "Copy time: " << elapsed - kernelElapsed << endl
        << "CPU time: " << cpuElapsed << endl
        << "OMP time (" << NUM_THREADS << " threads): " << ompElapsed << endl;
}

int main() {
    medianFilterExperiment();
    system("pause");
    return 0;
}