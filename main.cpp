#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MedianFilter.h"
#include "TotalVariation.h"

const int NUM_THREADS = 4;

void medianFilterExperiment() {
    cout << "Testing Median filter" << endl;
    cv::Mat image = cv::imread("Test_Image.png");
    cv::Mat outputImage = image.clone();

    double elapsed, kernelElapsed;
    MedianFilterCUDA(image.data, outputImage.data, image.rows, image.cols, image.channels(), elapsed, kernelElapsed);
    cv::imwrite("Median_Image_GPU.png", outputImage);

    clock_t start_s = clock();
    MedianFilterCPU(image.data, outputImage.data, image.rows, image.cols, image.channels());
    clock_t stop_s = clock();
    double cpuElapsed = (stop_s - start_s) / double(CLOCKS_PER_SEC);
    cv::imwrite("Median_Image_CPU.png", outputImage);

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

void totalVariationExperiment() {
    cout << "Testing Total Generalized Variation" << endl;
    cv::Mat image = cv::imread("BigExample.jpg");
    //cv::Mat image = cv::imread("Test_Image.png");
    const double lambda1 = 0.3;
    const double lambda2 = 0.7;

    double elapsed, kernelElapsed;
    double resultCUDA = TotalGeneralizedVariationCUDA(
        image.data, image.rows, image.cols, image.channels(),
        lambda1, lambda2,
        elapsed, kernelElapsed
    );

    clock_t start_s = clock();
    double resultCPU = TotalGeneralizedVariationCPU(
        image.data, image.rows, image.cols, image.channels(),
        lambda1, lambda2
    );
    clock_t stop_s = clock();
    double cpuElapsed = (stop_s - start_s) / double(CLOCKS_PER_SEC);

    omp_set_num_threads(NUM_THREADS);
    start_s = clock();
    double resultOMP = TotalGeneralizedVariationOMP(
        image.data, image.rows, image.cols, image.channels(),
        lambda1, lambda2
    );
    stop_s = clock();
    double ompElapsed = (stop_s - start_s) / double(CLOCKS_PER_SEC);

    cout << "All GPU time: " << elapsed << endl
        << "Kernel time: " << kernelElapsed << endl
        << "Copy time: " << elapsed - kernelElapsed << endl
        << "CPU time: " << cpuElapsed << endl
        << "OMP time (" << NUM_THREADS << " threads): " << ompElapsed << endl
        << "\nCUDA: TGV = " << resultCUDA << endl
        << "CPU: TGV = " << resultCPU << endl
        << "OMP: TGV = " << resultOMP << endl;
}

int main() {
    totalVariationExperiment();
    cout << endl;
    medianFilterExperiment();
    system("pause");
    return 0;
}