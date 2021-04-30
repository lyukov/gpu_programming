#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MedianFilter.h"

int main(){
    cv::Mat image = cv::imread("Test_Image.png");

    cout << "Height: " << image.rows 
        << ", Width: " << image.cols 
        << ", Channels: " << image.channels()
        << endl;

    cv::Mat outputImage(image);
    double elapsed, kernelElapsed;
    MedianFilterCUDA(image.data, outputImage.data, image.rows, image.cols, image.channels(), elapsed, kernelElapsed);

    cout << "All GPU time: " << elapsed << endl
        << "Kernel time: " << kernelElapsed << endl
        << "Copy time: " << elapsed - kernelElapsed << endl;

    cv::imwrite("Median_Image.png", outputImage);
    system("pause");
    return 0;
}