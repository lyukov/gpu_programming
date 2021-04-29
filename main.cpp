// This Program is Written by Abubakr Shafique (abubakr.shafique@gmail.com)
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Inversion_CUDA.h"
#include "Median_Filter_CUDA.h"

using namespace std;
using namespace cv;

int main(){
	Mat image = imread("Test_Image.png");

	cout << "Height: " << image.rows 
		<< ", Width: " << image.cols 
		<< ", Channels: " << image.channels()
		<< endl;

	Median_Filter_CUDA(image.data, image.rows, image.cols, image.channels());

	imwrite("Median_Image.png", image);

	return 0;
}