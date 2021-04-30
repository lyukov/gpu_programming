#include "MedianFilter.h"
#include <cstring>
#include <algorithm>

void MedianFilterCPU(uint8* inputImage, uint8* outputImage, int height, int width, int channels) {
    int radius = WIN_SIZE / 2;
    
    uint8 arr[WIN_SIZE * WIN_SIZE];
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int ch = 0; ch < channels; ++ch) {
                int ind = 0;
                int xFrom = max(x - radius, 0);
                int xTo = min(x + radius, width);
                int yFrom = max(y - radius, 0);
                int yTo = min(y + radius, height);
                for (int dx = xFrom; dx <= xTo; ++dx) {
                    for (int dy = yFrom; dy <= yTo; ++dy) {
                        arr[ind++] = inputImage[dy * width * channels + dx * channels + ch];
                    }
                }
                uint8 temp;
                for (int i = 0; i < ind - 1; i++) {
                    for (int j = 0; j < ind - i - 1; j++) {
                        if (arr[j] > arr[j + 1]) {
                            temp = arr[j];
                            arr[j] = arr[j + 1];
                            arr[j + 1] = temp;
                        }
                    }
                }
                outputImage[y * width * channels + x * channels + ch] = arr[ind / 2];
            }
        }
    }
}

void MedianFilterOMP(uint8* inputImage, uint8* outputImage, int height, int width, int channels);