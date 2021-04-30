#include "TotalVariation.h"
#include <algorithm>

double TotalGeneralizedVariationCPU(
    uint8* image,
    int height, int width, int channels,
    double lambda1, double lambda2
) {
    double difference = 0;
    for (int x = 1; x < width - 1; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int ch = 0; ch < channels; ++ch) {
                int idx = y * width * channels + x * channels + ch;
                difference += lambda1 * abs(
                    (double)image[idx + channels] - (double)image[idx]
                ) + lambda2 * abs(
                    (double)image[idx - channels]
                    - 2.0 * (double)image[idx]
                    + (double)image[idx + channels]
                );
            }
        }
    }
    for (int x = 0; x < width; ++x) {
        for (int y = 1; y < height - 1; ++y) {
            for (int ch = 0; ch < channels; ++ch) {
                int idx = y * width * channels + x * channels + ch;
                int yShift = width * channels;
                difference += lambda1 * abs(
                    (double)image[idx + yShift] - (double)image[idx]
                ) + lambda2 * abs(
                    (double)image[idx - yShift]
                    - 2.0 * (double)image[idx]
                    + (double)image[idx + yShift]
                );
            }
        }
    }
    return difference;
}

double TotalGeneralizedVariationOMP(
    uint8* image,
    int height, int width, int channels,
    double lambda1, double lambda2
) {
    double difference = 0;
    #pragma omp parallel for reduction(+ : difference)
    for (int x = 1; x < width - 1; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int ch = 0; ch < channels; ++ch) {
                int idx = y * width * channels + x * channels + ch;
                difference += lambda1 * abs(
                    (double)image[idx + channels] - (double)image[idx]
                ) + lambda2 * abs(
                    (double)image[idx - channels]
                    - 2.0 * (double)image[idx]
                    + (double)image[idx + channels]
                );
            }
        }
    }
    #pragma omp parallel for reduction(+ : difference)
    for (int x = 0; x < width; ++x) {
        for (int y = 1; y < height - 1; ++y) {
            for (int ch = 0; ch < channels; ++ch) {
                int idx = y * width * channels + x * channels + ch;
                int yShift = width * channels;
                difference += lambda1 * abs(
                    (double)image[idx + yShift] - (double)image[idx]
                ) + lambda2 * abs(
                    (double)image[idx - yShift]
                    - 2.0 * (double)image[idx]
                    + (double)image[idx + yShift]
                );
            }
        }
    }
    return difference;
}