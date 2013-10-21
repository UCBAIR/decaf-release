#include <cstdio>
#include <ctime>
#include <cstring>
#include "local_response_normalization.h"

#define N 10000
#define D 256
#define SIZE 5
#define ALPHA 1.0
#define BETA 0.75
#define ITER 100

int main(int argc, char** argv) {
    // prepare data
    float* bottom_data = new float[N*D];
    float* bottom_diff = new float[N*D];
    float* scale = new float[N*D];
    float* top_data = new float[N*D];
    float* top_diff = new float[N*D];

    memset(bottom_data, 0, sizeof(float) * N * D);
    memset(bottom_diff, 0, sizeof(float) * N * D);
    memset(scale, 0, sizeof(float) * N * D);
    memset(top_data, 0, sizeof(float) * N * D);
    memset(top_diff, 0, sizeof(float) * N * D);

    clock_t start = clock();
    for (int i = 0; i < ITER; ++i) {
        lrn_forward(sizeof(float), bottom_data, top_data, scale, N, D, SIZE, ALPHA, BETA, 1);
    }
    clock_t duration = clock() - start;

    printf("Forward elapsed time: %.2f milliseconds per round.\n",
           float(duration) / CLOCKS_PER_SEC / ITER * 1000);
    
    start = clock();
    for (int i = 0; i < ITER; ++i) {
        lrn_backward(sizeof(float), bottom_data, top_data, bottom_diff, top_diff,
                scale, N, D, SIZE, ALPHA, BETA, 1);
    }
    duration = clock() - start;

    printf("Backward elapsed time: %.2f milliseconds per round.\n",
           float(duration) / CLOCKS_PER_SEC / ITER * 1000);

    delete[] bottom_data;
    delete[] bottom_diff;
    delete[] scale;
    delete[] top_data;
    delete[] top_diff;

    return 0;
}
