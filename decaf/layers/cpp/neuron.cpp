#include <algorithm>
#include "neuron.h"

using std::max;

template <typename Dtype>
inline void _relu_forward(const Dtype* input, Dtype* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = max(input[i], Dtype(0));
    }
    return;
}

extern "C" {

void relu_forward(const int len, const void* input, void* output, int n) {
    switch(len) {
    case sizeof(float):
        _relu_forward<float>((const float*) input, (float*) output, n);
        break;
    case sizeof(double):
        _relu_forward<double>((const double*) input, (double*) output, n);
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
}

}
