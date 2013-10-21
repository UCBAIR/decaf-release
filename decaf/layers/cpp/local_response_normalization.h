#ifndef _DECAF_LOCAL_RESPONSE_NORMALIZATION_H
#define _DECAF_LOCAL_RESPONSE_NORMALIZATION_H

// C wrap functions.
extern "C" {

void lrn_forward(const int len, const void* bottom, void* top, void* scale,
        const int num_data, const int channels, const int size,
        const double k, const double alpha, const double beta,
        const int threads);

void lrn_backward(const int len, const void* bottom, const void* top,
        void* bottom_diff, const void* top_diff, const void* scale,
        const int num_data, const int channels, const int size,
        const double k, const double alpha, const double beta,
        const int threads);

} // extern "C"

#endif // _DECAF_LOCAL_RESPONSE_NORMALIZATION_H
