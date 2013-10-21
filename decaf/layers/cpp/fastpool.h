#ifndef _DECAF_FASTPOOL_H
#define _DECAF_FASTPOOL_H

#define EQUAL_THRESHOLD 1e-8


extern "C" {

void maxpooling_forward(const int len,
        const void* image, void* pooled, const int num, 
        const int height, const int width,
        const int nchannels, const int psize, const int stride);

void maxpooling_backward(const int len,
        const void* image, const void* pooled, void* image_grad,
        const void* pooled_grad, const int num,
        const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_forward(const int len,
        const void* image, void* pooled, const int num,
        const int height, const int width,
        const int nchannels, const int psize, const int stride);

void avepooling_backward(const int len,
        void* image_grad, const void* pooled_grad, 
        const int num, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride);

} // extern "C"

#endif // _DECAF_FASTPOOL_H
