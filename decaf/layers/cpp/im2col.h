#ifndef _DECAF_IM2COL_H
#define _DECAF_IM2COL_H

extern "C" {

void im2col_forward(const int len,
        const void* data_im,
        void* data_col,
        const int num,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride);

void im2col_backward(const int len,
        void* data_im,
        const void* data_col,
        const int num,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride);

} // extern "C"

#endif // _DECAF_IM2COL_H
