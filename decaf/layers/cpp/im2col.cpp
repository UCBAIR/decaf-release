// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "im2col.h"

template <typename Dtype>
inline void _im2col_forward(const Dtype* data_im,
        Dtype* data_col,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride) {
    // The naive im2col_forward_mc implementation
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    for (int idxh = 0; idxh < height_col; ++idxh) {
        Dtype* pointer_col = data_col + idxh * width_col * psize * step_col;
        for (int idxw = 0; idxw < width_col; ++idxw) {
            // copy image[idxh:idxh+psize, idxw:idxw+psize, :]
            int hstart = idxh * stride;
            const Dtype* pointer_im = data_im + (hstart * width + idxw * stride) * nchannels;
            for (int i = hstart; i < hstart + psize; ++i) {
                // copy image[i, idxw:idxw+psize, :]
                memcpy(pointer_col, pointer_im, sizeof(Dtype) * step_col);
                pointer_col += step_col;
                pointer_im += step_im;
            }
        }
    }
} // im2col_forward


template <typename Dtype>
inline void _im2col_backward(Dtype* data_im,
        const Dtype* data_col,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride) {
    memset(data_im, 0, sizeof(Dtype) * height * width * nchannels);
    int step_im = width * nchannels;
    int step_col = psize * nchannels;
    int height_col = (height - psize) / stride + 1;
    int width_col = (width - psize) / stride + 1;
    for (int idxh = 0; idxh < height_col; ++idxh) {
        const Dtype* pointer_col = data_col + idxh * width_col * psize * step_col;
        for (int idxw = 0; idxw < width_col; ++idxw) {
            // copy image[idxh:idxh+psize, idxw:idxw+psize, :]
            int hstart = idxh * stride;
            Dtype* pointer_im = data_im + (hstart * width + idxw * stride) * nchannels;
            for (int i = hstart; i < hstart + psize; ++i) {
                // Add image[i, idxw:idxw+psize, :]
                for (int j = 0; j < step_col; ++j) {
                    pointer_im[j] += pointer_col[j];
                }
                pointer_col += step_col;
                pointer_im += step_im;
            }
        }
    }
} // im2col_backward


extern "C" {

void im2col_forward(const int len,
        const void* data_im,
        void* data_col,
        const int num,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride) {
    const int height_col = (height - psize) / stride + 1;
    const int width_col = (width - psize) / stride + 1;
    const int image_step = height * width * nchannels;
    const int col_step = height_col * width_col * psize * psize * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _im2col_forward<float>(
                    ((const float*)data_im) + image_step * i,
                    ((float*)data_col) + col_step * i,
                    height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _im2col_forward<double>(
                    ((const double*)data_im) + image_step * i,
                    ((double*)data_col) + col_step * i,
                    height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    }
}

void im2col_backward(const int len,
        void* data_im,
        const void* data_col,
        const int num,
        const int height,
        const int width,
        const int nchannels,
        const int psize,
        const int stride) {
    const int height_col = (height - psize) / stride + 1;
    const int width_col = (width - psize) / stride + 1;
    const int image_step = height * width * nchannels;
    const int col_step = height_col * width_col * psize * psize * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _im2col_backward<float>(
                    ((float*)data_im) + image_step * i,
                    ((const float*)data_col) + col_step * i,
                    height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _im2col_backward<double>(
                    ((double*)data_im) + image_step * i,
                    ((const double*)data_col) + col_step * i,
                    height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    }
}

} // extern "C"
