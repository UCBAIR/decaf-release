#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "fastpool.h"

using std::max;
using std::min;

template <typename Dtype>
inline void _maxpooling_forward(
        const Dtype* image, Dtype* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    for (int i = 0; i < pooled_height * pooled_width * nchannels; ++i) {
        pooled[i] = -FLT_MAX;
    }
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            const Dtype* p_image = image + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_pooled[c] = max(p_pooled[c], p_image[c]);
                    }
                }
            }
        } // loop over width
    } // loop over height
}


template <typename Dtype>
inline void _maxpooling_backward(
        const Dtype* image, const Dtype* pooled, Dtype* image_grad,
        const Dtype* pooled_grad, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    memset(image_grad, 0, sizeof(Dtype) * height * width * nchannels);
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Processing pixel at [i,j].
            // First, compute the pooling region
            int h_start = (i < psize) ? 0 : (i - psize) / stride + 1;
            int h_end = min(i / stride + 1, pooled_height);
            int w_start = (j < psize) ? 0 : (j - psize) / stride + 1;
            int w_end = min(j / stride + 1, pooled_width);
            const Dtype* p_image = image + (i * width + j) * nchannels;
            Dtype* p_image_grad = image_grad + (i * width + j) * nchannels;
            for (int ph = h_start; ph < h_end; ++ph) {
                for (int pw = w_start; pw < w_end; ++pw) {
                    const Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
                    const Dtype* p_pooled_grad = pooled_grad + 
                        (ph * pooled_width + pw) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_image_grad[c] += p_pooled_grad[c] * (p_image[c] >= p_pooled[c]);
                    }
                }
            }
        } // loop over width
    } // loop over height
}


template <typename Dtype>
inline void _avepooling_forward(
        const Dtype* image, Dtype* pooled, const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    memset(pooled, 0, sizeof(Dtype) * pooled_height * pooled_width * nchannels);
    for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
            int h_start = ph * stride;
            int h_end = min(height, h_start + psize);
            int w_start = pw * stride;
            int w_end = min(width, w_start + psize);
            Dtype* p_pooled = pooled + (ph * pooled_width + pw) * nchannels;
            for (int i = h_start; i < h_end; ++i) {
                for (int j = w_start; j < w_end; ++j) {
                    const Dtype* p_image = image + (i * width + j) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_pooled[c] += p_image[c];
                    }
                }
            }
            // normalize
            Dtype scale = 1. / Dtype((h_end - h_start) * (w_end - w_start));
            for (int c = 0; c < nchannels; ++c) {
                p_pooled[c] *= scale;
            }
        }
    }
}

template <typename Dtype>
inline void _avepooling_backward(
        Dtype* image_grad, const Dtype* pooled_grad, const int height, 
        const int width, const int nchannels, const int psize,
        const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    memset(image_grad, 0, sizeof(Dtype) * height * width * nchannels);
    // This code is written in a forward mode: we go through the pixels once,
    // and write to all the pooled regions that it maps to.
    for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
            int h_start = ph * stride;
            int h_end = min(height, h_start + psize);
            int w_start = pw * stride;
            int w_end = min(width, w_start + psize);
            Dtype scale = 1. / Dtype((h_end - h_start) * (w_end - w_start));
            const Dtype* p_pooled_grad = pooled_grad
                    + (ph * pooled_width + pw) * nchannels;
            for (int i = h_start; i < h_end; ++i) {
                for (int j = w_start; j < w_end; ++j) {
                    Dtype* p_image_grad = image_grad + (i * width + j) * nchannels;
                    for (int c = 0; c < nchannels; ++c) {
                        p_image_grad[c] += p_pooled_grad[c] * scale;
                    }
                }
            }
        }
    }
}

extern "C" {

void maxpooling_forward(const int len,
        const void* image, void* pooled, const int num,
        const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    int image_step = height * width * nchannels;
    int pooled_step = pooled_height * pooled_width * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _maxpooling_forward<float>(
                ((const float*)image) + image_step * i, 
                ((float*)pooled) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _maxpooling_forward<double>(
                ((const double*)image) + image_step * i, 
                ((double*)pooled) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
}

void maxpooling_backward(const int len,
        const void* image, const void* pooled, void* image_grad,
        const void* pooled_grad, const int num,
        const int height, const int width,
        const int nchannels, const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    int image_step = height * width * nchannels;
    int pooled_step = pooled_height * pooled_width * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _maxpooling_backward<float>(
                ((const float*)image) + image_step * i,
                ((const float*)pooled) + pooled_step * i,
                ((float*)image_grad) + image_step * i,
                ((const float*)pooled_grad) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _maxpooling_backward<double>(
                ((const double*)image) + image_step * i,
                ((const double*)pooled) + pooled_step * i,
                ((double*)image_grad) + image_step * i,
                ((const double*)pooled_grad) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
}

void avepooling_forward(const int len,
        const void* image, void* pooled, const int num,
        const int height, const int width,
        const int nchannels, const int psize, const int stride){
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    int image_step = height * width * nchannels;
    int pooled_step = pooled_height * pooled_width * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _avepooling_forward<float>(
                ((const float*)image) + image_step * i, 
                ((float*)pooled) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _avepooling_forward<double>(
                ((const double*)image) + image_step * i, 
                ((double*)pooled) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
}

void avepooling_backward(const int len,
        void* image_grad, const void* pooled_grad, const int num, 
        const int height, const int width, const int nchannels,
        const int psize, const int stride) {
    int pooled_height = int(ceil(float(height - psize) / stride)) + 1;
    int pooled_width = int(ceil(float(width - psize) / stride)) + 1;
    int image_step = height * width * nchannels;
    int pooled_step = pooled_height * pooled_width * nchannels;
    switch(len) {
    case sizeof(float):
        for (int i = 0; i < num; ++i) {
            _avepooling_backward<float>(
                ((float*)image_grad) + image_step * i, 
                ((const float*)pooled_grad) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    case sizeof(double):
        for (int i = 0; i < num; ++i) {
            _avepooling_backward<double>(
                ((double*)image_grad) + image_step * i, 
                ((const double*)pooled_grad) + pooled_step * i,
                height, width, nchannels, psize, stride);
        }
        break;
    default:
        exit(EXIT_FAILURE);
    } // switch(len)
}

} // extern "C"

