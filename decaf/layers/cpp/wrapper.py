# pylint: disable=C0103
"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _DLL = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error
try:
    _OMP_NUM_THREADS=int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    try:
        import multiprocessing
        _OMP_NUM_THREADS=multiprocessing.cpu_count()
    except ImportError:
        _OMP_NUM_THREADS=1

################################################################################
# im2col operation
################################################################################
_DLL.im2col_forward.restype = _DLL.im2col_backward.restype = None

def im2col_forward(im, col, psize, stride):
    num, height, width, channels = im.shape
    _DLL.im2col_forward(ct.c_int(im.itemsize),
                im.ctypes.data_as(ct.c_void_p),
                col.ctypes.data_as(ct.c_void_p),
                ct.c_int(num),
                ct.c_int(height),
                ct.c_int(width),
                ct.c_int(channels),
                ct.c_int(psize),
                ct.c_int(stride))

def im2col_backward(im, col, psize, stride):
    num, height, width, channels = im.shape
    _DLL.im2col_backward(ct.c_int(im.itemsize),
                im.ctypes.data_as(ct.c_void_p),
                col.ctypes.data_as(ct.c_void_p),
                ct.c_int(num),
                ct.c_int(height),
                ct.c_int(width),
                ct.c_int(channels),
                ct.c_int(psize),
                ct.c_int(stride))

################################################################################
# pooling operation
################################################################################
_DLL.maxpooling_forward.restype = \
_DLL.maxpooling_backward.restype = \
_DLL.avepooling_forward.restype = \
_DLL.avepooling_backward.restype = None

def maxpooling_forward(image, pooled, psize, stride):
    num, height, width, channels = image.shape
    _DLL.maxpooling_forward(ct.c_int(image.itemsize),
                            image.ctypes.data_as(ct.c_void_p),
                            pooled.ctypes.data_as(ct.c_void_p),
                            ct.c_int(num),
                            ct.c_int(height),
                            ct.c_int(width),
                            ct.c_int(channels),
                            ct.c_int(psize),
                            ct.c_int(stride))

def avepooling_forward(image, pooled, psize, stride):
    num, height, width, channels = image.shape
    _DLL.avepooling_forward(ct.c_int(image.itemsize),
                            image.ctypes.data_as(ct.c_void_p),
                            pooled.ctypes.data_as(ct.c_void_p),
                            ct.c_int(num),
                            ct.c_int(height),
                            ct.c_int(width),
                            ct.c_int(channels),
                            ct.c_int(psize),
                            ct.c_int(stride))

def maxpooling_backward(image, pooled, image_diff, pooled_diff, psize,
                        stride):
    num, height, width, channels = image.shape
    _DLL.maxpooling_backward(ct.c_int(image.itemsize),
                             image.ctypes.data_as(ct.c_void_p),
                             pooled.ctypes.data_as(ct.c_void_p),
                             image_diff.ctypes.data_as(ct.c_void_p),
                             pooled_diff.ctypes.data_as(ct.c_void_p),
                             ct.c_int(num),
                             ct.c_int(height),
                             ct.c_int(width),
                             ct.c_int(channels),
                             ct.c_int(psize),
                             ct.c_int(stride))

def avepooling_backward(image_diff, pooled_diff, psize, stride):
    num, height, width, channels = image_diff.shape
    _DLL.avepooling_backward(ct.c_int(image_diff.itemsize),
                             image_diff.ctypes.data_as(ct.c_void_p),
                             pooled_diff.ctypes.data_as(ct.c_void_p),
                             ct.c_int(num),
                             ct.c_int(height),
                             ct.c_int(width),
                             ct.c_int(channels),
                             ct.c_int(psize),
                             ct.c_int(stride))



################################################################################
# local contrast normalization operation
################################################################################
_DLL.lrn_forward.restype = \
_DLL.lrn_backward.restype = None

def lrn_forward(bottom, top, scale, size, k, alpha, beta):
    _DLL.lrn_forward(ct.c_int(bottom.itemsize),
                     bottom.ctypes.data_as(ct.c_void_p),
                     top.ctypes.data_as(ct.c_void_p),
                     scale.ctypes.data_as(ct.c_void_p),
                     ct.c_int(bottom.size / bottom.shape[-1]),
                     ct.c_int(bottom.shape[-1]),
                     ct.c_int(size),
                     ct.c_double(k),
                     ct.c_double(alpha),
                     ct.c_double(beta),
                     ct.c_int(_OMP_NUM_THREADS))


def lrn_backward(bottom, top, bottom_diff, top_diff, scale, size, k, alpha,
                 beta):
    _DLL.lrn_backward(ct.c_int(bottom.itemsize),
                     bottom.ctypes.data_as(ct.c_void_p),
                     top.ctypes.data_as(ct.c_void_p),
                     bottom_diff.ctypes.data_as(ct.c_void_p),
                     top_diff.ctypes.data_as(ct.c_void_p),
                     scale.ctypes.data_as(ct.c_void_p),
                     ct.c_int(bottom.size / bottom.shape[-1]),
                     ct.c_int(bottom.shape[-1]),
                     ct.c_int(size),
                     ct.c_double(k),
                     ct.c_double(alpha),
                     ct.c_double(beta),
                     ct.c_int(_OMP_NUM_THREADS))

################################################################################
# local contrast normalization operation
################################################################################
_DLL.relu_forward.restype = None

def relu_forward(bottom, top):
    _DLL.relu_forward(ct.c_int(bottom.itemsize),
                      bottom.ctypes.data_as(ct.c_void_p),
                      top.ctypes.data_as(ct.c_void_p),
                      ct.c_int(bottom.size))
