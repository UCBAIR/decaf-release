# pylint: disable=C0103
"""Efficient dot functions by calling the basic blas functions from scipy."""

import numpy as np

# import submodules that implements the blas functions
import _numpy_blasdot

# The default backend would be the numpy blasdot.
_gemm_f_contiguous = _numpy_blasdot._gemm_f_contiguous
_gemm_c_contiguous = _numpy_blasdot._gemm_c_contiguous


def dot(A, B, out=None):
    '''
    a simple wrapper that mimics np.dot (if A and B are both matrices!)
    This function solves the problem that np.dot copies matrices when
    working on transposed matrices.
    Input:
        A, B: two matrices. should be either c-contiguous or f-contiguous
        out: (optional) the output matrix. If it is passed, the matrix should
            have the right shape and should be C_CONTIGUOUS.
    Output:
        out: the output matrix
    Raises:
        TypeError, if the type of matrices is wrong.
    '''
    if out == None:
        out = np.empty((A.shape[0], B.shape[1]), A.dtype, B.dtype)
    # Numpy seems to have bugs dealing with the flags of 1x1 matrices. Thus,
    # if we encounter 1x1 matrices, we manually deal with the calculation.
    if out.size == 1:
        out[:] = np.dot(A.flat, B.flat)
    elif out.flags.f_contiguous:
        out = _gemm_f_contiguous(1.0, A, B, out=out)
    else:
        out = _gemm_c_contiguous(1.0, A, B, out=out)
    return out

def dot_lastdim(A, B, out=None):
    """Performs dot for multi-dimensional matrices A and B, where
    A.shape[-1] = B.shape[0]. The returned matrix should have shape
    A.shape[:-1] + B.shape[1:].

    A and B should both be c-contiguous, otherwise the code will report
    an error.
    """
    if out == None:
        out = np.empty(A.shape[:-1] + B.shape[1:], A.dtype)
    lda = A.size / A.shape[-1]
    dim = A.shape[-1]
    ldb = B.size / B.shape[0]
    # using views
    Aview = A.view()
    Bview = B.view()
    outview = out.view()
    Aview.shape = (lda, dim)
    Bview.shape = (dim, ldb)
    outview.shape = (lda, ldb)
    dot(Aview, Bview, outview)
    return out

def dot_firstdims(A, B, out=None):
    """Performs dot for multi-dimensional matrices A and B, where
    prod(A.shape[:-1]) = prod(B.shape[:-1]), and the result would be
    dot(A.T, B) where A and B are treated as 2-dimensional matrices with shape
    (prod(shape[:-1]), shape[-1]). The returned matrix should have shape
    (A.shape[-1], B.shape[-1]). The code is often encountered in computing the
    gradient in e.g. convolutions.
    
    A and B should both be c-contiguous, otherwise the code will report
    an error.
    """
    if out == None:
        out = np.empty((A.shape[-1], B.shape[-1]), A.dtype)
    lda = A.shape[-1]
    dim = A.size / A.shape[-1]
    ldb = B.shape[-1]
    Aview = A.view()
    Bview = B.view()
    Aview.shape = (dim, lda)
    Bview.shape = (dim, ldb)
    dot(Aview.T, Bview, out)
    return out


