# pylint: disable=C0103
"""Efficient dot functions by calling the basic blas functions from scipy."""

import numpy as np
from scipy.linalg.blas import fblas

# pylint: disable=R0912
def _gemm_f_contiguous(alpha, A, B, out):
    '''A gemm function that uses scipy fblas functions, avoiding matrix copy
    when the input is transposed.
    
    The returned matrix is designed to be F_CONTIGUOUS.
    '''
    if out.shape != (A.shape[0], B.shape[1]):
        raise ValueError("Incorrect output shape.")
    if out.dtype != A.dtype:
        raise ValueError("Incorrect output dtype.")
    if not out.flags.f_contiguous:
        raise ValueError("Output is not f-contiguous.")
    if A.dtype != B.dtype:
        raise TypeError('The data type of the matrices should be the same.')
    if A.dtype == np.float32:
        gemm = fblas.sgemm
    elif A.dtype == np.float64:
        gemm = fblas.dgemm
    else:
        raise TypeError('Unfit data type.')
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not aligned")
    if A.flags.c_contiguous:
        A = A.T
        trans_a = True
    elif A.flags.f_contiguous:
        trans_a = False
    else:
        raise ValueError('Incorrect matrix flags for A.')
    if B.flags.c_contiguous:
        B = B.T
        trans_b = True
    elif B.flags.f_contiguous:
        trans_b = False
    else:
        raise ValueError('Incorrect matrix flags for B.')
    gemm(alpha, a=A, b=B, trans_a=trans_a, trans_b=trans_b, c=out,
         overwrite_c=True)
    return out

def _gemm_c_contiguous(alpha, A, B, out):
    """A wrapper that computes C_CONTIGUOUS gemm results."""
    _gemm_f_contiguous(alpha, B.T, A.T, out=out.T)
    return out
