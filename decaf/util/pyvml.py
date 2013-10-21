"""A module to wrap over vml functions. Requires mkl runtime."""

import ctypes as ct
import logging
import numpy as np

_VML_2VECS = ['Sqr', 'Mul', 'Abs', 'Inv', 'Sqrt', 'InvSqrt', 'Cbrt', 'InvCbrt',
             'Pow2o3', 'Pow3o2', 'Exp', 'Expm1', 'Ln', 'Log10', 'Log1p', 'Cos',
             'Sin', 'Tan', 'Acos', 'Asin', 'Atan', 'Cosh', 'Sinh', 'Tanh',
             'Acosh', 'Asinh', 'Atanh', 'Erf', 'Erfc', 'CdfNorm', 'ErfInv',
             'ErfcInv', 'CdfNormInv']
_VML_3VECS = ['Add', 'Sub', 'Div', 'Pow', 'Hypot', 'Atan2']


def vml_dtype_wrapper(func_name):
    """For a function that has two input types, this function creates a
    wrapper that directs to the correct function.
    """
    float_func = getattr(_DLL, 'vs' + func_name)
    double_func = getattr(_DLL, 'vd' + func_name)
    def _wrapped_func(*args):
        """A function that calls vml functions with checked dtypes."""
        dtype = args[0].dtype
        size = args[0].size
        if not all(arg.dtype == dtype for arg in args):
            raise ValueError('Args should have the same dtype.')
        if not all(arg.size == size for arg in args):
            raise ValueError('Args should have the same size.')
        if not (all(arg.flags.c_contiguous for arg in args) or
                all(arg.flags.f_contiguous for arg in args)):
            raise ValueError('Args should be contiguous in the same way.')
        if args[0].dtype == np.float32:
            return float_func(size, *args)
        elif args[0].dtype == np.float64:
            return double_func(size, *args)
        else:
            raise TypeError('Unsupported type: ' + str(dtype))
    return _wrapped_func


def _set_dll_funcs():
    """Set the restypes and argtypes of the functions."""
    for func_name in _VML_2VECS + _VML_3VECS:
        getattr(_DLL, 'vs' + func_name).restype = None
        getattr(_DLL, 'vd' + func_name).restype = None
    for func_name in _VML_2VECS:
        getattr(_DLL, 'vs' + func_name).argtypes = [
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32)]
        getattr(_DLL, 'vd' + func_name).argtypes = [
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64)]
    for func_name in _VML_3VECS:
        getattr(_DLL, 'vs' + func_name).argtypes = [
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32)]
        getattr(_DLL, 'vd' + func_name).argtypes = [
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.float64)]

#################################################
# The main pyvml routine.
#################################################
try:
    # Try to load the mkl dynamic library. This does not support windows yet.
    try:
        _DLL = ct.CDLL('libmkl_rt.so')
    except OSError:
        _DLL = ct.CDLL('libmkl_rt.dylib')
except OSError:
    logging.warning('decaf.util.pyvml: unable to load the mkl library. '
                    'Using fallback options.')
    # implement necessary fallback options.
    # Yangqing's note: I am not writing all the fallbacks, only the necessary
    # ones are provided.
    # pylint: disable=C0103
    Exp = lambda x, y: np.exp(x, out=y)
    Ln = lambda x, y: np.log(x, out=y)
else:
    _set_dll_funcs()
    for name in _VML_2VECS + _VML_3VECS:
        globals()[name] = vml_dtype_wrapper(name)
