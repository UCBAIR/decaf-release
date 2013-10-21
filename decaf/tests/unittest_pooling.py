from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestPoolingGrad(unittest.TestCase):
    def setUp(self):
        pass

    def testPoolingGrad(self):
        np.random.seed(1701)
        output_blob = base.Blob()
        input_blob = base.Blob((1,8,8,3), filler=fillers.GaussianRandFiller())
        psize = 3
        stride = 2
        mode = 'max'
        layer = core_layers.PoolingLayer(
            name='pool', psize=psize, stride=stride, mode=mode)
        layer.forward([input_blob], [output_blob])
        img = input_blob.data()[0]
        output = output_blob.data()[0]
        print img.shape, output.shape
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for c in range(output.shape[2]):
                    self.assertAlmostEqual(
                        output[i,j,c],
                        img[i*stride:i*stride+psize,
                            j*stride:j*stride+psize,
                            c].max())
        mode = 'ave'
        layer = core_layers.PoolingLayer(
            name='pool', psize=psize, stride=stride, mode=mode)
        layer.forward([input_blob], [output_blob])
        img = input_blob.data()[0]
        output = output_blob.data()[0]
        print img.shape, output.shape
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for c in range(output.shape[2]):
                    self.assertAlmostEqual(
                        output[i,j,c],
                        img[i*stride:i*stride+psize,
                            j*stride:j*stride+psize,
                            c].mean())


if __name__ == '__main__':
    unittest.main()
