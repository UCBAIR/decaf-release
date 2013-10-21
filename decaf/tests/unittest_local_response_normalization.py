from decaf.layers.cpp import wrapper
import logging
import numpy as np
import numpy.testing as npt
import unittest


class TestLRN(unittest.TestCase):
    def setUp(self):
        pass
    
    def reference_forward_implementation(self, features, size, k, alpha, beta):
        """A reference implementation of the local response normalization."""
        num_data = features.shape[0]
        channels = features.shape[1]
        output = np.zeros_like(features)
        scale = np.zeros_like(features)
        for n in range(num_data):
            for c in range(channels):
                local_start = c - (size - 1) / 2
                local_end = local_start + size
                local_start = max(local_start, 0)
                local_end = min(local_end, channels)
                scale[n, c] = k + \
                    (features[n, local_start:local_end]**2).sum() * \
                    alpha / size
                output[n, c] = features[n, c] / (scale[n, c] ** beta)
        return output, scale

    def testLocalResponseNormalizationForward(self):
        np.random.seed(1701)
        dtypes = [np.float32, np.float64]
        for dtype in dtypes:
            features = np.random.rand(5, 10).astype(dtype)
            output = np.random.rand(5, 10).astype(dtype)
            scale = np.random.rand(5, 10).astype(dtype)
            # odd size, k = 1
            wrapper.lrn_forward(features, output, scale, 5, 1., 1.5, 0.75)
            output_ref, scale_ref = self.reference_forward_implementation(
                features, 5, 1., 1.5, 0.75)
            np.testing.assert_array_almost_equal(output, output_ref)
            np.testing.assert_array_almost_equal(scale, scale_ref)
            # odd size, k = 2
            wrapper.lrn_forward(features, output, scale, 5, 2., 1.5, 0.75)
            output_ref, scale_ref = self.reference_forward_implementation(
                features, 5, 2., 1.5, 0.75)
            np.testing.assert_array_almost_equal(output, output_ref)
            np.testing.assert_array_almost_equal(scale, scale_ref)
            # even size
            wrapper.lrn_forward(features, output, scale, 6, 1., 1.5, 0.75)
            output_ref, scale_ref = self.reference_forward_implementation(
                features, 6, 1., 1.5, 0.75)
            np.testing.assert_array_almost_equal(output, output_ref)
            np.testing.assert_array_almost_equal(scale, scale_ref)

if __name__ == '__main__':
    unittest.main()
