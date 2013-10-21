from decaf.util import blasdot
import numpy as np
import unittest

class TestBlasdot(unittest.TestCase):
    """Test the blasdot module
    """
    def setUp(self, dtype=None):
        self.test_matrices = [
            (np.random.rand(1,1),
             np.random.rand(1,1)),
            (np.random.rand(1,5),
             np.random.rand(5,1)),
            (np.random.rand(5,1),
             np.random.rand(1,5)),
            (np.random.rand(5,5), 
             np.random.rand(5,5)),
            (np.random.rand(4,5),
             np.random.rand(5,4)),
            (np.random.rand(5,4),
             np.random.rand(4,5))]
        if dtype:
            self.test_matrices = [(m[0].astype(dtype), m[1].astype(dtype))
                                  for m in self.test_matrices]
        # Add the order-f case
        self.test_matrices += \
            [(a.copy(order='F'), b) for a, b in self.test_matrices] + \
            [(a, b.copy(order='F')) for a, b in self.test_matrices] + \
            [(a.copy(order='F'), b.copy(order='F'))
             for a, b in self.test_matrices]
        # Add explicit transpose
        self.test_matrices += [(b.T, a.T) for a,b in self.test_matrices]

    def testdot(self):
        for A, B in self.test_matrices:
            result = blasdot.dot(A, B)
            result_ref = np.dot(A,B)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)
    
    def testdot_with_out(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B)
            # c order
            result = np.empty(result_ref.shape, dtype = A.dtype)
            blasdot.dot(A, B, out = result)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)
            # f order
            result = np.empty(result_ref.shape, dtype = A.dtype, order='f')
            blasdot.dot(A, B, out = result)
            self.assertTrue(result.flags.f_contiguous)
            np.testing.assert_array_almost_equal(result, result_ref)


@unittest.skipIf(not blasdot._HAS_GPU, 
                 'No cuda gpu found.')
class TestBlasdotGPU(TestBlasdot):
    def setUp(self):
        blasdot.switch_backend('gpu')
        TestBlasdot.setUp(self, dtype=np.float32)
        
if __name__ == '__main__':
    unittest.main()

