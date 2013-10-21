from decaf import base
from decaf.layers import fillers
import numpy as np
import unittest

class TestFillers(unittest.TestCase):
    def testXavierFiller(self):
        np.random.seed(1701)
        filler = fillers.XavierFiller()
        mat = np.empty((100, 10))
        filler.fill(mat)
        scale = np.sqrt(3. / 100.)
        self.assertGreaterEqual(mat.min(), -scale)
        self.assertLessEqual(mat.max(), scale)
        self.assertLessEqual(mat.min(), -scale * 0.9)
        self.assertGreaterEqual(mat.max(), scale * 0.9)
        mat = np.empty((20, 5, 10))
        filler.fill(mat)
        self.assertGreaterEqual(mat.min(), -scale)
        self.assertLessEqual(mat.max(), scale)
        self.assertLessEqual(mat.min(), -scale * 0.9)
        self.assertGreaterEqual(mat.max(), scale * 0.9)

    def testXavierGaussianFiller(self):
        np.random.seed(1701)
        mat = np.empty((100, 1000))
        mat_ref = np.empty((100,1000))
        fillers.XavierGaussianFiller().fill(mat)
        fillers.XavierFiller().fill(mat_ref)
        self.assertAlmostEqual(mat.std(), mat_ref.std(), places=3)

if __name__ == '__main__':
    unittest.main()
