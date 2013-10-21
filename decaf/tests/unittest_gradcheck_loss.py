from decaf import base
from decaf.layers import core_layers, fillers, regularization
from decaf.util import gradcheck
import numpy as np
import unittest


class TestLossGrad(unittest.TestCase):
    def setUp(self):
        pass

    def _testWeight(self, layer, input_blobs):
        layer.forward(input_blobs, [])
        loss = layer.backward(input_blobs, [], True)
        layer.spec['weight'] = layer.spec['weight'] / 2
        layer.forward(input_blobs, [])
        self.assertAlmostEqual(
            layer.backward(input_blobs, [], True),
            loss / 2.)
        layer.spec['weight'] = layer.spec['weight'] * 2


    def testSquaredLossGrad(self):
        np.random.seed(1701)
        shapes = [(4,3), (1,10), (4,3,2)]
        layer = core_layers.SquaredLossLayer(name='squared')
        checker = gradcheck.GradChecker(1e-6)
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            target_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
            result = checker.check(layer, [input_blob,target_blob], [],
                                   check_indices = [0])
            print(result)
            self.assertTrue(result[0])
            # also, check if weight works.
            self._testWeight(layer, [input_blob, target_blob])

    def testLogisticLossGrad(self):
        np.random.seed(1701)
        layer = core_layers.LogisticLossLayer(name='logistic')
        checker = gradcheck.GradChecker(1e-6)
        input_blob = base.Blob((10,1), filler=fillers.GaussianRandFiller())
        target_blob = base.Blob((10,), dtype=np.int,
                                filler=fillers.RandIntFiller(high=2))
        result = checker.check(layer, [input_blob,target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # also, check if weight works.
        self._testWeight(layer, [input_blob, target_blob])
    
    def testAutoencoderLossGrad(self):
        np.random.seed(1701)
        shapes = [(4,3), (1,10), (4,3,2)]
        layer = core_layers.AutoencoderLossLayer(name='loss', ratio=0.5)
        checker = gradcheck.GradChecker(1e-5)
        for shape in shapes:
            input_blob = base.Blob(shape, filler=fillers.RandFiller(min=0.05, max=0.95))
            result = checker.check(layer, [input_blob], [])
            print(result)
            self.assertTrue(result[0])
            # also, check if weight works.
            self._testWeight(layer, [input_blob])

    def testMultinomialLogisticLossGrad(self):
        np.random.seed(1701)
        layer = core_layers.MultinomialLogisticLossLayer(name='loss')
        checker = gradcheck.GradChecker(1e-6)
        shape = (10,5)
        # check index input
        input_blob = base.Blob(shape, filler=fillers.GaussianRandFiller())
        target_blob = base.Blob(shape[:1], dtype=np.int,
                                filler=fillers.RandIntFiller(high=shape[1]))
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # also, check if weight works.
        self._testWeight(layer, [input_blob, target_blob])
        
        # check full input
        target_blob = base.Blob(shape, filler=fillers.RandFiller())
        # normalize target
        target_data = target_blob.data()
        target_data /= target_data.sum(1)[:, np.newaxis]
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # also, check if weight works.
        self._testWeight(layer, [input_blob, target_blob])

    def testKLDivergenceLossGrad(self):
        np.random.seed(1701)
        layer = core_layers.KLDivergenceLossLayer(name='loss')
        checker = gradcheck.GradChecker(1e-6)
        shape = (4,5)
        # For the input, we make sure it is not too close to 0 (which would
        # create numerical issues).
        input_blob = base.Blob(shape,
                               filler=fillers.RandFiller(min=0.1, max=0.9))
        # normalize input blob
        input_data = input_blob.data()
        input_data /= input_data.sum(1)[:, np.newaxis]
        # check index input
        target_blob = base.Blob(shape[:1], dtype=np.int,
                                filler=fillers.RandIntFiller(high=shape[1]))
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # also, check if weight works.
        self._testWeight(layer, [input_blob, target_blob])
        
        # check full input
        target_blob = base.Blob(shape, filler=fillers.RandFiller())
        # normalize target
        target_data = target_blob.data()
        target_data /= target_data.sum(1)[:, np.newaxis]
        result = checker.check(layer, [input_blob, target_blob], [],
                               check_indices = [0])
        print(result)
        self.assertTrue(result[0])
        # also, check if weight works.
        self._testWeight(layer, [input_blob, target_blob])

if __name__ == '__main__':
    unittest.main()
