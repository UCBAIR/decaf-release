from decaf import base
from decaf.layers import dropout
from decaf.layers import fillers
import numpy as np
import unittest

class TestDropout(unittest.TestCase):
    def testdropoutlayer(self):
        layer = dropout.DropoutLayer(name='dropout', ratio=0.5)
        np.random.seed(1701)
        filler = fillers.RandFiller(min=1, max=2)
        bottom = base.Blob((100,4), filler=filler)
        top = base.Blob()
        # run the dropout layer
        layer.forward([bottom], [top])
        # simulate a diff
        fillers.RandFiller().fill(top.init_diff())
        layer.backward([bottom], [top], True)
        np.testing.assert_array_equal(top.data()[top.data()!=0] * 0.5,
                                      bottom.data()[top.data()!=0])
        np.testing.assert_array_equal(bottom.diff()[top.data() == 0],
                                      0)
        np.testing.assert_array_equal(bottom.diff()[top.data() != 0],
                                      top.diff()[top.data() != 0] * 2.)
        # test if debug_freeze works
        layer = dropout.DropoutLayer(name='dropout', ratio=0.5,
                                     debug_freeze=True)
        layer.forward([bottom], [top])
        snapshot = top.data().copy()
        layer.forward([bottom], [top])
        np.testing.assert_array_equal(snapshot, top.data())


if __name__ == '__main__':
    unittest.main()
