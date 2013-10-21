from decaf import base
from decaf.layers import identity, fillers
import numpy as np
import unittest

class TestIdentity(unittest.TestCase):
    def testIdentityLayer(self):
        layer = identity.IdentityLayer(name='identity')
        np.random.seed(1701)
        filler = fillers.RandFiller()
        bottom = base.Blob((100,4), filler=filler)
        top = base.Blob()
        # run the dropout layer
        layer.forward([bottom], [top])
        # simulate a diff
        fillers.RandFiller().fill(top.init_diff())
        layer.backward([bottom], [top], True)
        np.testing.assert_array_equal(top.data(), bottom.data())
        np.testing.assert_array_equal(top.diff(), bottom.diff())

if __name__ == '__main__':
    unittest.main()
