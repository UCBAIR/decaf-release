from decaf import base
from decaf.layers import core_layers
from decaf.layers import fillers
from decaf.opt import core_solvers
from decaf.util import mpi
import numpy as np
import unittest

class TestLBFGS(unittest.TestCase):
    """Test the blasdot module
    """

    def _testSolver(self, solver):
        # We are going to test if the solver correctly deals with the mpi case
        # where multiple nodes host different data. To this end we will
        # create a dummy regression problem which, when run under mpi with
        # >1 nodes, will create a different result from a single-node run.
        np.random.seed(1701)
        X = base.Blob((10, 1),
                      filler=fillers.GaussianRandFiller(
                          mean=mpi.RANK, std=0.01))
        Y = base.Blob((10, 1),
                      filler=fillers.ConstantFiller(value=mpi.RANK + 1.))
        decaf_net = base.Net()
        decaf_net.add_layer(core_layers.InnerProductLayer(name='ip',
                                                          num_output=1),
                            needs='X', provides='pred')
        decaf_net.add_layer(core_layers.SquaredLossLayer(name='loss'),
                            needs=['pred','Y'])
        decaf_net.finish()
        solver.solve(decaf_net, previous_net = {'X': X, 'Y': Y})
        w, b = decaf_net.layers['ip'].param()
        print w.data(), b.data()
        if mpi.SIZE == 1:
            # If size is 1, we are fitting y = 0 * x + 1
            np.testing.assert_array_almost_equal(w.data(), 0., 2)
            np.testing.assert_array_almost_equal(b.data(), 1., 2)
        else:
            # if size is not 1, we are fitting y = x + 1
            np.testing.assert_array_almost_equal(w.data(), 1., 2)
            np.testing.assert_array_almost_equal(b.data(), 1., 2)
        self.assertTrue(True)

    def testLBFGS(self):
        # create solver
        solver = core_solvers.LBFGSSolver(
            lbfgs_args={'disp': 0, 'pgtol': 1e-8})
        self._testSolver(solver)

    def testSGD(self):
        # create solver
        solver = core_solvers.SGDSolver(
            base_lr=1., max_iter=1000, lr_policy='inv',
            gamma=0.1, momentum=0.5)
        self._testSolver(solver)

    def testAdagrad(self):
        # create solver
        solver = core_solvers.AdagradSolver(
            base_lr=1., max_iter=1000, base_accum=1.e-8)
        self._testSolver(solver)


if __name__ == '__main__':
    unittest.main()
