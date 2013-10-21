"""Implements the LBFGS solver."""

from decaf import base
from decaf.util import mpi
import logging
from scipy import optimize

_FMIN = optimize.fmin_l_bfgs_b

class LBFGSSolver(base.Solver):
    """The LBFGS solver.
    
    This solver heavily relies on scipy's optimize toolbox (specifically,
    fmin_l_bfgs_b) so we write it differently from the other solvers. When
    faced with very large-scale problems, the additional memory overhead of
    LBFGS may make the method inapplicable, in which you may want to use the
    stochastic solvers. Also, although LBFGS solver supports mpi based
    optimizations, due to the network communication overhead you may still be
    better off using stochastic solvers with large-scale problems..
    """
    
    def __init__(self, **kwargs):
        """The LBFGS solver. Necessary args is:
            lbfgs_args: a dictionary containg the parameters to be passed
                to lbfgs.
        """
        base.Solver.__init__(self, **kwargs)
        self._lbfgs_args = self.spec.get('lbfgs_args', {})
        self._param = None
        self._decaf_net = None
        self._previous_net = None

    def _collect_params(self, realloc=False):
        """Collect the network parameters into a long vector.
        """
        params_list = self._decaf_net.params()
        if self._param is None or realloc:
            total_size = sum(p.data().size for p in params_list)
            dtype = max(p.data().dtype for p in params_list)
            self._param = base.Blob(shape=total_size, dtype=dtype)
            self._param.init_diff()
        current = 0
        collected_param = self._param.data()
        collected_diff = self._param.diff()
        for param in params_list:
            size = param.data().size
            collected_param[current:current+size] = param.data().flat
            # If we are computing mpi, we will need to reduce the diff.
            diff = param.diff()
            if mpi.SIZE > 1:
                part = collected_diff[current:current+size]
                part.shape = diff.shape
                mpi.COMM.Allreduce(diff, part)
            else:
                collected_diff[current:current+size] = diff.flat
            current += size

    def _distribute_params(self):
        """Distribute the parameter to the net.
        """
        params_list = self._decaf_net.params()
        current = 0
        for param in params_list:
            size = param.data().size
            param.data().flat = self._param.data()[current:current+size]
            current += size

    def obj(self, variable):
        """The objective function that wraps around the net."""
        self._param.data()[:] = variable
        self._distribute_params()
        loss = self._decaf_net.forward_backward(self._previous_net)
        if mpi.SIZE > 1:
            loss = mpi.COMM.allreduce(loss)
        self._collect_params()
        return loss, self._param.diff()

    def solve(self, decaf_net, previous_net=None):
        """Solves the net."""
        # first, run an execute pass to initialize all the parameters.
        self._decaf_net = decaf_net
        self._previous_net = previous_net
        initial_loss = self._decaf_net.forward_backward(self._previous_net)
        logging.info('Initial loss: %f.', initial_loss)
        logging.info('(Under mpirun, the given loss will just be an estimate'
                     ' on the root node.)')
        if mpi.SIZE > 1:
            params = self._decaf_net.params()
            for param in params:
                mpi.COMM.Bcast(param.data())
        self._collect_params(True)
        # now, run LBFGS
        # pylint: disable=W0108
        result = _FMIN(lambda x: self.obj(x), self._param.data(), 
                       **self._lbfgs_args)
        # put the optimized result to the net.
        self._param.data()[:] = result[0]
        self._distribute_params()
        logging.info('Final function value: %f.', result[1])

