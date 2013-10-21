"""Implements the stochastic solvers."""
import cPickle as pickle
from decaf import base
from decaf.util import mpi
from decaf.util.timer import Timer
import logging
import numpy as np
import os

class StochasticSolver(base.Solver):
    """The Basic stochastic solver."""
    
    def __init__(self, **kwargs):
        """Initializes the Stochastic solver.
        
        kwargs:
            base_lr: the base learning rate.
            disp: if 0, no information is displayed. If a positive number, the
                optimization status will be printed out every few iterations,
                with the interval given by disp.
            max_iter: the maximum number of iterations. Default 1000.
            snapshot_interval: the snapshot interval. Default 0.
            folder: the snapshot folder. Should be provided
                if snapshot_interval is not zero.
        """
        base.Solver.__init__(self, **kwargs)
        self._max_iter = self.spec.get('max_iter', 1000)
        self._snapshot_interval = self.spec.get('snapshot_interval', 0)
        if self._snapshot_interval > 0 and 'folder' not in self.spec:
            raise ValueError('You should provide a folder to write result to.')
        self._decaf_net = None
        self._previous_net = None
        self._iter_idx = None

    def initialize_status(self):
        """A function that specific stochastic solvers can override
        to perform necessary initialization, after the net is run for the
        first time to allocate all the intermediate variables and gives an
        initial loss. The default initialization will make all the parameters
        the same by broadcasting the parameters from root.
        """
        if mpi.SIZE > 1:
            params = self._decaf_net.params()
            for param in params:
                mpi.COMM.Bcast(param.data())
        return

    def compute_update_value(self):
        """An abstract function that specific stochastic solvers have to
        implement to determine the update value. The gradients can be obtained
        as [param.diff() for param in decaf_net.params()], and the algorithm
        should write the update values into the param.diff() fields. Note that
        the diff values are already the averaged version over the mpi nodes
        by the solver, so you don't need to rebroadcast them again.

        Input:
            decaf_net: the network.
            loss: the computed loss. A specific solver may not actually need
                the loss value, but we provide it here for logging purpose.
        """
        raise NotImplementedError
    
    def snapshot(self, is_final=False):
        """A function that specific stochastic solvers can override to provide
        snapshots of the current net as well as necessary other bookkeeping
        stuff. The folder will be the place that the snapshot should be written
        to, and the function should create a subfolder named as the iter_idx,
        and write any necessary information there.
        
        In default, the snapshot function will store the network using the 
        network's save function.
        """
        folder = self.spec['folder']
        if is_final:
            subfolder = os.path.join(folder, 'final')
        else:
            subfolder = os.path.join(folder, str(self._iter_idx))
        try:
            os.makedirs(subfolder)
        except OSError:
            pass
        self._decaf_net.save(
            os.path.join(subfolder, self._decaf_net.name + '.net'))
        # return the subfolder name that we will use for further processing.
        return subfolder

    def iter_callback(self, loss):
        """Iteration callback. Override this function if anything should be
        carried out after each iteration.
        """
        pass

    def solve(self, decaf_net, previous_net=None):
        """Solves the net."""
        # first, run a pass to initialize all the parameters.
        logging.info('StochasticSolver: precomputing.')
        self._iter_idx = 0
        self._decaf_net = decaf_net
        self._previous_net = previous_net
        initial_loss = decaf_net.forward_backward(self._previous_net)
        logging.info('StochasticSolver: initial loss: %f.', initial_loss)
        logging.info('(Under mpirun, the given loss will just be an estimate'
                     ' on the root node.)')
        self.initialize_status()
        # the main iteration
        timer = Timer()
        logging.info('StochasticSolver: started.')
        for _ in range(self._max_iter):
            if mpi.SIZE > 1:
                loss = mpi.COMM.allreduce(
                    decaf_net.forward_backward(self._previous_net)) / mpi.SIZE
                # we need to broadcast and average the parameters
                params = decaf_net.params()
                for param in params:
                    diff = param.diff()
                    diff_cache = diff.copy()
                    mpi.COMM.Allreduce(diff_cache, diff)
                    diff /= mpi.SIZE
            else:
                loss = decaf_net.forward_backward(self._previous_net)
            self.compute_update_value()
            decaf_net.update()
            if (mpi.is_root() and 
                self._snapshot_interval > 0 and self._iter_idx > 0 and
                self._iter_idx % self._snapshot_interval == 0):
                # perform snapshot.
                self.snapshot()
            if (self.spec.get('disp', 0) and 
                self._iter_idx % self.spec['disp']):
                logging.info('Iter %d, loss %f, elapsed %s', self._iter_idx,
                             loss, timer.lap())
            self.iter_callback(loss)
            self._iter_idx += 1
        # perform last snapshot.
        if mpi.is_root() and 'folder' in self.spec:
            self.snapshot(True)
        mpi.barrier()
        logging.info('StochasticSolver: finished. Total time %s.',
                     timer.total())


class SGDSolver(StochasticSolver):
    """The SGD solver.
    """
    
    def __init__(self, **kwargs):
        """Initializes the SGD solver.

        kwargs:
            base_lr: the base learning rate.
            max_iter: the maximum number of iterations. Default 1000.
            lr_policy: the learning rate policy. could be:
                'fixed': rate will always be base_lr.
                'exp': exponent decay - rate will be
                    base_lr * (gamma ^ t)
                'inv': rate will be base_lr / (1 + gamma * t)^power
                where t in the above equations are the epoch, starting from 0.
            min_lr: the minimun learning rate. Default 0. If weight decay
                results in a learning rate smaller than min_lr, it is set to
                min_lr.
            max_lr: the maximum learning rate. Default Inf.
            gamma: the gamma parameter, see lr_policy.
            power: the power parameter, see lr_policy. Default 1.
            momentum: the momentum value. Should be in the range [0,1).
                Default 0.
            asgd: if True, use average sgd (Polyak 1992).
            asgd_skip: the number of iterations to skip before averaging.
                Default 1 (http://leon.bottou.org/projects/sgd/).
        """
        StochasticSolver.__init__(self, **kwargs)
        self.spec['momentum'] = self.spec.get('momentum', 0)
        self.spec['asgd'] = self.spec.get('asgd', False)
        self.spec['asgd_skip'] = self.spec.get('asgd_skip', 1)
        self.spec['power'] = self.spec.get('power', 1)
        self.spec['max_lr'] = self.spec.get('max_lr', float('inf'))
        self.spec['min_lr'] = self.spec.get('min_lr', 0.)
        self._momentum = None
        self._asgd = None

    def _get_learningrate(self):
        """get the learning rate."""
        policy = self.spec['lr_policy']
        base_lr = self.spec['base_lr']
        if policy == 'fixed':
            learningrate = base_lr
        elif policy == 'exp':
            learningrate = base_lr * (self.spec['gamma'] ** self._iter_idx)
        elif policy == 'inv':
            learningrate = base_lr / ((1 + self.spec['gamma'] * self._iter_idx)
                                      ** self.spec['power'])
        return min(max(learningrate, self.spec['min_lr']), self.spec['max_lr'])
    
    def initialize_status(self):
        """Initializes the status."""
        StochasticSolver.initialize_status(self)
        if self.spec['momentum']:
            # we need to maintain the momentum history
            params = self._decaf_net.params()
            self._momentum = [np.zeros_like(p.data())
                              for p in params]
        if self.spec['asgd']:
            # we need to maintain the asgd param values
            params = self._decaf_net.params()
            self._asgd = [np.zeros_like(p.data())
                          for p in params]
        
    def compute_update_value(self):
        """Computes the update value by multiplying the gradient with the
        learning rate.
        """
        learningrate = self._get_learningrate()
        logging.debug('learning rate %f', learningrate)
        if self.spec['momentum'] > 0:
            # we need to add momentum terms and keep track of them.
            for momentum, param in zip(self._momentum,
                                       self._decaf_net.params()):
                momentum *= self.spec['momentum']
                diff = param.diff()
                diff *= learningrate
                diff += momentum
                momentum[:] = diff
        else:
            for param in self._decaf_net.params():
                diff = param.diff()
                diff *= learningrate
        return

    def iter_callback(self, loss):
        """Iteration callback."""
        if self.spec['asgd'] and self._iter_idx >= self.spec['asgd_skip']:
            # we need to maintain the asgd values.
            # pylint: disable=W0612
            for asgd, param in zip(self._asgd, self._decaf_net.params()):
                # we will simply do addition. Note that when you try to get
                # the final net, you need to divide the asgd_data by the
                # number of iterations minus asgd_skip. 
                asgd += param.data()
    
    def snapshot(self, is_final = False):
        """perform snapshot."""
        subfolder = StochasticSolver.snapshot(
            self, is_final=is_final)
        if self.spec['momentum'] > 0:
            # we need to store momentum as well
            with open(os.path.join(subfolder, 'momentum'), 'wb') as fid:
                pickle.dump(self._momentum, fid,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if self.spec['asgd']:
            # let's store the accumulated asgd values.
            with open(os.path.join(subfolder, 'asgd'), 'wb') as fid:
                pickle.dump(self._asgd, fid, protocol=pickle.HIGHEST_PROTOCOL)


class AdagradSolver(StochasticSolver):
    """The Adagrad Solver."""
    def __init__(self, **kwargs):
        """Initializes the SGD solver.

        kwargs:
            base_lr: the base learning rate.
            max_iter: the maximum number of iterations. Default 1000.
            base_accum: the base value to initialize the accumulated gradient
                diagonal. Default 1e-8.
        """
        StochasticSolver.__init__(self, **kwargs) 
        self.spec['base_accum'] = self.spec.get('base_accum', 1e-8)
        self._accum = None

    def initialize_status(self):
        """Initializes the status."""
        # we need to maintain the momentum history
        params = self._decaf_net.params()
        self._accum = [base.Blob(p.data().shape, p.data().dtype)
                       for p in params]
        for accum in self._accum:
            accum_data = accum.data()
            accum_data[:] = self.spec['base_accum']
            # we initialize the diff as a buffer when computing things later.
            accum.init_diff()
        
    def compute_update_value(self):
        """Computes the update value by multiplying the gradient with the
        learning rate.
        """
        for param, accum in zip(self._decaf_net.params(), self._accum):
            diff = param.diff()
            # add the current gradient to the accumulation
            accum_data = accum.data()
            accum_buffer = accum.diff()
            accum_buffer[:] = diff
            accum_buffer *= diff
            accum_data += accum_buffer
            # compute the sqrt, and update diff
            np.sqrt(accum_data, out=accum_buffer)
            diff /= accum_buffer
            diff *= self.spec['base_lr']
        return

    def snapshot(self, is_final = False):
        """perform snapshot."""
        subfolder = StochasticSolver.snapshot(self, is_final=is_final)
        with open(os.path.join(subfolder, 'adagrad_accum'), 'wb') as fid:
            pickle.dump(self._accum, fid, protocol=pickle.HIGHEST_PROTOCOL)
        return subfolder

