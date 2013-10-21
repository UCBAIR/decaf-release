"""Utility to perform gradient check using scipy's check_grad method. 
"""

import numpy as np
from scipy import optimize
import unittest


def blobs_to_vec(blobs):
    """Collect the network parameters into a long vector.

    This method is not memory efficient - do NOT use in codes that require
    speed and memory.
    """
    if len(blobs) == 0:
        return np.array(())
    return np.hstack([blob.data().flatten() for blob in blobs])

def blobs_diff_to_vec(blobs):
    """Similar to blobs_to_vec, but copying diff."""
    if len(blobs) == 0:
        return np.array(())
    return np.hstack([blob.diff().flatten() for blob in blobs])

def vec_to_blobs(vec, blobs):
    """Distribute the values in the vec to the blobs.
    """
    current = 0
    for blob in blobs:
        size = blob.data().size
        blob.data().flat = vec[current:current+size]
        current += size

def vec_to_blobs_diff(vec, blobs):
    """Distribute the values in the vec to the blobs' diff part.
    """
    current = 0
    for blob in blobs:
        size = blob.diff().size
        blob.diff().flat = vec[current:current+size]
        current += size


class GradChecker(unittest.TestCase):
    """A gradient checker that utilizes scipy.optimize.check_grad to perform
    the gradient check.

    The gradient checker checks the gradient with respect to both the params
    and the bottom blobs if they exist. It checks 2 types of object functions:
        (1) the squared sum of all the outputs.
        (2) each of the output value.
    The total number of functions to be tested is (num of outputs + 1).

    The check is carried out by the check() function, which checks all the
    cases above. If any error exceeds the threshold, the check function will
    return a tuple: (False, index, err), where index is the index where error
    exceeds threshold, and err is the error value. index=-1 means the squared
    sum case. If all errors are under threshold, the check function will
    return (True, max_err) where max_err is the maximum error encountered.
    """

    def __init__(self, threshold):
        """Initializes the checker.

        Input:
            threshold: the threshold to reject the gradient value.
        """
        self._threshold = threshold
    
    @staticmethod
    def _func_net(x_init, decaf_net):
        """function wrapper for a net."""
        vec_to_blobs(x_init, decaf_net.params())
        return decaf_net.forward_backward()

    @staticmethod
    def _grad_net(x_init, decaf_net):
        """gradient wrapper for a net."""
        vec_to_blobs(x_init, decaf_net.params())
        decaf_net.forward_backward()
        return blobs_diff_to_vec(decaf_net.params())

    # pylint: disable=R0913
    @staticmethod
    def _func(x_init, layer, input_blobs, output_blobs, check_data, idx,
             checked_blobs):
        """The function. It returns the output at index idx, or if idx is
        negative, computes an overall loss by taking the squared sum of all
        output values.
        
        Input:
            x_init: the feature values.
            layer: the layer to be checked.
            input_blobs: the input blobs.
            output_blobs: the output blobs.
            check_data: if True, check the gradient with respect to the input
                data.
            idx: how we compute the loss function. If negative, the loss is
                the squared sum of all output. Note that the regularization
                term of the layer is always added to the loss.
            checked_blobs: when check_data is True, checked_blobs is a sublist
                of the input blobs whose gradients we need to check. Any input
                blob not in the checked_blobs would not be checked.
        """
        if check_data:
            vec_to_blobs(x_init, checked_blobs)
        else:
            vec_to_blobs(x_init, layer.param())
        layer.forward(input_blobs, output_blobs)
        if len(output_blobs) > 0:
            output = blobs_to_vec(output_blobs)
        else:
            # a dummy output
            output = np.array([0])
        for blob in output_blobs:
            blob.init_diff()
        # The layer may have reg terms, so we run a dummy backward
        additional_loss = layer.backward(input_blobs, output_blobs, True)
        if idx < 0:
            return np.dot(output, output) + additional_loss
        else:
            return output[idx] + additional_loss

    #pylint: disable=R0913
    @staticmethod
    def _grad(x_init, layer, input_blobs, output_blobs, check_data, idx,
              checked_blobs):
        """The coarse gradient. See _func for documentation."""
        if check_data:
            vec_to_blobs(x_init, checked_blobs)
        else:
            vec_to_blobs(x_init, layer.param())
        layer.forward(input_blobs, output_blobs)
        # initialize the diff
        for blob in output_blobs:
            blob.init_diff()
        if len(output_blobs) > 0:
            output = blobs_to_vec(output_blobs)
            if idx < 0:
                output *= 2.
            else:
                output[:] = 0
                output[idx] = 1.
            vec_to_blobs_diff(output, output_blobs)
        # Now, get the diff
        if check_data:
            layer.backward(input_blobs, output_blobs, True)
            return blobs_diff_to_vec(checked_blobs)
        else:
            layer.backward(input_blobs, output_blobs, False)
            return blobs_diff_to_vec(layer.param())

    def check_network(self, decaf_net):
        """Checks a whole decaf network. Your network should not contain any
        stochastic components: multiple forward backward passes should produce
        the same value for the same parameters.
        """
        # Run a round to initialize the params.
        decaf_net.forward_backward()
        param_backup = blobs_to_vec(decaf_net.params())
        x_init = param_backup.copy()
        # pylint: disable=E1101
        err = optimize.check_grad(GradChecker._func_net, GradChecker._grad_net,
                                  x_init, decaf_net)
        self.assertLessEqual(err, self._threshold)
        if err > self._threshold:
            return (False, err)
        else:
            return (True, err)

    def check(self, layer, input_blobs, output_blobs, check_indices = None):
        """Checks a layer with given input blobs and output blobs.
        """
        # pre-run to get the input and output shapes.
        if check_indices is None:
            checked_blobs = input_blobs
        else:
            checked_blobs = [input_blobs[i] for i in check_indices]
        layer.forward(input_blobs, output_blobs)
        input_backup = blobs_to_vec(checked_blobs)
        param_backup = blobs_to_vec(layer.param())
        num_output = blobs_to_vec(output_blobs).size
        max_err = 0
        # first, check grad w.r.t. param
        x_init = blobs_to_vec(layer.param())
        if len(x_init) > 0:
            for i in range(-1, num_output):
                # pylint: disable=E1101
                err = optimize.check_grad(
                    GradChecker._func, GradChecker._grad, x_init,
                    layer, input_blobs, output_blobs, False, i, checked_blobs)
                max_err = max(err, max_err)
                self.assertLessEqual(err, self._threshold)
                if err > self._threshold:
                    return (False, i, err, 'param')
            # restore param
            vec_to_blobs(param_backup, layer.param())
        # second, check grad w.r.t. input
        x_init = blobs_to_vec(checked_blobs)
        if len(x_init) > 0:
            for i in range(-1, num_output):
                # pylint: disable=E1101
                err = optimize.check_grad(
                    GradChecker._func, GradChecker._grad, x_init,
                    layer, input_blobs, output_blobs, True, i, checked_blobs)
                max_err = max(err, max_err)
                self.assertLessEqual(err, self._threshold)
                if err > self._threshold:
                    return (False, i, err, 'input')
            # restore input
            vec_to_blobs(input_backup, checked_blobs)
        return (True, max_err)
