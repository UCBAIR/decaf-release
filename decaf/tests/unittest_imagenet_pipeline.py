"""A code to perform logistic regression."""
import cPickle as pickle
from decaf import base
from decaf.layers import core_layers
from decaf.layers import regularization
from decaf.layers import fillers
import logging
import numpy as np
import os
import sys
import unittest

def imagenet_layers():
    return [
        core_layers.ConvolutionLayer(
            name='conv-220-3-to-55-96', num_kernels=96, ksize=11,
            stride=4, mode='same', filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-55-96'),
        core_layers.LocalResponseNormalizeLayer(
            name='lrn-55-96', k=2., alpha=0.0001, beta=0.75, size=5),
        core_layers.PoolingLayer(
            name='pool-55-to-27', psize=3, stride=2, mode='max'),
        core_layers.GroupConvolutionLayer(
            name='conv-27-256', num_kernels=128, group=2, ksize=5,
            stride=1, mode='same', filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-27-256'),
        core_layers.LocalResponseNormalizeLayer(
            name='lrn-27-256', k=2., alpha=0.0001, beta=0.75, size=5),
        core_layers.PoolingLayer(
            name='pool-27-to-13', psize=3, stride=2, mode='max'),
        core_layers.ConvolutionLayer(
            name='conv-13-384', num_kernels=384, ksize=3,
            stride=1, mode='same', filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-13-384'),
        core_layers.GroupConvolutionLayer(
            name='conv-13-384-second', num_kernels=192, group=2, ksize=3,
            stride=1, mode='same', filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-13-384-second'),
        core_layers.GroupConvolutionLayer(
            name='conv-13-256', num_kernels=128, group=2, ksize=3,
            stride=1, mode='same', filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-13-256'),
        core_layers.PoolingLayer(
            name='pool-13-to-6', psize=3, stride=2, mode='max'),
        core_layers.FlattenLayer(name='flatten'),
        core_layers.InnerProductLayer(
            name='fully-1', num_output=4096,
            filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-full1'),
        core_layers.InnerProductLayer(
            name='fully-2', num_output=4096,
            filler=fillers.XavierFiller()),
        core_layers.ReLULayer(name='relu-full2'),
        core_layers.InnerProductLayer(
            name='predict', num_output=1000,
            filler=fillers.XavierFiller()),
    ]

def imagenet_data():
    """We will create a dummy imagenet data of one single image."""
    data = np.random.rand(1, 220, 220, 3).astype(np.float32)
    label = np.random.randint(1000, size=1)
    dataset = core_layers.NdarrayDataLayer(name='data', sources=[data, label])
    return dataset

class TestImagenet(unittest.TestCase):
    def setUp(self):
        np.random.seed(1701)
    
    def testPredict(self):
        # testPredict tests performing prediction without loss layer.
        decaf_net = base.Net()
        # add data layer
        decaf_net.add_layers(imagenet_data(),
                             provides=['image', 'label'])
        decaf_net.add_layers(imagenet_layers(),
                             needs='image',
                             provides='prediction')
        decaf_net.finish()
        result = decaf_net.predict()
        self.assertTrue('label' in result)
        self.assertTrue('prediction' in result)
        self.assertEqual(result['prediction'].shape[-1], 1000)


    def testForwardBackward(self):
        # testForwardBackward tests the full f-b path.
        decaf_net = base.Net()
        # add data layer
        decaf_net.add_layers(imagenet_data(),
                             provides=['image', 'label'])
        decaf_net.add_layers(imagenet_layers(),
                             needs='image',
                             provides='prediction')
        loss_layer = core_layers.MultinomialLogisticLossLayer(
            name='loss')
        decaf_net.add_layer(loss_layer,
                            needs=['prediction', 'label'])
        decaf_net.finish()
        loss = decaf_net.forward_backward()
        self.assertGreater(loss, 0.)
        self.assertLess(loss, 10.)
    
    def testForwardBackwardWithPrevLayer(self):
        decaf_net = base.Net()
        prev_net = base.Net()
        # add data layer
        prev_net.add_layers(imagenet_data(),
                             provides=['image', 'label'])
        decaf_net.add_layers(imagenet_layers(),
                             needs='image',
                             provides='prediction')
        loss_layer = core_layers.MultinomialLogisticLossLayer(
            name='loss')
        decaf_net.add_layer(loss_layer,
                            needs=['prediction', 'label'])
        prev_net.finish()
        decaf_net.finish()
        loss = decaf_net.forward_backward(previous_net=prev_net)
        self.assertGreater(loss, 0.)
        self.assertLess(loss, 10.)

if __name__ == '__main__':
    unittest.main()
