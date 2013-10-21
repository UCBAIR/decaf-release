"""A code to perform logistic regression."""
import cPickle as pickle
from decaf import base
from decaf.layers import core_layers
from decaf.layers import regularization
from decaf.layers import fillers
from decaf.layers.data import mnist
from decaf.opt import core_solvers
from decaf.util import visualize
import logging
import numpy as np
import sys

# You may want to change these parameters when running the code.
ROOT_FOLDER='/u/vis/x1/common/mnist'
MINIBATCH=4096
NUM_NEURONS = 100
NUM_CLASS = 10
METHOD = 'sgd'

def main():
    logging.getLogger().setLevel(logging.INFO)
    ######################################
    # First, let's create the decaf layer.
    ######################################
    logging.info('Loading data and creating the network...')
    decaf_net = base.Net()
    # add data layer
    dataset = mnist.MNISTDataLayer(
        name='mnist', rootfolder=ROOT_FOLDER, is_training=True)
    decaf_net.add_layer(dataset,
                        provides=['image-all', 'label-all'])
    # add minibatch layer for stochastic optimization
    minibatch_layer = core_layers.BasicMinibatchLayer(
        name='batch', minibatch=MINIBATCH)
    decaf_net.add_layer(minibatch_layer,
                        needs=['image-all', 'label-all'],
                        provides=['image', 'label'])
    # add the two_layer network
    decaf_net.add_layers([
        core_layers.FlattenLayer(name='flatten'),
        core_layers.InnerProductLayer(
            name='ip1', num_output=NUM_NEURONS,
            filler=fillers.GaussianRandFiller(std=0.1),
            bias_filler=fillers.ConstantFiller(value=0.1)),
        core_layers.ReLULayer(name='relu1'),
        core_layers.InnerProductLayer(
            name='ip2', num_output=NUM_CLASS,
            filler=fillers.GaussianRandFiller(std=0.3))
        ], needs='image', provides='prediction')
    # add loss layer
    loss_layer = core_layers.MultinomialLogisticLossLayer(
        name='loss')
    decaf_net.add_layer(loss_layer,
                        needs=['prediction', 'label'])
    # finish.
    decaf_net.finish()
    ####################################
    # Decaf layer finished construction!
    ####################################
    
    # now, try to solve it
    if METHOD == 'adagrad':
        # The Adagrad Solver
        solver = core_solvers.AdagradSolver(base_lr=0.02, base_accum=1.e-6,
                                            max_iter=1000)
    elif METHOD == 'sgd':
        solver = core_solvers.SGDSolver(base_lr=0.1, lr_policy='inv',
                                        gamma=0.001, power=0.75, momentum=0.9,
                                        max_iter=1000)
    solver.solve(decaf_net)
    visualize.draw_net_to_file(decaf_net, 'mnist.png')
    decaf_net.save('mnist_2layers.decafnet')

    ##############################################
    # Now, let's load the net and run predictions 
    ##############################################
    prediction_net = base.Net.load('mnist_2layers.decafnet')
    visualize.draw_net_to_file(prediction_net, 'mnist_test.png')
    # obtain the test data.
    dataset_test = mnist.MNISTDataLayer(
        name='mnist', rootfolder=ROOT_FOLDER, is_training=False)
    test_image = base.Blob()
    test_label = base.Blob()
    dataset_test.forward([], [test_image, test_label])
    # Run the net.
    pred = prediction_net.predict(image=test_image)['prediction']
    accuracy = (pred.argmax(1) == test_label.data()).sum() / float(test_label.data().size)
    print 'Testing accuracy:', accuracy
    print 'Done.'

if __name__ == '__main__':
    main()
