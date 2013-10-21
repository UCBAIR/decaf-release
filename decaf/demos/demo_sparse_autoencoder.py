"""This demo will show how we train a sparse autoencoder as described in more
detail at:
    http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder

To run this demo, simply do python demo_sparse_autoencoders.py. 

You will need to have X connection if you are ssh into your server. The
program will output the network structure to sparse-autoencoder-structure.png,
and display the learned filters. The trained network (less the input and loss
layers) is saved at sparse-autoencoder.decaf_net.
"""

from decaf import base
from decaf.util import smalldata, visualize
from decaf.layers import core_layers, fillers, regularization
from decaf.opt import core_solvers
import logging
from matplotlib import pyplot
import numpy as np

################################################
# Setting up the parameters for the autoencoder.
################################################

NUM_PATCHES = 10000
PSIZE = 8
NUM_HIDDEN = 25
INIT_SCALE = np.sqrt(6. / (NUM_HIDDEN + PSIZE * PSIZE + 1))
MAXFUN = 500
np.random.seed(1701)

################################################
# Generating training data.
################################################

logging.getLogger().setLevel(logging.INFO)
logging.info('*** Get patches ***')
images = smalldata.whitened_images()
patch_extractor = core_layers.RandomPatchLayer(
    name='extractor', psize=PSIZE, factor=NUM_PATCHES / 10)
patches = base.Blob()
patch_extractor.forward([images], [patches])
logging.info('*** Patch stats: %s', str(patches.data().shape))
logging.info('*** Normalize patches ***')
patches_data = patches.data()
# subtract mean
patches_data -= patches_data.mean(axis=0)
std = patches_data.std()
np.clip(patches_data, - std * 3, std * 3, out=patches_data)
# We shrink the patch range a little, to [0.1, 0.9]
patches_data *= 0.4 / std / 3.
patches_data += 0.5
logging.info('*** Finished Patch Preparation ***')

#############################################
# Creating the decaf net for the autoencoder.
#############################################

logging.info('*** Constructing the network ***')
decaf_net = base.Net()
# The data layer
decaf_net.add_layer(
    core_layers.NdarrayDataLayer(name='data-layer', sources=[patches]),
    provides='patches-origin')
# We will flatten the patches to a flat vector
decaf_net.add_layer(
    core_layers.FlattenLayer(name='flatten'),
    needs='patches-origin',
    provides='patches-flatten')
# The first inner product layer
decaf_net.add_layer(
    core_layers.InnerProductLayer(
            name='ip',
            num_output=NUM_HIDDEN,
            filler=fillers.RandFiller(min=-INIT_SCALE, max=INIT_SCALE),
            reg=regularization.L2Regularizer(weight=0.00005)),
    needs='patches-flatten',
    provides='ip-out')
# The first sigmoid layer
decaf_net.add_layer(
    core_layers.SigmoidLayer(name='sigmoid'),
    needs='ip-out',
    provides='sigmoid-out')
# The sparsity term imposed on the sigmoid output
decaf_net.add_layer(
    core_layers.AutoencoderLossLayer(
            name='sigmoid-reg',
            weight=3.,
            ratio=0.01),
    needs='sigmoid-out')
# The second inner product layer
decaf_net.add_layer(
    core_layers.InnerProductLayer(
            name='ip2',
            num_output=PSIZE * PSIZE,
            filler=fillers.RandFiller(min=-INIT_SCALE, max=INIT_SCALE),
            reg=regularization.L2Regularizer(weight=0.00005)),
    needs='sigmoid-out',
    provides='ip2-out')
# The second sigmoid layer
decaf_net.add_layer(
    core_layers.SigmoidLayer(name='sigmoid2'),
    needs='ip2-out',
    provides='sigmoid2-out')
# The reconstruction loss function
decaf_net.add_layer(
    core_layers.SquaredLossLayer(name='loss'),
    needs=['sigmoid2-out', 'patches-flatten'])
# Finished running decaf_net.
decaf_net.finish()

# let's do a proof-of-concept run, and draw the graph network to file.
decaf_net.forward_backward()
visualize.draw_net_to_file(decaf_net, 'sparse-autoencoder-structure.png')

#############################################
# The optimization.
#############################################
logging.info('*** Calling LBFGS solver ***')
solver = core_solvers.LBFGSSolver(
    lbfgs_args={'maxfun': MAXFUN, 'disp': 1})
solver.solve(decaf_net)
# let's get the weight matrix and show it.
weight = decaf_net.layers['ip'].param()[0].data()
#visualize.show_multiple(weight.T)
decaf_net.save('sparse-autoencoder.decafnet')
#pyplot.show()

