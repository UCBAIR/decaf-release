"""This module implements a translator model that is able to convert a network
trained using Alex Krizhevsky's cuda-convnet code to a decaf net. This allows
one to use the GPU for efficient training of certain network components, but
to later embed them into a more flexible network or to deploy them on CPU-only
platforms.

Some of the network components are written by Jeff Donahue at our lab at UC
Berkeley, so it may not work directly with Alex's released code.
"""

# first of all, import the registerer
# pylint: disable=W0401
from registerer import *
from conversions import *

# In the lines below, we will import all the translators we implemented.
import translator_cmrnorm
import translator_conv
import translator_fc
import translator_neuron
import translator_pool
import translator_softmax
