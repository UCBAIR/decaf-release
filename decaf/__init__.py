"""
Decaf: a deep convolutional neural networks framework
=====

Decaf is a framework that implements convolutional neural networks, with the
goal of being efficient and flexible. It allows one to easily construct a
network in the form of an arbitrary Directed Acyclic Graph (DAG), and to
perform end-to-end training in a distributed fashion.
"""

__author__ = 'Yangqing Jia'
__email__  = 'jiayq84@gmail.com'

import base
import layers
import opt