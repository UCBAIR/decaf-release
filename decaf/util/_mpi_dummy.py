# pylint: disable=C0103, C0111, W0613
"""This implements some dummy functions that mimics the MPI behavior when the
size is 1. It is here solely to provide (probably limited) ability to run single
noded tasks when one cannot install mpi or mpi4py.
"""

import copy

class COMM(object):
    """The dummy mpi common world.
    """
    def __init__(self):
        raise RuntimeError, "COMM should not be instantiated."
    
    @staticmethod
    def Get_rank():
        return 0
    
    @staticmethod
    def Get_size():
        return 1
    
    @staticmethod
    def allgather(sendobj):
        return [copy.copy(sendobj)]
    
    @staticmethod
    def Allreduce(sendbuf, recvbuf, op = None):
        recvbuf[:] = sendbuf[:]
    
    @staticmethod
    def allreduce(sendobj, op = None):
        return copy.copy(sendobj)
    
    @staticmethod
    def bcast(sendobj, root = 0):
        return copy.copy(sendobj)
    
    @staticmethod
    def Bcast(buf, root = 0):
        pass
    
    @staticmethod
    def gather(sendobj, root = 0):
        return [copy.copy(sendobj)]
    
    @staticmethod
    def Reduce(sendbuf, recvbuf, op = None, root = 0):
        recvbuf[:] = sendbuf[:]
