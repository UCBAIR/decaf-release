"""This mpi4py dummy code here is to trigger an exception
so we can check the code under the no mpi case
"""
raise ImportError, "Entering dummy mpi"

if __name__ == "__main__":
    pass
