"""Implements the minibatch sampling methods for data stored in puff format.
"""

from decaf import base
from decaf.util import mpi
import numpy as np


class PuffSamplerLayer(base.DataLayer):
    """A layer that loads data from a set of puff files."""
    def __init__(self, **kwargs):
        """Initializes the Puff sampling layer.

        kwargs:
            minibatch: the minibatch size.
            puff: a list of puff files to be read. These files should have the
                same number of data points, and the sampler will return data
                points from different points with the same index.
            use_mpi: if set True, when the code is run with mpirun, each mpi
                node will only deal with the part of file that has index range
                [N * RANK / SIZE, N * (RANK+1) / SIZE). Note that in this
                case, one need to make sure that the minibatch size is smaller
                than the number of data points in the local range on every mpi
                node. Default True.
        """
        base.DataLayer.__init__(self, **kwargs)
        self._filenames = self.spec['puff']
        self._use_mpi = self.spec.get('use_mpi', True)
        self._minibatch = self.spec['minibatch']
        self._puffs = [base.Puff(filename) for filename in self._filenames]
        num_data = [puff.num_data() for puff in self._puffs]
        if len(set(num_data)) == 1:
            raise ValueError('The puff files have different number of data.')
        if self._use_mpi:
            local_start = int(num_data[0] * mpi.RANK / mpi.SIZE)
            local_end = int(num_data[0] * (mpi.RANK + 1) / mpi.SIZE)
            if mpi.mpi_any(local_end - local_start < self._minibatch):
                raise ValueError('Local range smaller than minibatch.')
            for puff in self._puffs:
                puff.set_range(local_start, local_end)
        return

    def forward(self, bottom, top):
        """The forward pass."""
        for puff, top_blob in zip(self._puffs, top):
            top_blob.mirror(puff.read(self._minibatch))
        return

