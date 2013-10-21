'''The Cifar dataset 
'''
import cPickle as pickle
from decaf.layers.data import ndarraydata
import numpy as np
import os


class CIFARDataLayer(ndarraydata.NdarrayDataLayer):
    """The CIFAR dataset
    """
    # some cifar constants
    __num_train = 50000
    __num_batches = 5 # for cifar 10
    __batchsize = 10000 # for cifar 10
    __num_test = 10000
    __image_dim = (32, 32, 3)
    __num_channels = 3
    __image_size = 1024
    __flat_dim = 3072
    
    def __init__(self, **kwargs):
        """Initializes the cifar layer.

        kwargs:
            is_training: whether to load the training data. Default True.
            is_gray: whether to load gray image. Default False.
            rootfolder: the folder that stores the mnist data.
            dtype: the data type. Default numpy.float64.
        """
        # get keywords
        is_training = kwargs.get('is_training', True)
        is_gray = kwargs.get('is_gray', False)
        rootfolder = kwargs['rootfolder']
        dtype = kwargs.get('dtype', np.float64)
        self._data = None
        self._label = None
        self._coarselabel = None
        # we will automatically determine if the data is cifar-10 or cifar-100
        if os.path.exists(os.path.join(rootfolder, 'batches.meta')):
            self.load_cifar10(rootfolder, is_training, dtype)
        elif os.path.exists(os.path.join(rootfolder, 'meta')):
            self.load_cifar100(rootfolder, is_training, dtype)
        else:
            raise IOError, 'Cannot understand the dataset format.'
        if is_gray:
            self._data = self._data.mean(axis=-1)
        # Normalize data to [0, 1)
        self._data /= 255.
        # Initialize as an NdarrayDataLayer
        ndarraydata.NdarrayDataLayer.__init__(
            self, sources=[self._data, self._label], **kwargs)
        
    @staticmethod
    def _get_images_from_matrix(mat, dtype):
        """Converts the order of the loaded matrix so each pixel is stored
        contiguously
        """
        mat = mat.reshape((mat.shape[0],
                           CIFARDataLayer.__num_channels,
                           CIFARDataLayer.__image_size))
        images = mat.swapaxes(1, 2).reshape(
            (mat.shape[0],) + CIFARDataLayer.__image_dim)
        return np.ascontiguousarray(images.astype(dtype))
    
    def load_cifar100(self, rootfolder, is_training, dtype):
        """loads the cifar-100 dataset
        """
        if is_training:
            filename = 'train'
        else:
            filename = 'test'
        with open(rootfolder + os.sep + filename) as fid:
            batch = pickle.load(fid)
        self._data = CIFARDataLayer._get_images_from_matrix(
            batch['data'], dtype)
        self._coarselabel = np.array(batch['coarse_labels']).astype(np.int)
        self._label = np.array(batch['fine_labels']).astype(np.int)
    
    def load_cifar10(self, rootfolder, is_training, dtype):
        """loads the cifar-10 dataset
        """
        if is_training:
            self._data = np.empty(
                (CIFARDataLayer.__num_train,) + CIFARDataLayer.__image_dim,
                dtype=dtype)
            self._label = np.empty(CIFARDataLayer.__num_train, dtype=np.int)
            # training batches
            for i in range(CIFARDataLayer.__num_batches):
                with open(os.path.join(rootfolder,
                        'data_batch_{0}'.format(i+1)),'r') as fid:
                    batch = pickle.load(fid)
                start_idx = CIFARDataLayer.__batchsize * i
                end_idx = CIFARDataLayer.__batchsize * (i+1)
                self._data[start_idx:end_idx] = \
                    CIFARDataLayer._get_images_from_matrix(batch['data'], dtype)
                self._label[start_idx:end_idx] = np.array(batch['labels'])
        else:
            with open(os.path.join(rootfolder, 'test_batch'), 'r') as fid:
                batch = pickle.load(fid)
            self._data = CIFARDataLayer._get_images_from_matrix(
                batch['data'], dtype)
            self._label = np.array(batch['labels']).astype(np.int)

