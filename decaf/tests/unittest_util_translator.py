import cPickle as pickle
from decaf import base
from decaf.util import translator
from decaf.util import visualize
from matplotlib import pyplot
import numpy as np
import os
import unittest

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'convnet_data')
_HAS_TEST_DATA = os.path.exists(os.path.join(_TEST_DATA_DIR, 'layers.pickle'))
_BATCH_SIZE = 32


@unittest.skipIf(not _HAS_TEST_DATA, 
                 'No cuda convnet test data found. Run'
                 ' convnet_data/get_decaf_testdata.sh to get the test data.')
class TestCudaConv(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        self._layers = pickle.load(open(os.path.join(_TEST_DATA_DIR,
                                                     'layers.pickle')))
        self._data = pickle.load(open(os.path.join(_TEST_DATA_DIR,
                                                   'data',
                                                   'data_batch_5')))
        self._decaf_data = translator.imgs_cudaconv_to_decaf(
            self._data['data'][:_BATCH_SIZE], 32, 3)
        #self._decaf_labels = self._data['labels'].flatten()[:_BATCH_SIZE]
        #self._decaf_labels = self._decaf_labels.astype(np.int)
        self._output_shapes = {'data': (32, 32, 3), 'labels': -1}
        self._net = translator.translate_cuda_network(
            self._layers, self._output_shapes)
        self._net.predict(data=self._decaf_data)
        #visualize.draw_net_to_file(self._net, 'test.png')

    def _testSingleLayer(self, decaf_name, cuda_name, reshape_size=0,
                         reshape_channels=0, decimal=6):
        output = self._net.feature(self._net.provides[decaf_name][0])
        self.assertEqual(output.shape[1:], self._output_shapes[decaf_name])
        ref_data = pickle.load(open(
            os.path.join(_TEST_DATA_DIR, cuda_name, 'data_batch_5')))
        ref_data = ref_data['data'][:_BATCH_SIZE]
        if reshape_size:
            ref_data = translator.imgs_cudaconv_to_decaf(
                ref_data, reshape_size, reshape_channels)
        # We rescale the data so that the decimal specified would also count
        # the original scale of the data.
        maxval = ref_data.max()
        ref_data /= maxval
        output /= maxval
        #print 'data range: [%f, %f], max diff: %f' % (
        #    ref_data.min(), ref_data.max(), np.abs(ref_data - output).max())
        np.testing.assert_array_almost_equal(ref_data, output, decimal)

    def testConv1(self):
        self._testSingleLayer('conv1', 'conv1', 32, 32)

    def testPool1(self):
        self._testSingleLayer('pool1', 'pool1', 16, 32)

    def testPool1Neuron(self):
        self._testSingleLayer('pool1_neuron', 'pool1_neuron', 16, 32)

    def testRnorm1(self):
        self._testSingleLayer('rnorm1', 'rnorm1', 16, 32)

    def testConv2(self):
        self._testSingleLayer('conv2_neuron', 'conv2_neuron', 16, 64)

    def testPool2(self):
        self._testSingleLayer('pool2', 'pool2', 8, 64)

    def testConv3(self):
        self._testSingleLayer('conv3_neuron', 'conv3_neuron', 8, 64, decimal=5)

    def testPool3(self):
        self._testSingleLayer('pool3', 'pool3', 4, 64)

    def testFc64(self):
        self._testSingleLayer('fc64_neuron', 'fc64_neuron')

    def testFc10(self):
        self._testSingleLayer('fc10', 'fc10')

    def testProbs(self):
        self._testSingleLayer('probs', 'probs')

if __name__ == '__main__':
    unittest.main()

