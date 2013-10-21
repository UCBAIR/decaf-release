"""imagenet implements a wrapper over the imagenet classifier trained by Jeff
Donahue using the cuda convnet code.
"""
import cPickle as pickle
from decaf.util import translator, transform
import logging
import numpy as np
import os

_JEFFNET_FILE = os.path.join(os.path.dirname(__file__),
                             'imagenet.decafnet.epoch90')
_META_FILE = os.path.join(os.path.dirname(__file__), 'imagenet.decafnet.meta')

# This is a legacy flag specifying if the network is trained with vertically
# flipped images, which does not hurt performance but requires us to flip
# the input image first.
_JEFFNET_FLIP = True

# Due to implementational differences between the CPU and GPU codes, our net
# takes in 227x227 images - which supports convolution with 11x11 patches and
# stride 4 to a 55x55 output without any missing pixels. As a note, the GPU
# code takes 224 * 224 images, and does convolution with the same setting and
# no padding. As a result, the last image location is only convolved with 8x8
# image regions.
INPUT_DIM = 227

class DecafNet(object):
    """A wrapper that returns the decafnet interface to classify images."""
    def __init__(self, net_file=None, meta_file=None):
        """Initializes DecafNet.

        Input:
            net_file: the trained network file.
            meta_file: the meta information for images.
        """
        logging.info('Initializing decafnet...')
        try:
            if not net_file:
                # use the internal decafnet file.
                net_file = _JEFFNET_FILE
            if not meta_file:
                # use the internal meta file.
                meta_file = _META_FILE
            cuda_decafnet = pickle.load(open(net_file))
            meta = pickle.load(open(meta_file))
        except IOError:
            raise RuntimeError('Cannot find DecafNet files.')
        # First, translate the network
        self._net = translator.translate_cuda_network(
            cuda_decafnet, {'data': (INPUT_DIM, INPUT_DIM, 3)})
        # Then, get the labels and image means.
        self.label_names = meta['label_names']
        self._data_mean = translator.img_cudaconv_to_decaf(
            meta['data_mean'], 256, 3)
        logging.info('Jeffnet initialized.')
        return

    def classify_direct(self, images):
        """Performs the classification directly, assuming that the input
        images are already of the right form.

        Input:
            images: a numpy array of size (num x 227 x 227 x 3), dtype
                float32, c_contiguous, and has the mean subtracted and the
                image flipped if necessary.
        Output:
            scores: a numpy array of size (num x 1000) containing the
                predicted scores for the 1000 classes.
        """
        return self._net.predict(data=images)['probs_cudanet_out']

    @staticmethod
    def oversample(image, center_only=False):
        """Oversamples an image. Currently the indices are hard coded to the
        4 corners and the center of the image, as well as their flipped ones,
        a total of 10 images.

        Input:
            image: an image of size (256 x 256 x 3) and has data type uint8.
            center_only: if True, only return the center image.
        Output:
            images: the output of size (10 x 227 x 227 x 3)
        """
        indices = [0, 256 - INPUT_DIM]
        center = int(indices[1] / 2)
        if center_only:
            return np.ascontiguousarray(
                image[np.newaxis, center:center + INPUT_DIM,
                      center:center + INPUT_DIM], dtype=np.float32)
        else:
            images = np.empty((10, INPUT_DIM, INPUT_DIM, 3),
                              dtype=np.float32)
            curr = 0
            for i in indices:
                for j in indices:
                    images[curr] = image[i:i + INPUT_DIM,
                                         j:j + INPUT_DIM]
                    curr += 1
            images[4] = image[center:center + INPUT_DIM,
                              center:center + INPUT_DIM]
            # flipped version
            images[5:] = images[:5, ::-1]
            return images
    
    def classify(self, image, center_only=False):
        """Classifies an input image.
        
        Input:
            image: an image of 3 channels and has data type uint8. Only the
                center region will be used for classification.
        Output:
            scores: a numpy vector of size 1000 containing the
                predicted scores for the 1000 classes.
        """
        # first, extract the 256x256 center.
        image = transform.scale_and_extract(transform.as_rgb(image), 256)
        # convert to [0,255] float32
        image = image.astype(np.float32) * 255.
        if _JEFFNET_FLIP:
            # Flip the image if necessary, maintaining the c_contiguous order
            image = image[::-1, :].copy()
        # subtract the mean
        image -= self._data_mean
        # oversample the images
        images = DecafNet.oversample(image, center_only)
        predictions = self.classify_direct(images)
        return predictions.mean(0)

    def top_k_prediction(self, scores, k):
        """Returns the top k predictions as well as their names as strings.
        
        Input:
            scores: a numpy vector of size 1000 containing the
                predicted scores for the 1000 classes.
        Output:
            indices: the top k prediction indices.
            names: the top k prediction names.
        """
        indices = scores.argsort()
        return (indices[:-(k+1):-1],
                [self.label_names[i] for i in indices[:-(k+1):-1]])

    def feature(self, blob_name):
        """Returns the feature of a specific blob.
        Input:
            blob_name: the name of the blob requested.
        Output:
            array: the numpy array storing the feature.
        """
        # We will copy the feature matrix in case further calls overwrite
        # it.
        return self._net.feature(blob_name).copy()


def main():
    """A simple demo showing how to run decafnet."""
    from decaf.util import smalldata, visualize
    logging.getLogger().setLevel(logging.INFO)
    net = DecafNet()
    lena = smalldata.lena()
    scores = net.classify(lena)
    print 'prediction:', net.top_k_prediction(scores, 5)
    visualize.draw_net_to_file(net._net, 'decafnet.png')
    print 'Network structure written to decafnet.png'


if __name__ == '__main__':
    main()
