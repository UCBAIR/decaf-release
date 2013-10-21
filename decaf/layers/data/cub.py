"""The Caltech-UCSD bird dataset
"""

from decaf.layers.data import ndarraydata
import numpy as np
import os
from scipy import misc
from skimage import io


class CUBDataLayer(ndarraydata.NdarrayDataLayer):
    """ The Caltech-UCSD bird dataset
    """
    def __init__(self, **kwargs):
        """Load the dataset.
        kwargs:
            root: the root folder of the CUB_200_2011 dataset.
            is_training: if true, load the training data. Otherwise, load the
                testing data.
            crop: if None, does not crop the bounding box. If a real value,
                crop is the ratio of the bounding box that gets cropped.
                e.g., if crop = 1.5, the resulting image will be 1.5 * the
                bounding box area.
            target_size: all images are resized to the size specified. Should
                be a tuple of two integers, like [256, 256].
            version: either '2011' or '2010'.
        Note that we will use the python indexing (labels start from 0).
        """
        root = kwargs['root']
        is_training = kwargs.get('is_training', True)
        crop = kwargs.get('crop', None)
        target_size = kwargs['target_size']
        version = kwargs.get('version', '2011')
        if version == '2011':
            images = [line.split()[1] for line in
                        open(os.path.join(root, 'images.txt'), 'r')]
            boxes = [line.split()[1:] for line in
                        open(os.path.join(root, 'bounding_boxes.txt'),'r')]
            labels = [int(line.split()[1]) - 1 for line in 
                        open(os.path.join(root, 'image_class_labels.txt'), 'r')]
            birdnames = [line.split()[1] for line in
                          open(os.path.join(root, 'classes.txt'), 'r')]
            name_to_id = dict(zip(birdnames, range(len(birdnames))))
            split = [int(line.split()[1]) for line in
                        open(os.path.join(root, 'train_test_split.txt'),'r')]
        elif version == '2010':
            # we are using version 2010. We load the data to mimic the 2011
            # version data format
            images = [line.strip() for line in
                        open(os.path.join(root, 'lists', 'files.txt'), 'r')]
            boxes = []
            # unfortunately, we need to load the boxes from matlab annotations
            for filename in images:
                matfile = io.loadmat(os.path.join(root, 'annotations-mat',
                                                  filename[:-3]+'mat'))
                left, top, right, bottom = \
                        [matfile['bbox'][0][0][i][0][0] for i in range(4)]
                boxes.append([left, top, right-left, bottom-top])
            # get the training and testing split.
            train_images = [line.strip() for line in
                        open(os.path.join(root, 'lists', 'train.txt'), 'r')]
            labels = [int(line[:line.find('.')]) - 1 for line in images]
            birdnames = [line.strip() for line in
                        open(os.path.join(root, 'lists', 'classes.txt'),'r')]
            name_to_id = dict(zip(birdnames, range(len(birdnames))))
            split = [int(line in train_images) for line in images]
        else:
            raise ValueError, "Unrecognized version: %s" % version
        # now, do training testing split
        if is_training:
            target = 1
        else:
            target = 0
        images = [image for image, val in zip(images, split) if val == target]
        boxes = [box for box, val in zip(boxes, split) if val == target]
        label = [label for label, val in zip(labels, split) if val == target]
        # for the boxes, we store them as a numpy array
        boxes = np.array(boxes, dtype=np.float32)
        boxes -= 1
        # load the data
        self._data = None
        self._load_data(root, images, boxes, crop, target_size)
        self._label = np.asarray(label, dtype=np.int)
        ndarraydata.NdarrayDataLayer.__init__(
            self, sources=[self._data, self._label], **kwargs)

    def _load_data(self, root, images, boxes, crop, target_size):
        num_imgs = len(images)
        self._data = np.empty((num_imgs, target_size[0], target_size[1], 3),
                              dtype=np.uint8)
        for i in range(num_imgs):
            image = io.imread(os.path.join(root, 'images', images[i]))
            if image.ndim == 2:
                image = np.tile(image[:,:,np.newaxis], (1, 1, 3))
            if image.shape[2] == 4:
                image = image[:, :, :3]
            if crop:
                image = self._crop_image(image, crop, boxes[i])
            self._data[i] = misc.imresize(image, target_size)
        return

    def _crop_image(self, image, crop, box):
        imheight, imwidth = image.shape[:2]
        x, y, width, height = box
        centerx = x + width / 2.
        centery = y + height / 2.
        xoffset = width * crop / 2.
        yoffset = height * crop / 2.
        xmin = max(int(centerx - xoffset + 0.5), 0)
        ymin = max(int(centery - yoffset + 0.5), 0)
        xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
        ymax = min(int(centery + yoffset + 0.5), imheight - 1)
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            raise ValueError("The cropped bounding box has size 0.")
        return image[ymin:ymax, xmin:xmax]
