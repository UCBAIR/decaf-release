"""transform implements a few common functions that are often used in multiple
applications.
"""

import numpy as np
from skimage import transform

def scale_and_extract(image, height, width=None):
    """This function scales the image and then extracts the center part of the
    image with the given height and width.

    Input:
        image: an ndarray or a skimage Image.
        height: the target height of the image.
        width: the target width of the image. If not provided, we will use
            width = height.
    output:
        image_out: an ndarray of (height * width), and of dtype float64.
    """
    if not width:
        width = height
    ratio = max(height / float(image.shape[0]), width / float(image.shape[1]))
    # we add 0.5 to the converted result to avoid numerical problems.
    image_reshape = transform.resize(
        image, (int(image.shape[0] * ratio + 0.5), int(image.shape[1] * ratio + 0.5)))
    h_offset = (image_reshape.shape[0] - height) / 2
    w_offset = (image_reshape.shape[1] - width) / 2
    return image_reshape[h_offset:h_offset+height, w_offset:w_offset+width]


def as_rgb(image):
    """Converts an image that could possibly be a grayscale or RGBA image to
    RGB.
    """
    if image.ndim == 2:
        return np.tile(image[:, :, np.newaxis], (1, 1, 3))
    elif image.shape[2] == 4:
        # An RGBA image. We will only use the first 3 channels.
        return image[:, :, :3]
    else:
        return image
