"""Functions that could be used to visualize patches."""

from decaf import base
from matplotlib import cm, pyplot
import numpy as np
import os
import pydot

LAYER_STYLE = {'shape': 'record', 'fillcolor': '#DFECF3',
               'style': 'filled,bold'}
BLOB_STYLE = {'shape': 'record', 'fillcolor': '#FEF9E3',
              'style': 'rounded,filled'}

def draw_net(decaf_net, ext='png'):
    """Draws a decaf net and returns the image string encoded using the given
    extension.
    
    Input:
        decaf_net: a decaf net.
        ext: the image extension. Default 'png'.
    """
    pydot_graph = pydot.Dot(graph_type='digraph')
    pydot_nodes = {}
    for name, layer in decaf_net.layers.iteritems():
        pydot_nodes[name] = pydot.Node(
            '{%s|%s}' % (name, layer.__class__.__name__), **LAYER_STYLE)
    for name, blob in decaf_net.blobs.iteritems():
        if blob.has_data():
            shapestr = 'x'.join(str(v) for v in blob.data().shape[1:])
            if blob.data().ndim == 1:
                shapestr = 'scalar'
            dtypestr = str(blob.data().dtype)
        else:
            shapestr = 'unknown shape'
            dtypestr = 'unknown dtype'
        pydot_nodes[name] = pydot.Node(
            '{%s|%s|%s}' % (name, shapestr, dtypestr), **BLOB_STYLE)
    for name in pydot_nodes:
        pydot_graph.add_node(pydot_nodes[name])
    # only write explicit edges
    for layername, blobnames in decaf_net.provides.iteritems():
        if layername.startswith(base.DECAF_PREFIX):
            continue
        for blobname in blobnames:
            pydot_graph.add_edge(
                pydot.Edge(pydot_nodes[layername], pydot_nodes[blobname]))
    for layername, blobnames in decaf_net.needs.iteritems():
        if layername.startswith(base.DECAF_PREFIX):
            continue
        for blobname in blobnames:
            pydot_graph.add_edge(
                pydot.Edge(pydot_nodes[blobname], pydot_nodes[layername]))
    return pydot_graph.create(format=ext)

def draw_net_to_file(decaf_net, filename):
    """Draws a decaf net, and saves it to file using the format given as the
    file extension.
    """
    ext = os.path.splitext(filename)[-1][1:]
    with open(filename, 'w') as fid:
        fid.write(draw_net(decaf_net, ext))


class PatchVisualizer(object):
    '''PatchVisualizer visualizes patches.
    '''
    def __init__(self, gap = 1):
        self.gap = gap
    
    def show_single(self, patch):
        """Visualizes one single patch. The input patch could be a vector (in
        which case we try to infer the shape of the patch), a 2-D matrix, or a
        3-D matrix whose 3rd dimension has 3 channels.
        """
        if len(patch.shape) == 1:
            patch = patch.reshape(self.get_patch_shape(patch))
        elif len(patch.shape) > 2 and patch.shape[2] != 3:
            raise ValueError('The input patch shape is incorrect.')
        # determine color
        if len(patch.shape) == 2:
            pyplot.imshow(patch, cmap = cm.gray)
        else:
            pyplot.imshow(patch)
        return patch
    
    def show_multiple(self, patches, ncols = None, bg_func = np.min):
        """Visualize multiple patches. In the passed in patches matrix, each row
        is a patch, in the shape of either n*n or n*n*3, either in a flattened
        format (so patches would be an 2-D array), or a multi-dimensional tensor
        (so patches will be higher dimensional). We will try our best to figure
        out automatically the patch size.
        """
        num_patches = patches.shape[0]
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(num_patches)))
        nrows = int(np.ceil(num_patches / float(ncols)))
        if len(patches.shape) == 2:
            patches = patches.reshape((patches.shape[0],) + 
                                  self.get_patch_shape(patches[0]))
        patch_size_expand = np.array(patches.shape[1:3]) + self.gap
        image_size = patch_size_expand * np.array([nrows, ncols]) - self.gap
        if len(patches.shape) == 4:
            if patches.shape[3] != 3:
                raise ValueError('The input patch shape is incorrect.')
            # color patches
            image_shape = tuple(image_size) + (3,)
            cmap = None
        else:
            image_shape = tuple(image_size)
            cmap = cm.gray
        image = np.ones(image_shape) * bg_func(patches)
        for pid in range(num_patches):
            row = pid / ncols * patch_size_expand[0]
            col = pid % ncols * patch_size_expand[1]
            image[row:row+patches.shape[1], col:col+patches.shape[2]] = \
                    patches[pid]
        # normalize the patches for better viewing results
        image -= np.min(image)
        image /= np.max(image) + np.finfo(np.float64).eps
        pyplot.imshow(image, cmap = cmap, interpolation='nearest')
        pyplot.axis('off')
        return image
    
    def show_channels(self, patch, bg_func = np.min):
        """ This function shows the channels of a patch. The incoming patch
        should have shape [height, width, num_channels], and each channel will
        be visualized as a separate gray patch.
        """
        if len(patch.shape) != 3:
            raise ValueError("The input patch shape %s isn't correct."
                             % str(patch.shape))
        patch_reordered = np.swapaxes(patch.T, 1, 2)
        return self.show_multiple(patch_reordered, bg_func = bg_func)

    def show_blob(self, blob):
        """This function shows a blob by trying really hard to figure out what
        type of blob it is, and what is the best way to visualize it.
        """
        if isinstance(blob, base.Blob):
            data = blob.data()
        else:
            data = blob
        # TODO: figure out how to find the best function of the bg function.
        bg_func = np.max
        if data.ndim == 4:
            # We will first see if the first dim has shape 1, in which case we
            # will just call the 3-dim version.
            if data.shape[0] == 1:
                return self.show_blobs(data[0], bg_func=bg_func)
            elif data.shape[-1] == 3:
                # see if it could be visualized as color images
                return self.show_multiple(data, bg_func=bg_func)
            elif data.shape[-1] == 1:
                # see if it could be visualized as grayscale images
                return self.show_multiple(data[:,:,:,0], bg_func=bg_func)
            else:
                # We will show the blob organized as a two-dimensional tiling of
                # blocks, where the rows are data points and cols are channels.
                num, height, width, channels = data.shape
                data = data.swapaxes(2,3).swapaxes(1,2)
                data = data.reshape(num*channels, height, width)
                return self.show_multiple(data, ncols=channels, bg_func=bg_func)
        elif data.ndim == 3:
            # We will first see if the last dim has shape 1 or 3.
            if data.shape[-1] == 1:
                return self.show_single(data[:,:,0])
            elif data.shape[-1] == 3:
                return self.show_single(data)
            else:
                # If not, we will show the data channel by channel.
                return self.show_channels(data, bg_func=bg_func)
        elif data.ndim == 2:
            # When the data dimension is 2, we do nothing but simply showing
            # the image.
            return self.show_single(data)

    
    def get_patch_shape(self, patch):
        """Gets the patch shape of a single patch. Basically it tries to
        interprete the patch as a square, and also check if it is in color (3
        channels)
        """
        edge_len = np.sqrt(patch.size)
        if edge_len != np.floor(edge_len):
            # we are given color patches
            edge_len = np.sqrt(patch.size / 3.)
            if edge_len != np.floor(edge_len):
                raise ValueError('Cannot determine patch shape from %d.'
                                 % patch.size)
            return (int(edge_len), int(edge_len), 3)
        else:
            edge_len = int(edge_len)
            return (edge_len, edge_len)

_DEFAULT_VISUALIZER = PatchVisualizer()


# Wrappers to utility functions that directly points to functions in the
# default visualizer.

def show_single(*args, **kwargs):
    """Wrapper of PatchVisualizer.show_single()"""
    return _DEFAULT_VISUALIZER.show_single(*args, **kwargs)

def show_multiple(*args, **kwargs):
    """Wrapper of PatchVisualizer.show_multiple()"""
    return _DEFAULT_VISUALIZER.show_multiple(*args, **kwargs)

def show_channels(*args, **kwargs):
    """Wrapper of PatchVisualizer.show_channels()"""
    return _DEFAULT_VISUALIZER.show_channels(*args, **kwargs)

def show_blob(*args, **kwargs):
    """Wrapper of PatchVisualizer.show_blob()"""
    return _DEFAULT_VISUALIZER.show_blob(*args, **kwargs)
