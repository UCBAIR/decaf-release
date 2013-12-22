Decaf
=====

Decaf is a framework that implements convolutional neural networks, with the
goal of being efficient and flexible. It allows one to easily construct a
network in the form of an arbitrary Directed Acyclic Graph (DAG) and to
perform end-to-end training.

For more usage check out [the wiki](https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki).
A great place to start is running [ImageNet classification on an image](https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet).

For the pre-trained imagenet DeCAF feature and its analysis, please see our
[technical report on arXiv](http://arxiv.org/abs/1310.1531). Please consider
citing our paper if you use Decaf in your research:

    @article{donahue2013decaf,
      title={DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition},
      author={Donahue, Jeff and Jia, Yangqing and Vinyals, Oriol and Hoffman, Judy and Zhang, Ning and Tzeng, Eric and Darrell, Trevor},
      journal={arXiv preprint arXiv:1310.1531},
      year={2013}
    }

For Anaconda users experiencing libm error: it is because anaconda ships with a libm.so binary that does not support GLIBC_2.15, which gets loaded earlier than the system libm. You can fix this error by you can replacing anaconda's libm file with a newer version.
