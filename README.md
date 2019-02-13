# DISC

DISC adopts deep learning to learn specific patterns within predicted inter-residue contacts and subsequently identifies the objects having these patterns as inter-SSE contacts.

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)

## Need to do

1. Set your config in `./libs/config/config_v1.py`;
2. Specify your raw data path in `read_into_tfrecord.py`, and run `python read_into_tfrecord.py`;
3. run `train.py` for training.
