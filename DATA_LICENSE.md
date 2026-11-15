# MNIST data attribution and license

Dataset: **MNIST handwritten digits**, credited to Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. MNIST derives from the original NIST handwritten-character datasets.

The [official Keras MNIST documentation](https://keras.io/api/datasets/mnist/) identifies Yann LeCun and Corinna Cortes as the copyright holders and states that the dataset is distributed under **Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)**.

- License: https://creativecommons.org/licenses/by-sa/3.0/
- Original MNIST page: http://yann.lecun.com/exdb/mnist/
- Archive mirror used: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
- Official Keras loader publishing the archive checksum: https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py
- SHA256: `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`

The redistributed `data/mnist.npz` archive is unchanged. This experiment additionally defines a validation subset, converts pixel values to [0,1] for computation, and creates translated/noisy image derivatives on demand. The original archive and any redistributed image derivatives retain the dataset attribution and CC BY-SA 3.0 terms. The data creators do not endorse this experiment or its conclusions.

These dataset terms do not assert ownership of the original data by this experiment's author and do not replace license notices for separately distributed software.
