# Check-Tensorflow-Use-GPU

import tensorflow as tf
print(tf.__version__)

tf.config.list_physical_devices()

tf.test.is_built_with_cuda()

tf.test.is_gpu_available()

from __future__ import absolute_import, division, print_function, unicode_literals
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))

print(tf.constant('Hello from TensorFlow ' + tf.__version__) )

tf.test.gpu_device_name()

from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

