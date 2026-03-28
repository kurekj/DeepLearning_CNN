import tensorflow as tf
from tensorflow.python.platform import build_info

print("CUDA version:", build_info.build_info['cuda_version'])
print("cuDNN version:", build_info.build_info['cudnn_version'])

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    a = tf.random.normal([2000, 2000])
    b = tf.matmul(a, a)

print(a.device)
print(b)
