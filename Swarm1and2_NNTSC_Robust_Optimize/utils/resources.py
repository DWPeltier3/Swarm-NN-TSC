import os
import tensorflow as tf

def print_resources():
    CPUs = os.cpu_count()
    GPUs = len(tf.config.list_physical_devices('GPU'))
    print('\n*** RESOURCES ***')
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"GPUs available: {GPUs}")
    print(f"CPUs available: {CPUs}") 
    return GPUs