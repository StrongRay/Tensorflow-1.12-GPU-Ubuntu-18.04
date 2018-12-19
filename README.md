# Tensorflow-1.12-GPU-Ubuntu-18.04
Building Tensorflow 1.12 with GPU from scratch

Why this is needed:  Not every hardware is the same, some will have GPU, some have different CUDA versions, etc.  Building from scratch helps cater to your specific hardware especially your combination of versions of python, CUDA, CUDNN and NCCL   And this helps to fine tune your UNIX skills to “debug” where one has gone wrong and how to search for solutions off the internet. 

## 1.  Get the Tensorflow source codes

and from your home directory
```
git clone https://github.com/tensorflow/tensorflow.git  
cd tensorflow
git checkout r1.12 (using r1.12 instead of r1.13 )
```
## 2.  Check your Bazel version

Bazel **MUST BE 0.18 and not HIGHER** , I tried 0.20 and failed and so have to undo and reinstall bazel.
Uninstall other bazel version if another is already installed (check with **bazel version**)
```
***Use the binary installer method***
go to https://github.com/bazelbuild/bazel/tags and download the bazel-0.18.0-installer-linux-x86_64.sh
the file will likely be downloaded to /home/xxxx/Downloads directory
```

```bazel shutdown
rm -fr ~/.bazel ~/.bazelrc ~/.cache/bazel
cd Downloads
chmod +x bazel-0.18.0-installer-linux-x86_64.sh
./bazel-0.18.0-installer-linux-x86_64.sh –user
bazel version  
```
```
Build label: 0.18.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Fri Nov 2 10:09:48 2018 (1541153388)
Build timestamp: 1541153388
Build timestamp as int: 1541153388
```

## 3.  Verify a few stuff before starting the configure.

a. Identify your python version  3.6 (add to ~/.bashrc - alias python=python3 )  
b. Make sure nvcc is installed ( use **nvcc –version** to verify )  
c. Check CUDA version 9.2  ( use nvcc –version )  
d. Check CUDNN version  7.2.1 ( **locate cudnn | grep “libcudnn.so” | tail -n1** )   
e. Check NCCL 2.2.1 ( **locate nccl | grep “libnccl.so” | tail -n1** )  
f. Check **nvidia-smi** to see your GPU is working well 
```
Wed Dec 19 18:54:19 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.44                 Driver Version: 396.44                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce 940MX       Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P0    N/A /  N/A |    241MiB /  2004MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1743      G   /usr/lib/xorg/Xorg                           210MiB |
|    0     19048      C   /usr/lib/libreoffice/program/soffice.bin      20MiB |
+-----------------------------------------------------------------------------+
```
My gcc version is (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0 works well
```
./configure
```
    Python library paths: /usr/bin/python3
    Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: Y
    Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: Y
    Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
    Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
    Do you wish to build TensorFlow with XLA JIT support? [y/N]: N
    Do you wish to build TensorFlow with GDR support? [y/N]: N
    Do you wish to build TensorFlow with VERBS support? [y/N]: N
    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
    Do you wish to build TensorFlow with CUDA support? [y/N]: Y
    Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.2
    Please specify the location where CUDA 9.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda
    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.2.1
    Please specify the location where cuDNN 7.2.1 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/usr/lib/x86_64-linux-gnu
    Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1] 5.0
    Do you want to use clang as CUDA compiler? [y/N]: N
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc
    Do you wish to build TensorFlow with MPI support? [y/N]: N
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=native
    If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N

    Configuration finished [there was an NCCL question asked but I did not capture it here.. ]


## 3.  Do the build 

### a.   Start the bazel build [from your tensorflow directory .. for me was /home/kenghee/tensorflow
```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```
Go for lunch .. and tea break , took me 4.13 hours on a Intel® Core™ i7-7500U CPU @ 2.70GHz × 4 , 12 GB memory laptop.
```
Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 14898.186s, Critical Path: 249.87s
INFO: 14074 processes: 14074 local.
INFO: Build completed successfully, 17481 total actions
```
### b.  Build the wheel file
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
pip3 install tensorflow_pkg/tensorflow*
```
```
Wed Dec 19 18:19:17 +08 2018 : === Preparing sources in dir: /tmp/tmp.4VF8RATQrH
~/tensorflow ~/tensorflow
~/tensorflow
Wed Dec 19 18:19:32 +08 2018 : === Building wheel
warning: no files found matching '*.pd' under directory '*'
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
warning: no files found matching '*.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '*' under directory 'tensorflow/include/Eigen'
warning: no files found matching '*.h' under directory 'tensorflow/include/google'
warning: no files found matching '*' under directory 'tensorflow/include/third_party'
warning: no files found matching '*' under directory 'tensorflow/include/unsupported'
Wed Dec 19 18:20:09 +08 2018 : === Output wheel file is in: /home/kenghee/tensorflow/tensorflow_pkg
```
### c.  Install the WHEEL file

Before installing make sure you remove any existing instance of Tensorflow 
```
pip3 list  [checks the list of modules installed for python3 ]
pip3 uninstall tensorflow
pip3 uninstall tensorboard
```
from the tensorflow directory [/home/xxxx/tensorflow]
```
pip3 install ./tensorflow_pkg/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
```
## 4. Test the build

Create a python file
```
nano test-tf.py

import tensorflow as tf

print(tf.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  #  R2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

sess.close()
```
python test-tf.py
```
1.12.0
2018-12-19 18:24:11.670236: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-12-19 18:24:11.670851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce 940MX major: 5 minor: 0 memoryClockRate(GHz): 1.189
pciBusID: 0000:01:00.0
totalMemory: 1.96GiB freeMemory: 1.83GiB
2018-12-19 18:24:11.670867: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2018-12-19 18:24:11.943173: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-19 18:24:11.943205: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2018-12-19 18:24:11.943211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2018-12-19 18:24:11.943363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1587 MB memory) -> physical GPU (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0, compute capability: 5.0)
```
We are good to go! Somehow this build appears faster than the standard pip3 install tensorflow.  
