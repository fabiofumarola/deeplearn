# Deeply

this project contains baseline example plus demo for deep learning tasks.

## Setup


### Nvidia Drivers

1. download nvidia drivers from [nvidia drivers]()http://www.nvidia.it/Download/index.aspx?lang=it-it
2. install `yum install -y kernel-devel kernel-headers gcc make bzip2 hdf5 git gcc-c++`
3. run `nvidia-smi` to get gpu informations

### Miniconda

1. install [Miniconda](http://conda.pydata.org/miniconda.html)

#### Environment

The first step is to create a conda env to handle dependencies

```bash
conda create --name gpu_python3 python=3.4
```

Then start the environment

```bash
source activate gpu_python3
```

**Optional: Install CUDA (GPUs on Linux)**

[Download and install Cuda Toolkit 7.0])https://developer.nvidia.com/cuda-toolkit-70)

    ```bash
    wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-rhel7-7-0-local-7.0-28.x86_64.rpm
    chmod +x cuda-repo-rhel7-7-0-local-7.0-28.x86_64.rpm 
    rpm -ivh cuda-repo-rhel7-7-0-local-7.0-28.x86_64.rpm 
    yum clean expire-cache
    yum update
    yum install cuda
    ```
    
    [official documentation](http://developer.download.nvidia.com/compute/cuda/7_0/Prod/doc/CUDA_Getting_Started_Linux.pdf)

[Download and install CUDNN Toolkit 6.5](https://developer.nvidia.com/rdp/cudnn-archive)

please refere to [Tensorflow documentation](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) as example

### Tensorflow GPU and CPU support


```bash
pip install --upgrade pip
pip install -I --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl

```

### Keras.io

Installation prerequisites

```bash
conda install scipy pyyaml h5py
```


#### Install Theano support

```bash

pip install git+https://github.com/Theano/Theano.git
```

To install keras:

```bash
pip install keras
```

create a file with confs

```bash

nano ~/.keras/keras.json

# set backend to theano to switch to theano
{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}

```