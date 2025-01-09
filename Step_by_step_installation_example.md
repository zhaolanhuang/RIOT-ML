# RIOT-ML Installation and Troubleshooting Guide

This guide provides a step-by-step process for installing RIOT-ML, including solutions for common errors that may occur during installation and usage due to dependencies or configurations.

### Tested Configuration

This guide assumes the following software versions:

* Ubuntu: 24.04.1 LTS
* Python: 3.11
* LLVM: 19.1.4
* CMake: 3.28.3

## Table of Contents
Section 1. Step-by-Step RIOT-ML Installation

Section 2. Troubleshooting Errors During TVM Build

Section 3. Resolving Errors When Running TVM in RIOT-ML

Section 4. Setting Up RIOT-ML After Installation

## Section 1. Step-by-Step RIOT-ML Installation

### Step 1: Clone RIOT-ML Repository
```
(base) git clone --recursive git@github.com:TinyPART/RIOT-ML.git
```

### Step 2: Set Up a Build Environment for TVM

Create and activate a Conda environment for building TVM:
```
(base) conda create -n tvm-build-venv -c conda-forge \
	"llvmdev>=15" \
	"cmake>=3.24" \
	git \
	python=3.11
(base) conda activate tvm-build-venv
```
### Step 3: Build TVM

1. Clone the TVM repository (you can also follow [official docs](https://tvm.apache.org/docs/install/from_source.html)):
```
(tvm-build-venv) git clone --recursive https://github.com/apache/tvm tvm
(tvm-build-venv) cd tvm
```
2. Set up the build directory:
```
(tvm-build-venv)~/tvm$ rm -rf build && mkdir build && cd build
(tvm-build-venv)~/tvm/build$ cp ../cmake/config.cmake .
(tvm-build-venv)~/tvm/build$ nano config.cmake
```
Update config.cmake based on the instructions for TVM installation in the README and [official docs](https://tvm.apache.org/docs/install/from_source.html).

3. Build TVM:
```
(tvm-build-venv)~/tvm/build$ cmake .. && cmake --build . --parallel $(nproc)
```
If the build fails, refer to **section 2 Troubleshooting Errors During TVM Build**.

4. Deactivate the build environment:
```
(tvm-build-venv)~/tvm/build$ conda deactivate
```

### Step 4: Set Up a Working Environment

1. Create and activate a new environment for testing:
```
(base)~/tvm/build$ conda create -n my-test-env python=3.11
(base)~/tvm/build$ conda activate my-test-env
```
2. Set the required environment variables:
```
(my-test-env)~/tvm/build$ export TVM_HOME={path-to-tvm-base-dir}
(my-test-env)~/tvm/build$ export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```
3. Install RIOT-ML tools and dependencies (you can also follow [official docs](https://doc.riot-os.org/getting-started.html#compiling-riot)):
```
(my-test-env)~/tvm/build$ cd ../../RIOT-ML
(my-test-env)~/RIOT-ML$ sudo apt install git gcc-arm-none-eabi make gcc-multilib libstdc++-arm-none-eabi-newlib openocd gdb-multiarch doxygen wget unzip python3-serial 
(my-test-env)~/RIOT-ML$ pip install -r requirements.txt
```

4. Test TVM installation:
```
(my-test-env)~/RIOT-ML$ python -c "import tvm; print(tvm.__file__)"
```
If an error occurs, refer to **section 3 Resolving Errors When Running TVM in RIOT-ML**.

5. Execute model using native:
```
(my-test-env)~/RIOT-ML$ python u-toe.py --per-model --board native ./model_zoo/mnist_0.983_quantized.tflite
```

## Section 2. Troubleshooting Errors During TVM Build
* ### Debugging TVM Build

Enable debug mode in config.cmake by setting:
```
set(USE_RELAY_DEBUG ON)
```

* ### Common Issues and Solutions
#### Error: Missing libbacktrace

Solution:
    
1. Turn off libbacktrace in config.cmake and build again:
```   
set(USE_LIBBACKTRACE OFF)
```
2. Install libbacktrace:
```
sudo apt-get install librust-backtrace libbacktrace-dev
```

## Section 3. Resolving Errors When Running TVM in RIOT-ML
* ### Common Issues and Solutions
#### Error: ModuleNotFoundError: No module named 'tvm'

Solution:
```
pip install -e {path-to-tvm}/python
```
#### Error: OSError: {path-to-miniconda3}/envs/test/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by {path-to-tvm}/build/libtvm.so)

Solution:

1. Install the required GCC version:
```
conda install -c conda-forge gcc=12.1.0
```

2. If the error persists, run:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
```

To check that TVM can be successfully run: 
```
python -c "import tvm; print(tvm.__file__)"
```

## Section 4. Setting Up RIOT-ML After Installation

After building the TVM libraries, you need to set environment variables to ensure Python can find the required packages. Run the following commands:
```
export TVM_HOME={path-to-tvm-base-dir}
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

### If Errors Persist

If you encounter errors mentioned in Section 3, run the following:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD
```




