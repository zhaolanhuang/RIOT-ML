RIOT-ML: Toolkit for Over-the-Air Secure Updates and Performance Evaluation of TinyML Models
=====================
If you cite this work, please use: 
```
Z. Huang, K. Zandberg, K. Schleiser and E. Baccelli. RIOT-ML: Toolkit for Over-the-Air Secure Updates and Performance Evaluation of TinyML Models. Annals of Telecommunications, 2024 (to appear).
```
For more information on this work, please consult the preprint on [Inria HAL](https://inria.hal.science/hal-04506370v1/document)

# Get Source Code

It is important to clone the submodules along, with `--recursive` option.

```
git clone --recursive git@github.com:TinyPART/RIOT-ML.git
```

Or open the git shell after clone, and type the following command.

```
git submodule init
git submodule update
```

# Prequisites
This toolkit is tested under:
- Linux Mint 21.1 (5.15.0-58-generic)
- Python 3.10.8
- LLVM 14.0.0
- CMAKE 3.22.1


## Install TVM from Source
First, Follow the instruction from [official docs](https://tvm.apache.org/docs/install/from_source.html).

It is recommended to switch on the following options in `config.cmake`:
```
USE_GRAPH_EXECUTOR 
USE_PROFILER 
USE_MICRO
USE_LLVM
```
For CUDA users, it is recommended to switch `USE_CUDA` on.  
For PyTorch users, it is recommended to set `(USE_LLVM "/path/to/llvm-config --link-static")` and `set(HIDE_PRIVATE_SYMBOLS ON)`.

After built up the TVM libraries, set the environment variable to tell python where to find the packages:
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
### Patch Graph Debug Executor
The vanilla executor in TVM calls inexistent *__nop* operators, which will crash the RPC runtime on the board. If you want to try the Per-Operator evaluation, please patch the executor and (re)build TVM with the following commands:

```
cp <path-to-utoe>/patch/fixup_skip_nop_op.patch /path/to/tvm
cd /path/to/tvm
git apply fixup_skip_nop_op.patch
( (re)build tvm )

```

## Install RIOT Toolchains
Please refer to the doc [Getting started - Compiling RIOT](https://doc.riot-os.org/getting-started.html#compiling-riot)

For RISC-V toolchain, please enable __multilib__ support.

## Install Python Packages
Use the command below to install the dependencies:
```
pip install -r requirements.txt
```
## (Optional) FIT IoT-LAB
To use remote boards on [FIT IoT-LAB](https://www.iot-lab.info/docs/getting-started/introduction/), please first [register a testbed account](https://www.iot-lab.info/docs/getting-started/user-account/) and set up [SSH access](https://www.iot-lab.info/docs/getting-started/ssh-access/). After that, use following commands to store the credentials for U-TOE:

```
iotlab-auth -u <login user name>
iotlab-auth --add-ssh-key
```
# Usage
! FOR PYTORCH USER ! Please first [transform your model in *TorchScript* representation](#how-to-torchscript-your-model) !

Before running the evaluation, please get your model files ready from ML frameworks (TFLite, PyTorch etc.). You can find some model file examples in `model_zoo` folder.

## Model Update
Please refer to [Running Model Update on IoT Boards](https://github.com/zhaolanhuang/RIOT-ML/blob/main/examples/README.model_update.md).

## CLI of U-TOE

U-TOE provides a command line interface (CLI) entry `u-toe.py`, see below:

```
> python u-toe.py -h

usage: u-toe.py [-h] [--per-model | --per-ops] [--board BOARD] [--use-iotlab] [--iotlab-node IOTLAB_NODE] [--random-seed RANDOM_SEED] [--trials-num TRIALS_NUM] model_file

positional arguments:
  model_file            path to machine leearning model file.

options:
  -h, --help            show this help message and exit
  --per-model           Per-Model Evaluation (default).
  --per-ops             Per-Operator Evaluation.
  --board BOARD         IoT board name
  --use-iotlab          use remote board in FIT IoT-LAB.
  --iotlab-node IOTLAB_NODE
                        remote node url. It will start an new experiment if this field is empty.
  --random-seed RANDOM_SEED
                        default: 42
  --trials-num TRIALS_NUM
                        defalut: 10
```

## Per-Model Evaluation
- Local example:

```
python u-toe.py --per-model --board stm32f746g-disco ./model_zoo/mnist_0.983_quantized.tflite
```
This command will start a Per-Model evaluation on local board `stm32f746g-disco`, using `LeNet-5 INT8` model.

- FIT-IoT Lab example:

```
python u-toe.py --per-model --use-iotlab  --board iotlab-m3 ./model_zoo/mnist_0.983_quantized.tflite
```
This command will start a IoT-Lab experiment and execute a Per-Model evaluation on remote board `iotlab-m3`, using `LeNet-5 INT8` model.

- Output example:
```
Board        Memory (KB)    Storage (KB)  95-CI (ms)          Mean (ms)    Median (ms)    Min. (ms)    Max. (ms)
---------  -------------  --------------  ----------------  -----------  -------------  -----------  -----------
iotlab-m3          11.08          65.232  [97.739, 97.757]       97.748         97.751       97.733       97.764
```
## Per-Operator Evaluation
! Please first [patch TVM executor](#patch-graph-debug-executor) before trying out this feature. !

- Local example
```
python u-toe.py --per-ops --board stm32f746g-disco ./model_zoo/sinus_float.tflite
```
This command will start a Per-Operator evaluation on local board `stm32f746g-disco`, using `sinus` model.

- FIT IoT-Lab Example: comming soon...

- Output example

```
Ops                                            Time (us)    Time (%)  Params          Memory (KB)    Storage (KB)
-------------------------------------------  -----------  ----------  ------------  -------------  --------------
tvmgen_default_fused_nn_dense_add_nn_relu          8.853      15.217  ['p0', 'p1']          0.128           0.128
tvmgen_default_fused_nn_dense_add_nn_relu_1       46.682      80.236  ['p2', 'p3']          0.128           1.088
tvmgen_default_fused_nn_dense_add                  2.646       4.548  ['p4', 'p5']          0.02            0.068
```
## How-to: TorchScript your model
The following code is adapted from https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html

```
# the vanilla pytorch model
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save("torchscrpited_model.pth")
```

# Model Zoo
| Model                  | Task                 | Description                                               | File name                    |
| ---------------------- | -------------------- | --------------------------------------------------------- | ---------------------------- |
| LeNet-5 INT8           | Image Classification | Quantized LeNet-5 in INT8, trained on MINST dataset       | mnist_0.983_quantized.tflite |
| MobileNetV1-0.25x INT8 | Visual Wake Words    | Quantized MobileNetV1 in INT8, with width multiplier 0.25 | vww_96_int8.tflite           |
| DS-CNN Small INT8      | Keyword Spotting     | Quantized depthwise separable CNN in INT8                 | ds_cnn_s_quantized.tflite    |
| Deep AutoEncoder INT8  | Anomaly Detection    | Quantized deep autoencoder in INT8                        | ad01_int8.tflite             |
| RNNoise INT8           | Noise Suppression    | Quantized GRU-based network in INT8                       | rnnoise_INT8.tflite          |
| Sinus                  | Regression           | TFLite sine value example                                 | sinus_float.tflite           |


# Benchmark
We benchmarked various IoT boards with representative ML models, which can be found in the model zoo.
## Per-Model Eval on LeNet-5 INT8
| Board / MCU                    | Core             | Memory (KB) | Storage (KB) | Computational Latency   (ms) |         |         |         |
|--------------------------------|------------------|-------------|--------------|------------------------------|---------|---------|---------|
|                                |                  |             |              | 95%-CI                       | Median  | Min.    | Max.    |
| b-l072z-lrwan1 / STM32L072CZ   | M0+ @ 32 MHz     | 11.288      | 64.34        | [261.829, 262.249]       | 262.187 | 261.35  | 262.216 |
| samr21-xpro / ATSAMR21G18A   | M0+ @ 48 MHz     | 11.292      | 64.956       | [182.058, 182.083]       | 182.068 | 182.04  | 182.097 |
| samr30-xpro / ATSAMR30G18A     | M0+ @ 48 MHz     | 11.208      | 65.168       | [176.936, 176.965]       | 176.958 | 176.924 | 176.975 |
| samr34-xpro / ATSAMR34J18      | M0+ @ 48 MHz     | 11.296      | 65.436       | [178.686, 178.718]       | 178.708 | 178.669 | 178.732 |
| arduino-zero   / ATSAMD21G18   | M0+ @ 48 MHz     | 11.292      | 64.94        | [182.061, 182.082]       | 182.068 | 182.051 | 182.098 |
| openmote-b   / CC2538SF53      | M3 @ 32 MHz      | 11.1        | 66.08        | [200.337, 200.384]       | 200.367 | 200.323 | 200.404 |
| IoT-LAB M3 / STM32F103REY      | M3 @ 72 MHz      | 11.296      | 62.26        | [97.74, 97.757]          | 97.751  | 97.733  | 97.764  |
| nucleo-wl55jc   / STM32WL55JC  | M4 @ 48 MHz      | 11.288      | 63.18        | [98.649, 98.668]         | 98.661  | 98.637  | 98.679  |
|  nrf52dk   / nRF52832         | M4 @ 64 MHz      | 11.328      | 61.012       | [66.124, 66.152]         | 66.132  | 66.096  | 66.158  |
| nrf52840dk   / nRF52840        | M4 @ 64 MHz      | 11.348      | 61.332       | [66.078, 66.112]         | 66.088  | 66.087  | 66.163  |
| b-l475e-iot01a   / STM32L475VG | M4 @ 80 MHz      | 11.288      | 61.604       | [52.9, 52.901]           | 52.901  | 52.9    | 52.902  |
| stm32f746g-disco / STM32F746NG | M7 @ 216 MHz     | 11.076      | 64.712       | [39.6, 39.602]           | 39.601  | 39.599  | 39.604  |
| esp32-wroom-32 / ESP32-D0WDQ6  | ESP32 @ 80 MHz   | 115.958     | 157.719      | [85.58, 85.583]          | 85.582  | 85.576  | 85.584  |
| hifive1b / SiFive FE310-G002   | RISC-V @ 320 MHz | 60.884      | 66.492       | [153.621, 154.166]       | 153.747 | 153.717 | 154.938 |
