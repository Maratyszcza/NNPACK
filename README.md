<p align="center"><img src="https://maratyszcza.github.io/NNPACK/NNPACK.png" alt="NNPACK Logo" title="NNPACK"/></p>

# NNPACK

[![BSD (2 clause) License](https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg)](https://github.com/Maratyszcza/NNPACK/blob/master/LICENSE)
[![Build Status](https://img.shields.io/travis/Maratyszcza/NNPACK.svg)](https://travis-ci.org/Maratyszcza/NNPACK)

NNPACK is an acceleration package for neural network computations. NNPACK aims to provide high-performance implementations of convnet layers for multi-core CPUs.

NNPACK is not intended to be directly used by machine learning researchers; instead it provides low-level performance primitives leveraged in leading deep learning frameworks, such as [PyTorch](http://pytorch.org/), [Caffe2](https://caffe2.ai/), [MXNet](http://mxnet.io), 
[tiny-dnn](https://tiny-dnn.readthedocs.io/), [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/), and [Darknet](https://pjreddie.com/darknet/).

## Platforms and requirements

| Environment  | Architecture  | CPU requirements                 |
| ------------ | ------------- | -------------------------------- |
| Linux        | x86-64        | AVX2 and 3-level cache hierarchy |
| Linux        | ARM           | NEON                             |
| Linux        | ARM64         |                                  |
| macOS        | x86-64        | AVX2 and 3-level cache hierarchy |
| Android      | ARM           | NEON                             |
| Android      | ARM64         |                                  |
| Android      | x86           |                                  |
| Android      | x86-64        |                                  |
| iOS          | ARM           |                                  |
| iOS          | ARM64         |                                  |
| Emscripten   | Asm.js        | Temporarily disabled             |
| Emscripten   | WebAssembly   | Temporarily disabled             |
| PNaCl        | Native Client | Temporarily disabled             |

## Features

- Multiple algorithms for convolutiona layers:
  - Fast convolution based on Fourier transform (for kernels up to 16x16 without stride)
  - Fast convolution based on Winograd transform (for 3x3 kernels without stride)
  - Implicit matrix-matrix multiplication algorithm (no limitations)
  - Direct convolution algorithm (for 1x1 kernels without stride)
- Multi-threaded SIMD-aware implementations of neural network layers
- Implemented in C99 and Python without external dependencies
- Extensive coverage with unit tests

## Layers

- Convolutional layer
  - Inference-optimized forward propagation (`nnp_convolution_inference`)
  - Training-optimized forward propagation (`nnp_convolution_output`)
  - Training-optimized backward input gradient update (`nnp_convolution_input_gradient`)
  - Training-optimized backward kernel gradient update (`nnp_convolution_kernel_gradient`)
- Fully-connected layer
  - Inference-optimized forward propagation (`nnp_fully_connected_inference` and `nnp_fully_connected_inference_f16f32` version for FP16 weights)
  - Training-optimized forward propagation (`nnp_fully_connected_output`)
- Max pooling layer
  - Forward propagation, both for training and inference, (`nnp_max_pooling_output`)
- ReLU layer (with parametrized negative slope)
  - Forward propagation, both for training and inference, optionally in-place, (`nnp_relu_output`)
  - Backward input gradient update (`nnp_relu_input_gradient`)
- Softmax layer
  - Forward propagation, both for training and inference, optionally in-place (`nnp_softmax_output`)

## Building

For most users, the recommended way to build NNPACK is through CMake:

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

Note: if `ninja` is not available on your system, configure without `-G Ninja`, and use `make` instead of `ninja`.

### Cross-compilation for Android

To cross-compile for Android, add extra configuration options for `cmake`: `-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.cmake.toolchain` (where `$ANDROID_NDK` is the path to Android NDK directorory, e.g. `/opt/android-ndk-r15c`) **AND** arguments from the table below

| ABI         | Extra cmake args                                    | Restrictions               |
| ----------- | --------------------------------------------------- | -------------------------- |
| armeabi     | `-DANDROID_ABI=armeabi -DANDROID_TOOLCHAIN=gcc`     | Requires CPU with ARM NEON |
| armeabi-v7a | `-DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN=gcc` | Requires CPU with ARM NEON |
| arm64-v8a   | `-DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN=clang` | Requires clang toolchain   |
| x86         | `-DANDROID_ABI=x86`                                 |                            |
| x86_64      | `-DANDROID_ABI=x86_64`                              |                            |

Notes:
- On **armeabi** and **armeabi-v7a** `nnp_initialize` will fail with `nnp_status_unsupported_hardware` if the mobile CPU does not support ARM NEON. Don't set `-DANDROID_ARM_NEON=1` for NNPACK compilation as it can make `nnp_initialize` crash on CPUs without ARM NEON.
- NNPACK builds for **armeabi** and **armeabi-v7a** are up to 2x slower if you use **clang** toolchain.
- **mips** and **mips64** are not supported, and we have no plans to add it (pull request would be welcome, though)
- **x86_64** build will use generic 128-bit (SSE2) micro-kernels rather than AVX2 micro-kernels in native build

### Development builds

For NNPACK development, we recommend a different workflow, supported on macOS and Linux.

Install [ninja](https://ninja-build.org) build system
```bash
sudo apt-get install ninja-build || brew install ninja
```

Install [PeachPy](https://github.com/Maratyszcza/PeachPy) assembler and [confu](https://github.com/Maratyszcza/confu) configuration system
```bash
[sudo] pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
[sudo] pip install --upgrade git+https://github.com/Maratyszcza/confu
```

Then clone NNPACK, install dependencies, configure, and build
```bash
git clone https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
confu setup
python configure.py
ninja
ninja smoketest # or `ninja test`
```

#### Cross-compilation
- To cross-compile NNPACK for Android, set `$ANDROID_NDK` and `$ANDROID_SDK` variables, and add an extra configuration argument `--target=arm-android`, `--target=arm64-android`, `--target=x86-android`, or `--target=x86_64-android` to target, respectively, **armeabi-v7a**, **arm64-v8a**, **x86**, or **x86_64** Android ABI.
- To cross-compile NNPACK for Asm.js or WebAssembly, download, build, and activate Emscripten SDK, and use configuration argument `--target=wasm` or `--target=asmjs`
- To cross-compile NNPACK for Portable Native Client, download Native Client SDK, set `NACL_SDK_ROOT` variable, and configure NNPACK with `--target=pnacl` option.

## Ecosystem

### Deep Learning Frameworks
- [Caffe2](http://caffe2.ai) natively supports NNPACK
- [MXNet](http://mxnet.io) - supports NNPACK for inference in convolutional layers, fully-connected, and max-pooling layers. See [MXNet wiki](https://mxnet.incubator.apache.org/how_to/nnpack.html) for configuration instructions and performance benchmarks).
- [PyTorch](http://pytorch.org/) - supports NNPACK as an optional dependency.
- [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) - header-only deep learning framework in C++11, which natively supports NNPACK.
- [darknet-nnpack](https://github.com/thomaspark-pkj/darknet-nnpack) - fork of [Darknet](https://pjreddie.com/darknet/) framework with NNPACK support.
- [szagoruyko/nnpack.torch](https://github.com/szagoruyko/nnpack.torch) - integration of NNPACK into Lua Torch via ffi
- [Maratyszcza/caffe](https://github.com/Maratyszcza/caffe) - up-to-date integration of NNPACK (convolutional, fully-connected, max-pooling, and ReLU layers) into Caffe based on `nnpack-pr` branch in [ajtulloch/caffe](https://github.com/ajtulloch/caffe/tree/nnpack-pr).
- [Maratyszcza/caffe-nnpack](https://github.com/Maratyszcza/caffe-nnpack) - older and unmaintained integration of NNPACK (convolutional layers only) into Caffe.
- See also discussion in [Issue #1](https://github.com/Maratyszcza/NNPACK/issues/1)

### Languages and Environments
- [nnpack-windows](https://github.com/zeno40/nnpack-windows) - unofficial port for Windows
- [node-nnpack](https://www.npmjs.com/package/node-nnpack) - Node.js bindings
- [peterhj/libnnpack](https://github.com/peterhj/libnnpack) - Rust bindings

### Users

- [Facebook](https://www.facebook.com) uses NNPACK in production.
- [Prisma](https://prisma-ai.com) uses NNPACK in the mobile app.

## Acknowledgements

[![HPC Garage logo](https://github.com/Maratyszcza/PeachPy/blob/master/logo/hpcgarage.png)](http://hpcgarage.org)
[![Georgia Tech College of Computing logo](https://github.com/Maratyszcza/PeachPy/blob/master/logo/college-of-computing.gif)](http://www.cse.gatech.edu/)

The library is developed by [Marat Dukhan](http://www.maratdukhan.com) of Georgia Tech with extensive advice from [Nicolas Vasilache](https://research.facebook.com/nicolas-vasilache) and [Soumith Chintala](http://soumith.ch/) of Facebook Artificial Intelligence Research. [Andrew Tulloch](http://tullo.ch/) of Facebook Artificial Intelligence Research contributed Caffe integration. We thank [Andrew Lavin](https://github.com/andravin) for fruitful discussions on Winograd transform-based implementations. NNPACK is a research project at [Richard Vuduc](http://vuduc.org)'s HPC Garage lab in the Georgia Institute of Technology, College of Computing, School of Computational Science and Engineering.

This material is based upon work supported by the U.S. National Science Foundation (NSF) Award Number 1339745. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of NSF.
