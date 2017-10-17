CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(NNPACK LANGUAGES C CXX)

# ---[ Options.
SET(BACKEND "auto" CACHE STRING "Backend for micro-kernels implementation")
SET_PROPERTY(CACHE BACKEND PROPERTY STRINGS auto psimd scalar)
OPTION(NNPACK_BUILD_TESTS "Build NNPACK unit tests" ON)

# ---[ CMake options
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
IF(NNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Build flags
IF(NOT MSVC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

IF(NOT CMAKE_SYSTEM_PROCESSOR)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR not defined")
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|armv5te|armv7-a|armv7l|aarch64)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
ENDIF()

IF(BACKEND STREQUAL "auto")
  IF(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    SET(BACKEND "scalar")
  ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET(BACKEND "x86-64")
    SET(BACKEND "psimd") # temporary fix
  ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l|aarch64)$")
    SET(BACKEND "neon")
  ELSE()
    SET(BACKEND "psimd")
  ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---[ cpuinfo library
SET(NNPACK_SRCS
  src/init.c
  src/convolution-inference.c
  src/convolution-input-gradient.c
  src/convolution-kernel-gradient.c
  src/convolution-output.c
  src/fully-connected-inference.c
  src/fully-connected-output.c
  src/pooling-output.c
  src/relu-input-gradient.c
  src/relu-output.c
  src/softmax-output.c)

IF(BACKEND STREQUAL "x86-64")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/x86_64-fma/2d-fourier-8x8.py
    src/x86_64-fma/2d-fourier-16x16.py
    src/x86_64-fma/2d-winograd-8x8-3x3.py
    # Pooling
    src/x86_64-fma/max-pooling.py
    # ReLU
    src/x86_64-fma/relu.py
    # Softmax
    src/x86_64-fma/softmax.py
    src/x86_64-fma/softmax.c
    # Tuple GEMM
    src/x86_64-fma/blas/s8gemm.py
    src/x86_64-fma/blas/c8gemm.py
    src/x86_64-fma/blas/s4c6gemm.py
    # Direct convolution
    src/x86_64-fma/blas/conv1x1.py
    # BLAS microkernels
    src/x86_64-fma/blas/sgemm.py
    src/x86_64-fma/blas/sdotxf.py
    src/x86_64-fma/blas/shdotxf.py)
ELSEIF(BACKEND STREQUAL "scalar")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    scalar/2d-fourier-8x8.c
    scalar/2d-fourier-16x16.c
    scalar/2d-winograd-8x8-3x3.c
    # ReLU and Softmax
    scalar/relu.c
    scalar/softmax.c
    # Tuple GEMM
    scalar/blas/s2gemm.c
    scalar/blas/s2gemm-transc.c
    scalar/blas/cgemm.c
    scalar/blas/cgemm-conjb.c
    scalar/blas/cgemm-conjb-transc.c
    # Direct convolution
    scalar/blas/conv1x1.c
    # BLAS microkernels
    scalar/blas/sgemm.c
    scalar/blas/sdotxf.c
    scalar/blas/shdotxf.c)
ELSEIF(BACKEND STREQUAL "neon")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/psimd/2d-fourier-8x8.c
    src/psimd/2d-fourier-16x16.c
    src/psimd/2d-winograd-8x8-3x3.c
    src/neon/2d-winograd-8x8-3x3.c
    src/neon/2d-winograd-8x8-3x3-fp16.c
    # ReLU
    src/neon/relu.c
    # Softmax
    src/psimd/softmax.c
    # Tuple GEMM
    src/neon/blas/h4gemm.c
    src/neon/blas/s4gemm.c
    src/neon/blas/c4gemm.c
    src/neon/blas/s4c2gemm.c
    src/neon/blas/c4gemm-conjb.c
    src/neon/blas/s4c2gemm-conjb.c
    src/neon/blas/c4gemm-conjb-transc.c
    src/neon/blas/s4c2gemm-conjb-transc.c
    # Direct convolution
    src/neon/blas/conv1x1.c
    # BLAS microkernels
    src/neon/blas/sgemm.c
    src/neon/blas/sdotxf.c
    src/psimd/blas/shdotxf.c)
ELSEIF(BACKEND STREQUAL "psimd")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/psimd/2d-fourier-8x8.c
    src/psimd/2d-fourier-16x16.c
    src/psimd/2d-winograd-8x8-3x3.c
    # ReLU
    src/psimd/relu.c
    # Softmax
    src/psimd/softmax.c
    # Tuple GEMM
    src/psimd/blas/s4gemm.c
    src/psimd/blas/c4gemm.c
    src/psimd/blas/s4c2gemm.c
    src/psimd/blas/c4gemm-conjb.c
    src/psimd/blas/s4c2gemm-conjb.c
    src/psimd/blas/c4gemm-conjb-transc.c
    src/psimd/blas/s4c2gemm-conjb-transc.c
    # Direct convolution
    src/psimd/blas/conv1x1.c
    # BLAS microkernels
    src/psimd/blas/sgemm.c
    src/psimd/blas/sdotxf.c
    src/psimd/blas/shdotxf.c)
ENDIF()

ADD_LIBRARY(nnpack ${NNPACK_SRCS} ${NNPACK_BACKEND_SRCS})
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l)$")
  SET_SOURCE_FILES_PROPERTIES(${NNPACK_BACKEND_SRCS} PROPERTIES COMPILE_FLAGS "-mfpu=neon-fp16")
ENDIF()
SET_TARGET_PROPERTIES(nnpack PROPERTIES PUBLIC_HEADER include/nnpack.h)
TARGET_INCLUDE_DIRECTORIES(nnpack PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE src)

FIND_PACKAGE(PThreadPool REQUIRED)
TARGET_LINK_LIBRARIES(nnpack PUBLIC ${PTHREADPOOL_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnpack PUBLIC ${PTHREADPOOL_INCLUDE_DIRS})

FIND_PACKAGE(FXdiv REQUIRED)
ADD_DEPENDENCIES(nnpack ${FXDIV_DEPENDENCY})
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE ${FXDIV_INCLUDE_DIRS})

FIND_PACKAGE(PSIMD REQUIRED)
ADD_DEPENDENCIES(nnpack ${PSIMD_DEPENDENCY})
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE ${PSIMD_INCLUDE_DIRS})

FIND_PACKAGE(FP16 REQUIRED)
ADD_DEPENDENCIES(nnpack ${FP16_DEPENDENCY})
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE ${FP16_INCLUDE_DIRS})

IF(CMAKE_SYSTEM_NAME STREQUAL "Android")
  ADD_LIBRARY(cpufeatures ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)
  TARGET_INCLUDE_DIRECTORIES(cpufeatures PUBLIC ${ANDROID_NDK}/sources/android/cpufeatures)
  TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE ${ANDROID_NDK}/sources/android/cpufeatures)
  TARGET_LINK_LIBRARIES(cpufeatures PUBLIC dl)
  TARGET_LINK_LIBRARIES(nnpack PRIVATE cpufeatures)
ENDIF()

INSTALL(TARGETS nnpack
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

IF(BUILD_TESTS)
  FIND_PACKAGE(GTest REQUIRED)
ENDIF()
