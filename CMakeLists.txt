CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(NNPACK C CXX ASM)

# ---[ Options.
SET(NNPACK_BACKEND "auto" CACHE STRING "Backend for micro-kernels implementation")
SET_PROPERTY(CACHE NNPACK_BACKEND PROPERTY STRINGS auto psimd scalar)
OPTION(NNPACK_CONVOLUTION_ONLY "Build only NNPACK functions for convolutional layer" OFF)
OPTION(NNPACK_INFERENCE_ONLY "Build only NNPACK functions for inference" OFF)
OPTION(NNPACK_CUSTOM_THREADPOOL "Build NNPACK for custom thread pool" OFF)
SET(NNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE NNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(NNPACK_BUILD_TESTS "Build NNPACK unit tests" ON)

# ---[ CMake options
IF(NNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

MACRO(NNPACK_TARGET_ENABLE_C99 target)
  IF(${CMAKE_VERSION} VERSION_LESS "3.1")
    IF(NOT MSVC)
      TARGET_COMPILE_OPTIONS(${target} PRIVATE -std=c99)
    ENDIF()
  ELSE()
    SET_TARGET_PROPERTIES(${target} PROPERTIES
      C_STANDARD 99
      C_EXTENSIONS YES)
  ENDIF()
ENDMACRO()

MACRO(NNPACK_TARGET_ENABLE_CXX11 target)
  IF(${CMAKE_VERSION} VERSION_LESS "3.1")
    IF(NOT MSVC)
      TARGET_COMPILE_OPTIONS(${target} PRIVATE -std=gnu++11)
    ENDIF()
  ELSE()
    SET_TARGET_PROPERTIES(${target} PROPERTIES
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS YES)
  ENDIF()
ENDMACRO()

# ---[ Build flags
IF(NOT CMAKE_SYSTEM_PROCESSOR)
  IF(NOT IOS)
    MESSAGE(WARNING "CMAKE_SYSTEM_PROCESSOR is not defined, automatic configuration may choose suboptimal options")
  ENDIF()
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|armv5te|armv7-a|armv7l|aarch64)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
ENDIF()

IF(NNPACK_BACKEND STREQUAL "auto")
  IF(IOS)
    SET(NNPACK_BACKEND "neon")
  ELSEIF(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    SET(NNPACK_BACKEND "scalar")
  ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR)
    MESSAGE(WARNING "CMAKE_SYSTEM_PROCESSOR is not defined, using PSIMD backend")    
    SET(NNPACK_BACKEND "psimd")
  ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    SET(NNPACK_BACKEND "x86-64")
  ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l|aarch64)$")
    SET(NNPACK_BACKEND "neon")
  ELSE()
    SET(NNPACK_BACKEND "psimd")
  ENDIF()
ENDIF()

IF(NNPACK_BACKEND STREQUAL "x86-64")
  # ---[ We need a Python interpreter to build PeachPy sources for x86-64
  FIND_PACKAGE(PythonInterp)
  IF(NOT PYTHONINTERP_FOUND)
    MESSAGE(STATUS "Python interpreter not found; and PeachPy sources for x86-64 backend can't be built without it. Fall back to PSIMD backend")
    SET(NNPACK_BACKEND "psimd")
  ENDIF()
ENDIF()

# ---[ Download deps
SET(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

IF(NNPACK_BACKEND STREQUAL "x86-64")
  IF(NOT DEFINED PYTHON_SIX_SOURCE_DIR)
    MESSAGE(STATUS "Downloading six (Python package) to ${CONFU_DEPENDENCIES_SOURCE_DIR}/six (define PYTHON_SIX_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadSix.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/six-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/six-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/six-download")
    SET(PYTHON_SIX_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/six" CACHE STRING "six (Python package) source directory")
  ENDIF()

  IF(NOT DEFINED PYTHON_ENUM_SOURCE_DIR)
    IF(${PYTHON_VERSION_STRING} VERSION_LESS 3.4)
      # ---[ Python < 3.4 does not natively support enums, and needs a polyfill
      MESSAGE(STATUS "Downloading enum (Python package) to ${CONFU_DEPENDENCIES_SOURCE_DIR}/enum (define PYTHON_ENUM_SOURCE_DIR to avoid it)")
      CONFIGURE_FILE(cmake/DownloadEnum.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/enum-download/CMakeLists.txt")
      EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/enum-download")
      EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/enum-download")
      SET(PYTHON_ENUM_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/enum" CACHE STRING "enum (Python package) source directory")
    ELSE()
      SET(PYTHON_ENUM_SOURCE_DIR "" CACHE STRING "enum (Python package) source directory")
    ENDIF()
  ENDIF()

  IF(NOT DEFINED PYTHON_PEACHPY_SOURCE_DIR)
    # ---[ PeachPy requires Opcodes for installation
    IF(NOT DEFINED PYTHON_OPCODES_SOURCE_DIR)
      MESSAGE(STATUS "Downloading opcodes (Python package) to ${CONFU_DEPENDENCIES_SOURCE_DIR}/opcodes (define PYTHON_OPCODES_SOURCE_DIR to avoid it)")
      CONFIGURE_FILE(cmake/DownloadOpcodes.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/opcodes-download/CMakeLists.txt")
      EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/opcodes-download")
      EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/opcodes-download")
      SET(PYTHON_OPCODES_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/opcodes" CACHE STRING "opcodes (Python package) source directory")
    ENDIF()

    MESSAGE(STATUS "Downloading PeachPy (Python package) to ${CONFU_DEPENDENCIES_SOURCE_DIR}/peachpy (define PYTHON_PEACHPY_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadPeachPy.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/peachpy-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/peachpy-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/peachpy-download")
    SET(PYTHON_PEACHPY_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/peachpy" CACHE STRING "PeachPy (Python package) source directory")
  ENDIF()

  IF(${PYTHON_VERSION_STRING} VERSION_LESS 3.4)
    # ---[ Python < 3.4 does not natively support enums, and needs a polyfill
    SET(PEACHPY_PYTHONPATH "${PYTHON_SIX_SOURCE_DIR}:${PYTHON_ENUM_SOURCE_DIR}:${PYTHON_PEACHPY_SOURCE_DIR}")
  ELSE()
    SET(PEACHPY_PYTHONPATH "${PYTHON_SIX_SOURCE_DIR}:${PYTHON_PEACHPY_SOURCE_DIR}")
  ENDIF()
ENDIF()

IF(NOT DEFINED CPUINFO_SOURCE_DIR)
  MESSAGE(STATUS "Downloading cpuinfo to ${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo (define CPUINFO_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadCpuinfo.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
  SET(CPUINFO_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo" CACHE STRING "cpuinfo source directory")
ENDIF()

IF(NOT DEFINED FP16_SOURCE_DIR)
  MESSAGE(STATUS "Downloading FP16 to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16 (define FP16_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadFP16.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  SET(FP16_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16" CACHE STRING "FP16 source directory")
ENDIF()

IF(NOT DEFINED FXDIV_SOURCE_DIR)
  MESSAGE(STATUS "Downloading FXdiv to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv (define FXDIV_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadFXdiv.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  SET(FXDIV_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv" CACHE STRING "FXdiv source directory")
ENDIF()

IF(NOT DEFINED PSIMD_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PSimd to ${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd (define PSIMD_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  SET(PSIMD_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd" CACHE STRING "PSimd source directory")
ENDIF()

IF(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
  MESSAGE(STATUS "Downloading pthreadpool to ${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool (define PTHREADPOOL_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPThreadPool.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  SET(PTHREADPOOL_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool" CACHE STRING "pthreadpool source directory")
ENDIF()

IF(NNPACK_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  SET(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
ENDIF()

# ---[ NNPACK library
SET(NNPACK_INIT_SRCS src/init.c)
SET(NNPACK_LAYER_SRCS src/convolution-inference.c)
IF(NOT NNPACK_CONVOLUTION_ONLY)
  LIST(APPEND NNPACK_LAYER_SRCS
    src/fully-connected-inference.c
    src/pooling-output.c
    src/relu-output.c
    src/softmax-output.c)
  IF(NOT NNPACK_INFERENCE_ONLY)
    LIST(APPEND NNPACK_LAYER_SRCS
      src/fully-connected-output.c
      src/relu-input-gradient.c)
  ENDIF()
ENDIF()
IF(NOT NNPACK_INFERENCE_ONLY)
  LIST(APPEND NNPACK_LAYER_SRCS
    src/convolution-input-gradient.c
    src/convolution-kernel-gradient.c
    src/convolution-output.c)
ENDIF()

SET(NNPACK_REFERENCE_LAYERS_SRCS
  src/ref/convolution-output.c
  src/ref/convolution-input-gradient.c
  src/ref/convolution-kernel.c
  src/ref/fully-connected-output.c
  src/ref/max-pooling-output.c
  src/ref/softmax-output.c
  src/ref/relu-output.c
  src/ref/relu-input-gradient.c)

SET(NNPACK_REFERENCE_FFT_SRCS
  src/ref/fft/aos.c
  src/ref/fft/soa.c
  src/ref/fft/forward-real.c
  src/ref/fft/forward-dualreal.c
  src/ref/fft/inverse-real.c
  src/ref/fft/inverse-dualreal.c)

IF(NNPACK_BACKEND STREQUAL "x86-64")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/x86_64-fma/2d-fourier-8x8.py
    src/x86_64-fma/2d-fourier-16x16.py
    src/x86_64-fma/2d-winograd-8x8-3x3.py
    # Tuple GEMM
    src/x86_64-fma/blas/s8gemm.py
    src/x86_64-fma/blas/c8gemm.py
    src/x86_64-fma/blas/s4c6gemm.py
    # Direct convolution
    src/x86_64-fma/blas/conv1x1.py
    # BLAS microkernels
    src/x86_64-fma/blas/sgemm.py)
  IF(NOT NNPACK_CONVOLUTION_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # Pooling
      src/x86_64-fma/max-pooling.py
      # ReLU
      src/x86_64-fma/relu.py
      # Softmax
      src/x86_64-fma/softmax.py
      src/x86_64-fma/softmax.c
      # BLAS microkernels
      src/x86_64-fma/blas/sdotxf.py
      src/x86_64-fma/blas/shdotxf.py)
  ENDIF()
ELSEIF(NNPACK_BACKEND STREQUAL "scalar")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/scalar/2d-fourier-8x8.c
    src/scalar/2d-fourier-16x16.c
    src/scalar/2d-winograd-8x8-3x3.c
    # Tuple GEMM
    src/scalar/blas/s2gemm.c
    src/scalar/blas/cgemm-conjb.c
    # Direct convolution
    src/scalar/blas/conv1x1.c
    # BLAS microkernels
    src/scalar/blas/sgemm.c)
  IF(NOT NNPACK_CONVOLUTION_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # ReLU and Softmax
      src/scalar/relu.c
      src/scalar/softmax.c
      # BLAS microkernels
      src/scalar/blas/sdotxf.c
      src/scalar/blas/shdotxf.c)
  ENDIF()
  IF(NOT NNPACK_INFERENCE_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # Tuple GEMM
      src/scalar/blas/s2gemm-transc.c
      src/scalar/blas/cgemm.c
      src/scalar/blas/cgemm-conjb-transc.c)
  ENDIF()
ELSEIF(NNPACK_BACKEND STREQUAL "neon")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/psimd/2d-fourier-8x8.c
    src/psimd/2d-fourier-16x16.c
    src/neon/2d-winograd-8x8-3x3.c
    src/neon/2d-winograd-8x8-3x3-fp16.c
    # Tuple GEMM
    src/neon/blas/h4gemm.c
    src/neon/blas/s4gemm.c
    src/neon/blas/c4gemm-conjb.c
    src/neon/blas/s4c2gemm-conjb.c
    # Direct convolution
    src/neon/blas/conv1x1.c
    # BLAS microkernels
    src/neon/blas/sgemm.c)
  IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l)$")
    LIST(APPEND NNPACK_BACKEND_SRCS
      # BLAS microkernels
      src/neon/blas/sgemm-aarch32.S)
  ENDIF()
  IF(NOT NNPACK_CONVOLUTION_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # ReLU
      src/neon/relu.c
      # Softmax
      src/psimd/softmax.c
      # BLAS microkernels
      src/neon/blas/sdotxf.c
      src/psimd/blas/shdotxf.c)
  ENDIF()
  IF(NOT NNPACK_INFERENCE_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # Tuple GEMM
      src/neon/blas/c4gemm.c
      src/neon/blas/s4c2gemm.c
      src/neon/blas/c4gemm-conjb-transc.c
      src/neon/blas/s4c2gemm-conjb-transc.c)
  ENDIF()
ELSEIF(NNPACK_BACKEND STREQUAL "psimd")
  SET(NNPACK_BACKEND_SRCS
    # Transformations
    src/psimd/2d-fourier-8x8.c
    src/psimd/2d-fourier-16x16.c
    src/psimd/2d-winograd-8x8-3x3.c
    # Tuple GEMM
    src/psimd/blas/s4gemm.c
    src/psimd/blas/c4gemm-conjb.c
    src/psimd/blas/s4c2gemm-conjb.c
    # Direct convolution
    src/psimd/blas/conv1x1.c
    # BLAS microkernels
    src/psimd/blas/sgemm.c)
  IF(NOT NNPACK_CONVOLUTION_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # ReLU
      src/psimd/relu.c
      # Softmax
      src/psimd/softmax.c
      # BLAS microkernels
      src/psimd/blas/sdotxf.c
      src/psimd/blas/shdotxf.c)
  ENDIF()
  IF(NOT NNPACK_INFERENCE_ONLY)
    LIST(APPEND NNPACK_BACKEND_SRCS
      # Tuple GEMM
      src/psimd/blas/c4gemm.c
      src/psimd/blas/s4c2gemm.c
      src/psimd/blas/c4gemm-conjb-transc.c
      src/psimd/blas/s4c2gemm-conjb-transc.c)
  ENDIF()
ENDIF()

FILE(GLOB_RECURSE PEACHPY_POTENTIAL_DEPS "src/*.py")

SET(NNPACK_BACKEND_PEACHPY_OBJS)
IF(NNPACK_BACKEND STREQUAL "x86-64")
  SET(NNPACK_BACKEND_C_SRCS)
  IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    SET(PEACHPY_IMAGE_FORMAT mach-o)
  ELSE()
    SET(PEACHPY_IMAGE_FORMAT elf)
  ENDIF()
  FOREACH(src ${NNPACK_BACKEND_SRCS})
    GET_FILENAME_COMPONENT(src_ext ${src} EXT)
    IF(src_ext STREQUAL ".py")
      SET(obj "${PROJECT_BINARY_DIR}/${src}${CMAKE_C_OUTPUT_EXTENSION}")
      GET_FILENAME_COMPONENT(obj_dir ${obj} DIRECTORY)
      FILE(MAKE_DIRECTORY ${obj_dir})
      ADD_CUSTOM_COMMAND(
        OUTPUT ${obj}
        COMMAND "PYTHONPATH=${PEACHPY_PYTHONPATH}"
          ${PYTHON_EXECUTABLE} -m peachpy.x86_64
            -mabi=sysv -g4 -mimage-format=${PEACHPY_IMAGE_FORMAT}
            "-I${PROJECT_SOURCE_DIR}/src" "-I${PROJECT_SOURCE_DIR}/src/x86_64-fma" "-I${FP16_SOURCE_DIR}/include"
            -o ${obj} "${PROJECT_SOURCE_DIR}/${src}"
        DEPENDS ${NNPACK_BACKEND_PEACHPY_OBJS})
      LIST(APPEND NNPACK_BACKEND_PEACHPY_OBJS ${obj})
    ELSE()
      LIST(APPEND NNPACK_BACKEND_C_SRCS ${src})
    ENDIF()
  ENDFOREACH(src)
ELSE()
  SET(NNPACK_BACKEND_C_SRCS ${NNPACK_BACKEND_SRCS})
ENDIF()

IF(NNPACK_LIBRARY_TYPE STREQUAL "default")
  ADD_LIBRARY(nnpack ${NNPACK_INIT_SRCS} ${NNPACK_LAYER_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSEIF(NNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(nnpack SHARED ${NNPACK_INIT_SRCS} ${NNPACK_LAYER_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSEIF(NNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(nnpack STATIC ${NNPACK_INIT_SRCS} ${NNPACK_LAYER_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported NNPACK library type \"${NNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
NNPACK_TARGET_ENABLE_C99(nnpack)
IF(IOS OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l)$")
  SET_PROPERTY(SOURCE ${NNPACK_BACKEND_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon-fp16 ")

  INCLUDE(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT)
  IF(COMPILER_SUPPORTS_FP16_FORMAT)
    SET_PROPERTY(SOURCE ${NNPACK_BACKEND_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfp16-format=ieee ")
  ENDIF()
ENDIF()
SET_PROPERTY(SOURCE ${NNPACK_INIT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Os ")
IF(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET_PROPERTY(SOURCE ${NNPACK_LAYER_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
ENDIF()
IF(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET_PROPERTY(SOURCE ${NNPACK_BACKEND_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O3 -ffast-math ")
ENDIF()
TARGET_INCLUDE_DIRECTORIES(nnpack PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE src)
IF(NNPACK_BACKEND STREQUAL "psimd")
  TARGET_COMPILE_DEFINITIONS(nnpack PRIVATE NNP_BACKEND_PSIMD=1)
ELSEIF(NNPACK_BACKEND STREQUAL "scalar")
  TARGET_COMPILE_DEFINITIONS(nnpack PRIVATE NNP_BACKEND_SCALAR=1)
ENDIF()
IF(NNPACK_CONVOLUTION_ONLY)
  TARGET_COMPILE_DEFINITIONS(nnpack PUBLIC NNP_CONVOLUTION_ONLY=1)
ELSE()
  TARGET_COMPILE_DEFINITIONS(nnpack PUBLIC NNP_CONVOLUTION_ONLY=0)
ENDIF()
IF(NNPACK_INFERENCE_ONLY)
  TARGET_COMPILE_DEFINITIONS(nnpack PUBLIC NNP_INFERENCE_ONLY=1)
ELSE()
  TARGET_COMPILE_DEFINITIONS(nnpack PUBLIC NNP_INFERENCE_ONLY=0)
ENDIF()
SET_TARGET_PROPERTIES(nnpack PROPERTIES PUBLIC_HEADER include/nnpack.h)

ADD_LIBRARY(nnpack_reference_layers STATIC ${NNPACK_REFERENCE_LAYERS_SRCS})
NNPACK_TARGET_ENABLE_C99(nnpack_reference_layers)
TARGET_INCLUDE_DIRECTORIES(nnpack_reference_layers PUBLIC include)

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${CPUINFO_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE cpuinfo)

# ---[ Configure pthreadpool
IF(NOT TARGET pthreadpool)
  SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
  SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${PTHREADPOOL_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
ENDIF()
IF(NNPACK_CUSTOM_THREADPOOL)
  # Depend on pthreadpool interface, but not on implementation.
  # This is used when NNPACK user (e.g. Caffe2) provides its own threadpool implementation.
  TARGET_LINK_LIBRARIES(nnpack PUBLIC pthreadpool_interface)
ELSE()
  TARGET_LINK_LIBRARIES(nnpack PUBLIC pthreadpool)
ENDIF()
TARGET_LINK_LIBRARIES(nnpack_reference_layers PUBLIC pthreadpool)

# ---[ Configure FXdiv
IF(NOT TARGET fxdiv)
  SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
  SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FXDIV_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE fxdiv)

# ---[ Configure psimd
IF(NOT TARGET psimd)
  ADD_SUBDIRECTORY(
    "${PSIMD_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE psimd)

# ---[ Configure FP16
IF(NOT TARGET fp16)
  SET(FP16_BUILD_TESTS OFF CACHE BOOL "")
  SET(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FP16_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE fp16)
TARGET_LINK_LIBRARIES(nnpack_reference_layers PUBLIC fp16)

INSTALL(TARGETS nnpack
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

IF(NNPACK_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  ENDIF()

  ADD_EXECUTABLE(convolution-inference-smoketest test/convolution-inference/smoke.cc)
  NNPACK_TARGET_ENABLE_CXX11(convolution-inference-smoketest)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-smoketest convolution-inference-smoketest)

  ADD_EXECUTABLE(convolution-inference-alexnet-test test/convolution-inference/alexnet.cc)
  NNPACK_TARGET_ENABLE_CXX11(convolution-inference-alexnet-test)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-alexnet convolution-inference-alexnet-test)

  ADD_EXECUTABLE(convolution-inference-overfeat-test test/convolution-inference/overfeat-fast.cc)
  NNPACK_TARGET_ENABLE_CXX11(convolution-inference-overfeat-test)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-overfeat convolution-inference-overfeat-test)

  ADD_EXECUTABLE(convolution-inference-vgg-test test/convolution-inference/vgg-a.cc)
  NNPACK_TARGET_ENABLE_CXX11(convolution-inference-vgg-test)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-vgg convolution-inference-vgg-test)

  IF(NOT NNPACK_INFERENCE_ONLY)
    ADD_EXECUTABLE(convolution-output-smoketest test/convolution-output/smoke.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-output-smoketest)
    TARGET_INCLUDE_DIRECTORIES(convolution-output-smoketest PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-output-smoketest convolution-output-smoketest)

    ADD_EXECUTABLE(convolution-output-alexnet-test test/convolution-output/alexnet.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-output-alexnet-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-output-alexnet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-output-alexnet convolution-output-alexnet-test)

    ADD_EXECUTABLE(convolution-output-overfeat-test test/convolution-output/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-output-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-output-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-output-overfeat convolution-output-overfeat-test)

    ADD_EXECUTABLE(convolution-output-vgg-test test/convolution-output/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-output-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-output-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-output-vgg convolution-output-vgg-test)

    ADD_EXECUTABLE(convolution-input-gradient-smoketest test/convolution-input-gradient/smoke.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-input-gradient-smoketest)
    TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-smoketest PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-input-gradient-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-input-gradient-smoketest convolution-input-gradient-smoketest)

    ADD_EXECUTABLE(convolution-input-gradient-alexnet-test test/convolution-input-gradient/alexnet.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-input-gradient-alexnet-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-alexnet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-input-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-input-gradient-alexnet convolution-input-gradient-alexnet-test)

    ADD_EXECUTABLE(convolution-input-gradient-overfeat-test test/convolution-input-gradient/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-input-gradient-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-input-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-input-gradient-overfeat convolution-input-gradient-overfeat-test)

    ADD_EXECUTABLE(convolution-input-gradient-vgg-test test/convolution-input-gradient/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-input-gradient-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-input-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-input-gradient-vgg convolution-input-gradient-vgg-test)

    ADD_EXECUTABLE(convolution-kernel-gradient-smoketest test/convolution-kernel-gradient/smoke.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-kernel-gradient-smoketest)
    TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-smoketest PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-kernel-gradient-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-kernel-gradient-smoketest convolution-kernel-gradient-smoketest)

    ADD_EXECUTABLE(convolution-kernel-gradient-alexnet-test test/convolution-kernel-gradient/alexnet.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-kernel-gradient-alexnet-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-alexnet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-kernel-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-kernel-gradient-alexnet convolution-kernel-gradient-alexnet-test)

    ADD_EXECUTABLE(convolution-kernel-gradient-overfeat-test test/convolution-kernel-gradient/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-kernel-gradient-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-kernel-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-kernel-gradient-overfeat convolution-kernel-gradient-overfeat-test)

    ADD_EXECUTABLE(convolution-kernel-gradient-vgg-test test/convolution-kernel-gradient/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(convolution-kernel-gradient-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(convolution-kernel-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(convolution-kernel-gradient-vgg convolution-kernel-gradient-vgg-test)
  ENDIF()

  IF(NOT NNPACK_CONVOLUTION_ONLY)
    ADD_EXECUTABLE(fully-connected-inference-alexnet-test test/fully-connected-inference/alexnet.cc)
    NNPACK_TARGET_ENABLE_CXX11(fully-connected-inference-alexnet-test)
    TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-alexnet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(fully-connected-inference-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(fully-connected-inference-alexnet fully-connected-inference-alexnet-test)

    ADD_EXECUTABLE(fully-connected-inference-overfeat-test test/fully-connected-inference/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(fully-connected-inference-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(fully-connected-inference-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(fully-connected-inference-overfeat fully-connected-inference-overfeat-test)

    ADD_EXECUTABLE(fully-connected-inference-vgg-test test/fully-connected-inference/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(fully-connected-inference-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(fully-connected-inference-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(fully-connected-inference-vgg fully-connected-inference-vgg-test)

    IF(NOT NNPACK_INFERENCE_ONLY)
      ADD_EXECUTABLE(fully-connected-output-smoketest test/fully-connected-output/smoke.cc)
      NNPACK_TARGET_ENABLE_CXX11(fully-connected-output-smoketest)
      TARGET_INCLUDE_DIRECTORIES(fully-connected-output-smoketest PRIVATE test)
      TARGET_LINK_LIBRARIES(fully-connected-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(fully-connected-output-smoketest fully-connected-output-smoketest)

      ADD_EXECUTABLE(fully-connected-output-alexnet-test test/fully-connected-output/alexnet.cc)
      NNPACK_TARGET_ENABLE_CXX11(fully-connected-output-alexnet-test)
      TARGET_INCLUDE_DIRECTORIES(fully-connected-output-alexnet-test PRIVATE test)
      TARGET_LINK_LIBRARIES(fully-connected-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(fully-connected-output-alexnet fully-connected-output-alexnet-test)

      ADD_EXECUTABLE(fully-connected-output-overfeat-test test/fully-connected-output/overfeat-fast.cc)
      NNPACK_TARGET_ENABLE_CXX11(fully-connected-output-overfeat-test)
      TARGET_INCLUDE_DIRECTORIES(fully-connected-output-overfeat-test PRIVATE test)
      TARGET_LINK_LIBRARIES(fully-connected-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(fully-connected-output-overfeat fully-connected-output-overfeat-test)

      ADD_EXECUTABLE(fully-connected-output-vgg-test test/fully-connected-output/vgg-a.cc)
      NNPACK_TARGET_ENABLE_CXX11(fully-connected-output-vgg-test)
      TARGET_INCLUDE_DIRECTORIES(fully-connected-output-vgg-test PRIVATE test)
      TARGET_LINK_LIBRARIES(fully-connected-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(fully-connected-output-vgg fully-connected-output-vgg-test)
    ENDIF()

    ADD_EXECUTABLE(max-pooling-output-smoketest test/max-pooling-output/smoke.cc)
    NNPACK_TARGET_ENABLE_CXX11(max-pooling-output-smoketest)
    TARGET_INCLUDE_DIRECTORIES(max-pooling-output-smoketest PRIVATE test)
    TARGET_LINK_LIBRARIES(max-pooling-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(max-pooling-output-smoketest max-pooling-output-smoketest)

    ADD_EXECUTABLE(max-pooling-output-overfeat-test test/max-pooling-output/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(max-pooling-output-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(max-pooling-output-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(max-pooling-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(max-pooling-output-overfeat max-pooling-output-overfeat-test)

    ADD_EXECUTABLE(max-pooling-output-vgg-test test/max-pooling-output/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(max-pooling-output-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(max-pooling-output-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(max-pooling-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(max-pooling-output-vgg max-pooling-output-vgg-test)

    ADD_EXECUTABLE(relu-output-alexnet-test test/relu-output/alexnet.cc)
    NNPACK_TARGET_ENABLE_CXX11(relu-output-alexnet-test)
    TARGET_INCLUDE_DIRECTORIES(relu-output-alexnet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(relu-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(relu-output-alexnet relu-output-alexnet-test)

    ADD_EXECUTABLE(relu-output-overfeat-test test/relu-output/overfeat-fast.cc)
    NNPACK_TARGET_ENABLE_CXX11(relu-output-overfeat-test)
    TARGET_INCLUDE_DIRECTORIES(relu-output-overfeat-test PRIVATE test)
    TARGET_LINK_LIBRARIES(relu-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(relu-output-overfeat relu-output-overfeat-test)

    ADD_EXECUTABLE(relu-output-vgg-test test/relu-output/vgg-a.cc)
    NNPACK_TARGET_ENABLE_CXX11(relu-output-vgg-test)
    TARGET_INCLUDE_DIRECTORIES(relu-output-vgg-test PRIVATE test)
    TARGET_LINK_LIBRARIES(relu-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(relu-output-vgg relu-output-vgg-test)

    IF(NOT NNPACK_INFERENCE_ONLY)
      ADD_EXECUTABLE(relu-input-gradient-alexnet-test test/relu-input-gradient/alexnet.cc)
      NNPACK_TARGET_ENABLE_CXX11(relu-input-gradient-alexnet-test)
      TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-alexnet-test PRIVATE test)
      TARGET_LINK_LIBRARIES(relu-input-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(relu-input-gradient-alexnet relu-input-gradient-alexnet-test)

      ADD_EXECUTABLE(relu-input-gradient-overfeat-test test/relu-input-gradient/overfeat-fast.cc)
      NNPACK_TARGET_ENABLE_CXX11(relu-input-gradient-overfeat-test)
      TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-overfeat-test PRIVATE test)
      TARGET_LINK_LIBRARIES(relu-input-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(relu-input-gradient-overfeat relu-input-gradient-overfeat-test)

      ADD_EXECUTABLE(relu-input-gradient-vgg-test test/relu-input-gradient/vgg-a.cc)
      NNPACK_TARGET_ENABLE_CXX11(relu-input-gradient-vgg-test)
      TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-vgg-test PRIVATE test)
      TARGET_LINK_LIBRARIES(relu-input-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
      ADD_TEST(relu-input-gradient-vgg relu-input-gradient-vgg-test)
    ENDIF()

    ADD_EXECUTABLE(softmax-output-smoketest test/softmax-output/smoke.cc)
    NNPACK_TARGET_ENABLE_CXX11(softmax-output-smoketest)
    TARGET_INCLUDE_DIRECTORIES(softmax-output-smoketest PRIVATE test)
    TARGET_LINK_LIBRARIES(softmax-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(softmax-output-smoketest softmax-output-smoketest)

    ADD_EXECUTABLE(softmax-output-imagenet-test test/softmax-output/imagenet.cc)
    NNPACK_TARGET_ENABLE_CXX11(softmax-output-imagenet-test)
    TARGET_INCLUDE_DIRECTORIES(softmax-output-imagenet-test PRIVATE test)
    TARGET_LINK_LIBRARIES(softmax-output-imagenet-test PRIVATE nnpack nnpack_reference_layers gtest)
    ADD_TEST(softmax-output-imagenet softmax-output-imagenet-test)
  ENDIF()
ENDIF()
