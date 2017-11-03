CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(NNPACK C CXX)

# ---[ Options.
SET(NNPACK_BACKEND "auto" CACHE STRING "Backend for micro-kernels implementation")
SET_PROPERTY(CACHE NNPACK_BACKEND PROPERTY STRINGS auto psimd scalar)
OPTION(NNPACK_CUSTOM_THREADPOOL "Build NNPACK for custom thread pool" OFF)
SET(NNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE NNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(NNPACK_BUILD_TESTS "Build NNPACK unit tests" ON)

# ---[ CMake options
IF(NNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Build flags
IF(NOT MSVC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
ENDIF()

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
  CONFIGURE_FILE(cmake/DownloadSix.cmake "${CMAKE_BINARY_DIR}/six-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/six-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/six-download")

  IF(${PYTHON_VERSION_STRING} VERSION_LESS 3.4)
    # ---[ Python < 3.4 does not natively support enums, and needs a polyfill
    CONFIGURE_FILE(cmake/DownloadEnum.cmake "${CMAKE_BINARY_DIR}/enum-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/enum-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/enum-download")
  ENDIF()

  CONFIGURE_FILE(cmake/DownloadOpcodes.cmake "${CMAKE_BINARY_DIR}/opcodes-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/opcodes-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/opcodes-download")

  CONFIGURE_FILE(cmake/DownloadPeachPy.cmake "${CMAKE_BINARY_DIR}/peachpy-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/peachpy-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/peachpy-download")
ENDIF()

IF(NOT TARGET fp16)
  CONFIGURE_FILE(cmake/DownloadFP16.cmake "${CMAKE_BINARY_DIR}/fp16-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/fp16-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/fp16-download")
ENDIF()

IF(NOT TARGET fxdiv)
  CONFIGURE_FILE(cmake/DownloadFXdiv.cmake "${CMAKE_BINARY_DIR}/fxdiv-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/fxdiv-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/fxdiv-download")
ENDIF()

IF(NOT TARGET psimd)
  CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CMAKE_BINARY_DIR}/psimd-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
ENDIF()

IF(NOT TARGET pthreadpool)
  CONFIGURE_FILE(cmake/DownloadPThreadPool.cmake "${CMAKE_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
ENDIF()

IF(NOT TARGET gtest)
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
ENDIF()

# ---[ NNPACK library
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
ELSEIF(NNPACK_BACKEND STREQUAL "scalar")
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
ELSEIF(NNPACK_BACKEND STREQUAL "neon")
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
ELSEIF(NNPACK_BACKEND STREQUAL "psimd")
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
        COMMAND PYTHONPATH=${CONFU_DEPENDENCIES_SOURCE_DIR}/six:${CONFU_DEPENDENCIES_SOURCE_DIR}/enum:${CONFU_DEPENDENCIES_SOURCE_DIR}/peachpy
          ${PYTHON_EXECUTABLE} -m peachpy.x86_64
            -mabi=sysv -g4 -mimage-format=${PEACHPY_IMAGE_FORMAT}
            "-I${PROJECT_SOURCE_DIR}/src" "-I${PROJECT_SOURCE_DIR}/src/x86_64-fma" "-I${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16/include"
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
  ADD_LIBRARY(nnpack ${NNPACK_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSEIF(NNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(nnpack SHARED ${NNPACK_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSEIF(NNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(nnpack STATIC ${NNPACK_SRCS} ${NNPACK_BACKEND_C_SRCS} ${NNPACK_BACKEND_PEACHPY_OBJS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported NNPACK library type \"${NNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
IF(IOS OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv5te|armv7-a|armv7l)$")
  SET_PROPERTY(SOURCE ${NNPACK_BACKEND_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon-fp16 ")

  INCLUDE(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT)
  IF(COMPILER_SUPPORTS_FP16_FORMAT)
    SET_PROPERTY(SOURCE ${NNPACK_BACKEND_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfp16-format=ieee ")
  ENDIF()
ENDIF()
TARGET_INCLUDE_DIRECTORIES(nnpack PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(nnpack PRIVATE src)
IF(NNPACK_BACKEND STREQUAL "psimd")
  TARGET_COMPILE_DEFINITIONS(nnpack PRIVATE "-DNNP_BACKEND_PSIMD=1")
ELSEIF(NNPACK_BACKEND STREQUAL "scalar")
  TARGET_COMPILE_DEFINITIONS(nnpack PRIVATE "-DNNP_BACKEND_SCALAR=1")
ENDIF()
SET_TARGET_PROPERTIES(nnpack PROPERTIES PUBLIC_HEADER include/nnpack.h)

ADD_LIBRARY(nnpack_reference_layers ${NNPACK_REFERENCE_LAYERS_SRCS})
TARGET_INCLUDE_DIRECTORIES(nnpack_reference_layers PUBLIC include)

# ---[ Configure pthreadpool
IF(NOT TARGET pthreadpool)
  SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
  SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool"
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
    "${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE fxdiv)

# ---[ Configure psimd
IF(NOT TARGET psimd)
  ADD_SUBDIRECTORY(
    "${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE psimd)

# ---[ Configure FP16
IF(NOT TARGET fp16)
  SET(FP16_BUILD_TESTS OFF CACHE BOOL "")
  SET(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16")
ENDIF()
TARGET_LINK_LIBRARIES(nnpack PRIVATE fp16)
TARGET_LINK_LIBRARIES(nnpack_reference_layers PUBLIC fp16)

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

IF(NNPACK_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  ENDIF()

  ADD_EXECUTABLE(convolution-output-smoketest test/convolution-output/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-output-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-output-smoketest convolution-output-smoketest)

  ADD_EXECUTABLE(convolution-output-alexnet-test test/convolution-output/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-output-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-output-alexnet convolution-output-alexnet-test)

  ADD_EXECUTABLE(convolution-output-overfeat-test test/convolution-output/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-output-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-output-overfeat convolution-output-overfeat-test)

  ADD_EXECUTABLE(convolution-output-vgg-test test/convolution-output/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-output-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-output-vgg convolution-output-vgg-test)

  ADD_EXECUTABLE(convolution-input-gradient-smoketest test/convolution-input-gradient/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-input-gradient-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-input-gradient-smoketest convolution-input-gradient-smoketest)

  ADD_EXECUTABLE(convolution-input-gradient-alexnet-test test/convolution-input-gradient/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-input-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-input-gradient-alexnet convolution-input-gradient-alexnet-test)

  ADD_EXECUTABLE(convolution-input-gradient-overfeat-test test/convolution-input-gradient/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-input-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-input-gradient-overfeat convolution-input-gradient-overfeat-test)

  ADD_EXECUTABLE(convolution-input-gradient-vgg-test test/convolution-input-gradient/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-input-gradient-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-input-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-input-gradient-vgg convolution-input-gradient-vgg-test)

  ADD_EXECUTABLE(convolution-kernel-gradient-smoketest test/convolution-kernel-gradient/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-kernel-gradient-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-kernel-gradient-smoketest convolution-kernel-gradient-smoketest)

  ADD_EXECUTABLE(convolution-kernel-gradient-alexnet-test test/convolution-kernel-gradient/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-kernel-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-kernel-gradient-alexnet convolution-kernel-gradient-alexnet-test)

  ADD_EXECUTABLE(convolution-kernel-gradient-overfeat-test test/convolution-kernel-gradient/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-kernel-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-kernel-gradient-overfeat convolution-kernel-gradient-overfeat-test)

  ADD_EXECUTABLE(convolution-kernel-gradient-vgg-test test/convolution-kernel-gradient/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-kernel-gradient-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-kernel-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-kernel-gradient-vgg convolution-kernel-gradient-vgg-test)

  ADD_EXECUTABLE(convolution-inference-smoketest test/convolution-inference/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-smoketest convolution-inference-smoketest)

  ADD_EXECUTABLE(convolution-inference-alexnet-test test/convolution-inference/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-alexnet convolution-inference-alexnet-test)

  ADD_EXECUTABLE(convolution-inference-overfeat-test test/convolution-inference/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-overfeat convolution-inference-overfeat-test)

  ADD_EXECUTABLE(convolution-inference-vgg-test test/convolution-inference/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(convolution-inference-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(convolution-inference-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(convolution-inference-vgg convolution-inference-vgg-test)

  ADD_EXECUTABLE(fully-connected-output-smoketest test/fully-connected-output/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-output-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-output-smoketest fully-connected-output-smoketest)

  ADD_EXECUTABLE(fully-connected-output-alexnet-test test/fully-connected-output/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-output-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-output-alexnet fully-connected-output-alexnet-test)

  ADD_EXECUTABLE(fully-connected-output-overfeat-test test/fully-connected-output/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-output-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-output-overfeat fully-connected-output-overfeat-test)

  ADD_EXECUTABLE(fully-connected-output-vgg-test test/fully-connected-output/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-output-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-output-vgg fully-connected-output-vgg-test)

  ADD_EXECUTABLE(fully-connected-inference-alexnet-test test/fully-connected-inference/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-inference-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-inference-alexnet fully-connected-inference-alexnet-test)

  ADD_EXECUTABLE(fully-connected-inference-overfeat-test test/fully-connected-inference/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-inference-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-inference-overfeat fully-connected-inference-overfeat-test)

  ADD_EXECUTABLE(fully-connected-inference-vgg-test test/fully-connected-inference/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-inference-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(fully-connected-inference-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(fully-connected-inference-vgg fully-connected-inference-vgg-test)

  ADD_EXECUTABLE(max-pooling-output-smoketest test/max-pooling-output/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-output-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(max-pooling-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(max-pooling-output-smoketest max-pooling-output-smoketest)

  ADD_EXECUTABLE(max-pooling-output-overfeat-test test/max-pooling-output/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-output-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(max-pooling-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(max-pooling-output-overfeat max-pooling-output-overfeat-test)

  ADD_EXECUTABLE(max-pooling-output-vgg-test test/max-pooling-output/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-output-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(max-pooling-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(max-pooling-output-vgg max-pooling-output-vgg-test)

  ADD_EXECUTABLE(relu-output-alexnet-test test/relu-output/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-output-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-output-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-output-alexnet relu-output-alexnet-test)

  ADD_EXECUTABLE(relu-output-overfeat-test test/relu-output/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-output-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-output-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-output-overfeat relu-output-overfeat-test)

  ADD_EXECUTABLE(relu-output-vgg-test test/relu-output/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-output-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-output-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-output-vgg relu-output-vgg-test)

  ADD_EXECUTABLE(relu-input-gradient-alexnet-test test/relu-input-gradient/alexnet.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-alexnet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-input-gradient-alexnet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-input-gradient-alexnet relu-input-gradient-alexnet-test)

  ADD_EXECUTABLE(relu-input-gradient-overfeat-test test/relu-input-gradient/overfeat-fast.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-overfeat-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-input-gradient-overfeat-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-input-gradient-overfeat relu-input-gradient-overfeat-test)

  ADD_EXECUTABLE(relu-input-gradient-vgg-test test/relu-input-gradient/vgg-a.cc)
  TARGET_INCLUDE_DIRECTORIES(relu-input-gradient-vgg-test PRIVATE test)
  TARGET_LINK_LIBRARIES(relu-input-gradient-vgg-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(relu-input-gradient-vgg relu-input-gradient-vgg-test)

  ADD_EXECUTABLE(softmax-output-smoketest test/softmax-output/smoke.cc)
  TARGET_INCLUDE_DIRECTORIES(softmax-output-smoketest PRIVATE test)
  TARGET_LINK_LIBRARIES(softmax-output-smoketest PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(softmax-output-smoketest softmax-output-smoketest)

  ADD_EXECUTABLE(softmax-output-imagenet-test test/softmax-output/imagenet.cc)
  TARGET_INCLUDE_DIRECTORIES(softmax-output-imagenet-test PRIVATE test)
  TARGET_LINK_LIBRARIES(softmax-output-imagenet-test PRIVATE nnpack nnpack_reference_layers gtest)
  ADD_TEST(softmax-output-imagenet softmax-output-imagenet-test)
ENDIF()
