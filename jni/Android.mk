LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := pthreadpool
LOCAL_SRC_FILES := $(LOCAL_PATH)/deps/pthreadpool/src/threadpool-pthreads.c
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/deps/pthreadpool/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES) $(LOCAL_PATH)/deps/fxdiv/include
LOCAL_CFLAGS := -std=gnu99
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gtest
LOCAL_SRC_FILES := $(LOCAL_PATH)/deps/googletest/googletest/src/gtest-all.cc
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/deps/googletest/googletest/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES) $(LOCAL_PATH)/deps/googletest/googletest
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnpack_ukernels
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64 armeabi-v7a arm64-v8a))
LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/src/psimd/2d-fourier-8x8.c \
	$(LOCAL_PATH)/src/psimd/2d-fourier-16x16.c \
	$(LOCAL_PATH)/src/psimd/softmax.c \
	$(LOCAL_PATH)/src/psimd/blas/shdotxf.c
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
LOCAL_SRC_FILES += \
	$(LOCAL_PATH)/src/neon/relu.c \
	$(LOCAL_PATH)/src/neon/2d-winograd-8x8-3x3.c \
	$(LOCAL_PATH)/src/neon/2d-winograd-8x8-3x3-fp16.c \
	$(LOCAL_PATH)/src/neon/blas/conv1x1.c \
	$(LOCAL_PATH)/src/neon/blas/h4gemm.c \
	$(LOCAL_PATH)/src/neon/blas/s4gemm.c \
	$(LOCAL_PATH)/src/neon/blas/c4gemm.c \
	$(LOCAL_PATH)/src/neon/blas/s4c2gemm.c \
	$(LOCAL_PATH)/src/neon/blas/c4gemm-conjb.c \
	$(LOCAL_PATH)/src/neon/blas/s4c2gemm-conjb.c \
	$(LOCAL_PATH)/src/neon/blas/c4gemm-conjb-transc.c \
	$(LOCAL_PATH)/src/neon/blas/s4c2gemm-conjb-transc.c \
	$(LOCAL_PATH)/src/neon/blas/sgemm.c \
	$(LOCAL_PATH)/src/neon/blas/sdotxf.c
else
LOCAL_SRC_FILES += \
	$(LOCAL_PATH)/src/psimd/relu.c \
	$(LOCAL_PATH)/src/psimd/2d-winograd-8x8-3x3.c \
	$(LOCAL_PATH)/src/psimd/blas/conv1x1.c \
	$(LOCAL_PATH)/src/psimd/blas/s4gemm.c \
	$(LOCAL_PATH)/src/psimd/blas/c4gemm.c \
	$(LOCAL_PATH)/src/psimd/blas/s4c2gemm.c \
	$(LOCAL_PATH)/src/psimd/blas/c4gemm-conjb.c \
	$(LOCAL_PATH)/src/psimd/blas/s4c2gemm-conjb.c \
	$(LOCAL_PATH)/src/psimd/blas/c4gemm-conjb-transc.c \
	$(LOCAL_PATH)/src/psimd/blas/s4c2gemm-conjb-transc.c \
	$(LOCAL_PATH)/src/psimd/blas/sgemm.c \
	$(LOCAL_PATH)/src/psimd/blas/sdotxf.c
endif
else
LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/src/scalar/2d-fourier-8x8.c \
	$(LOCAL_PATH)/src/scalar/2d-fourier-16x16.c \
	$(LOCAL_PATH)/src/scalar/2d-winograd-8x8-3x3.c \
	$(LOCAL_PATH)/src/scalar/relu.c \
	$(LOCAL_PATH)/src/scalar/softmax.c \
	$(LOCAL_PATH)/src/scalar/blas/shdotxf.c \
	$(LOCAL_PATH)/src/scalar/blas/conv1x1.c \
	$(LOCAL_PATH)/src/scalar/blas/s2gemm.c \
	$(LOCAL_PATH)/src/scalar/blas/cgemm.c \
	$(LOCAL_PATH)/src/scalar/blas/cgemm-conjb.c \
	$(LOCAL_PATH)/src/scalar/blas/s2gemm-transc.c \
	$(LOCAL_PATH)/src/scalar/blas/cgemm-conjb-transc.c \
	$(LOCAL_PATH)/src/scalar/blas/sgemm.c \
	$(LOCAL_PATH)/src/scalar/blas/sdotxf.c
endif
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include $(LOCAL_PATH)/src $(LOCAL_PATH)/deps/fp16/include $(LOCAL_PATH)/deps/psimd/include
LOCAL_CFLAGS := -std=gnu99 -D__STDC_CONSTANT_MACROS=1
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_CFLAGS += -mfpu=neon-fp16
endif # TARGET_ARCH_ABI == armeabi-v7a
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnpack_test_ukernels
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include $(LOCAL_PATH)/src
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64 armeabi-v7a arm64-v8a))
LOCAL_C_INCLUDES += $(LOCAL_PATH)/deps/psimd/include
LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/src/psimd/fft-aos.c \
	$(LOCAL_PATH)/src/psimd/fft-soa.c \
	$(LOCAL_PATH)/src/psimd/fft-real.c \
	$(LOCAL_PATH)/src/psimd/fft-dualreal.c
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
LOCAL_SRC_FILES += $(LOCAL_PATH)/src/neon/winograd-f6k3.c
else
LOCAL_SRC_FILES += $(LOCAL_PATH)/src/psimd/winograd-f6k3.c
endif # TARGET_ARCH_ABI is armeabi-v7a arm64-v8a
else  # TARGET_ARCH_ABI is x86, x86-64, armeabi-v7a, or arm64-v8a
LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/src/scalar/fft-aos.c \
	$(LOCAL_PATH)/src/scalar/fft-soa.c \
	$(LOCAL_PATH)/src/scalar/fft-real.c \
	$(LOCAL_PATH)/src/scalar/fft-dualreal.c \
	$(LOCAL_PATH)/src/scalar/winograd-f6k3.c
endif # TARGET_ARCH_ABI is x86, x86-64, armeabi-v7a, or arm64-v8a
LOCAL_CFLAGS := -std=gnu99 -D__STDC_CONSTANT_MACROS=1
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -mfpu=neon-fp16
endif # TARGET_ARCH_ABI == armeabi-v7a
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnpack
LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/src/init.c \
	$(LOCAL_PATH)/src/convolution-output.c \
	$(LOCAL_PATH)/src/convolution-input-gradient.c \
	$(LOCAL_PATH)/src/convolution-kernel-gradient.c \
	$(LOCAL_PATH)/src/convolution-inference.c \
	$(LOCAL_PATH)/src/fully-connected-output.c \
	$(LOCAL_PATH)/src/fully-connected-inference.c \
	$(LOCAL_PATH)/src/pooling-output.c \
	$(LOCAL_PATH)/src/softmax-output.c \
	$(LOCAL_PATH)/src/relu-output.c \
	$(LOCAL_PATH)/src/relu-input-gradient.c
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES) $(LOCAL_PATH)/deps/fxdiv/include $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=gnu99
ifneq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
endif
LOCAL_STATIC_LIBRARIES := nnpack_ukernels pthreadpool
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_STATIC_LIBRARIES += cpufeatures
endif # TARGET_ARCH_ABI == armeabi-v7a
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnpack_reference
LOCAL_SRC_FILES := \
    $(LOCAL_PATH)/src/ref/convolution-output.c \
	$(LOCAL_PATH)/src/ref/convolution-input-gradient.c \
	$(LOCAL_PATH)/src/ref/convolution-kernel.c \
	$(LOCAL_PATH)/src/ref/fully-connected-output.c \
	$(LOCAL_PATH)/src/ref/max-pooling-output.c \
	$(LOCAL_PATH)/src/ref/softmax-output.c \
	$(LOCAL_PATH)/src/ref/relu-output.c \
	$(LOCAL_PATH)/src/ref/relu-input-gradient.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -std=gnu99 -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := pthreadpool
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := fourier_reference
LOCAL_SRC_FILES := \
    $(LOCAL_PATH)/src/ref/fft/aos.c \
	$(LOCAL_PATH)/src/ref/fft/soa.c \
	$(LOCAL_PATH)/src/ref/fft/forward-real.c \
	$(LOCAL_PATH)/src/ref/fft/forward-dualreal.c \
	$(LOCAL_PATH)/src/ref/fft/inverse-real.c \
	$(LOCAL_PATH)/src/ref/fft/inverse-dualreal.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include $(LOCAL_PATH)/src $(LOCAL_PATH)/src/ref
LOCAL_CFLAGS := -std=gnu99 -D__STDC_CONSTANT_MACROS=1
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := bench_utils
LOCAL_SRC_FILES := $(LOCAL_PATH)/bench/median.c $(LOCAL_PATH)/bench/memread.c
LOCAL_STATIC_LIBRARIES := nnpack
LOCAL_CFLAGS := -std=gnu99
include $(BUILD_STATIC_LIBRARY)

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64 armeabi-v7a arm64-v8a))
include $(CLEAR_VARS)
LOCAL_MODULE := fourier-psimd-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fourier/psimd.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_test_ukernels fourier_reference gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is x86, x86_64, armeabi-v7a, or arm64-v8a

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),mips mips64))
include $(CLEAR_VARS)
LOCAL_MODULE := fourier-scalar-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fourier/scalar.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_test_ukernels fourier_reference gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is mips or mips64

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
include $(CLEAR_VARS)
LOCAL_MODULE := winograd-neon-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/winograd/neon.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_test_ukernels gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is armeabi-v7a or arm64-v8a

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64))
include $(CLEAR_VARS)
LOCAL_MODULE := winograd-psimd-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/winograd/psimd.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_test_ukernels gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is x86 or x86_64

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),mips mips64))
include $(CLEAR_VARS)
LOCAL_MODULE := winograd-scalar-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/winograd/scalar.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_test_ukernels gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is mips or mips64

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi-v7a arm64-v8a))
include $(CLEAR_VARS)
LOCAL_MODULE := sgemm-neon-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/sgemm/neon.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is armeabi-v7a or arm64-v8a

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64))
include $(CLEAR_VARS)
LOCAL_MODULE := sgemm-psimd-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/sgemm/psimd.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is x86 or x86_64

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),mips mips64))
include $(CLEAR_VARS)
LOCAL_MODULE := sgemm-scalar-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/sgemm/scalar.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack gtest
include $(BUILD_EXECUTABLE)
endif # TARGET_ARCH_ABI is mips or mips64

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-output-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-output/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-output-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-output/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-output-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-output/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-output-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-output/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-input-gradient-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-input-gradient/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-input-gradient-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-input-gradient/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-input-gradient-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-input-gradient/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-input-gradient-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-input-gradient/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-kernel-gradient-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-kernel-gradient/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-kernel-gradient-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-kernel-gradient/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-kernel-gradient-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-kernel-gradient/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-kernel-gradient-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-kernel-gradient/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-inference-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-inference/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-inference-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-inference/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-inference-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-inference/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-inference-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/convolution-inference/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-output-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-output/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-output-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-output/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-output-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-output/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-output-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-output/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-inference-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-inference/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-inference-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-inference/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-inference-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/fully-connected-inference/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := max-pooling-output-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/max-pooling-output/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := max-pooling-output-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/max-pooling-output/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := max-pooling-output-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/max-pooling-output/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-output-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-output/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-output-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-output/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-output-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-output/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-input-gradient-alexnet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-input-gradient/alexnet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-input-gradient-vgg-a-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-input-gradient/vgg-a.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-input-gradient-overfeat-fast-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/relu-input-gradient/overfeat-fast.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test $(LOCAL_PATH)/deps/fp16/include
LOCAL_CFLAGS := -D__STDC_CONSTANT_MACROS=1
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := softmax-output-smoketest
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/softmax-output/smoke.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := softmax-output-imagenet-test
LOCAL_SRC_FILES := $(LOCAL_PATH)/test/softmax-output/imagenet.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/test
LOCAL_STATIC_LIBRARIES := nnpack nnpack_reference gtest
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := convolution-benchmark
LOCAL_SRC_FILES := $(LOCAL_PATH)/bench/convolution.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/bench
LOCAL_STATIC_LIBRARIES := nnpack bench_utils
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := fully-connected-benchmark
LOCAL_SRC_FILES := $(LOCAL_PATH)/bench/fully-connected.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/bench
LOCAL_STATIC_LIBRARIES := nnpack bench_utils
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := relu-benchmark
LOCAL_SRC_FILES := $(LOCAL_PATH)/bench/relu.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/bench
LOCAL_STATIC_LIBRARIES := nnpack bench_utils
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := pooling-benchmark
LOCAL_SRC_FILES := $(LOCAL_PATH)/bench/pooling.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/bench
LOCAL_STATIC_LIBRARIES := nnpack bench_utils
include $(BUILD_EXECUTABLE)

$(call import-module,android/cpufeatures)
