#!/usr/bin/env python


import confu
parser = confu.standard_parser()
parser.add_argument("--backend", dest="backend", default="auto",
                    choices=["auto", "psimd", "scalar"])


def main(args):
    options = parser.parse_args(args)

    backend = options.backend
    if backend == "auto":
        if options.target.is_x86_64:
            backend = "x86_64"
        elif options.target.is_emscripten:
            backend = "scalar"
        else:
            backend = "psimd"
    if backend == "psimd":
        options.toolchain = "clang"

    build = confu.Build.from_options(options)

    macro = None
    if backend == "psimd":
        macro = "NNP_ARCH_PSIMD"
    if backend == "scalar":
        macro = "NNP_ARCH_SCALAR"

    build.export_cpath("include", ["nnpack.h"])

    with build.options(source_dir="src", macros=macro,
            deps={
                (build.deps.pthreadpool, build.deps.fxdiv, build.deps.fp16): any,
                build.deps.psimd: backend == "psimd",
            },
            extra_include_dirs={
                ("src", "src/ref"): any,
                "src/x86_64-fma": options.target.is_x86_64
            }):

        nnpack_objects = [
            build.cc("init.c"),
            build.cc("convolution-output.c"),
            build.cc("convolution-input-gradient.c"),
            build.cc("convolution-kernel.c"),
            build.cc("convolution-inference.c"),
            build.cc("fully-connected-output.c"),
            build.cc("fully-connected-inference.c"),
            build.cc("pooling-output.c"),
            build.cc("softmax-output.c"),
            build.cc("relu-output.c"),
            build.cc("relu-input-gradient.c"),
        ]

        if backend == "x86_64":
            arch_nnpack_objects = [
                # Transformations
                build.peachpy("x86_64-fma/2d-fourier-8x8.py"),
                build.peachpy("x86_64-fma/2d-fourier-16x16.py"),
                build.peachpy("x86_64-fma/2d-winograd-8x8-3x3.py"),
                # Pooling
                build.peachpy("x86_64-fma/max-pooling.py"),
                # ReLU and Softmax
                build.peachpy("x86_64-fma/relu.py"),
                build.peachpy("x86_64-fma/softmax.py"),
                build.cc("x86_64-fma/softmax.c"),
                # FFT block accumulation
                build.peachpy("x86_64-fma/fft-block-mac.py"),
                # Tuple GEMM
                build.peachpy("x86_64-fma/blas/s8gemm.py"),
                build.peachpy("x86_64-fma/blas/c8gemm.py"),
                build.peachpy("x86_64-fma/blas/s4c6gemm.py"),
                # BLAS microkernels
                build.peachpy("x86_64-fma/blas/sgemm.py"),
                build.peachpy("x86_64-fma/blas/sdotxf.py"),
                build.peachpy("x86_64-fma/blas/shdotxf.py"),
            ]
        elif backend == "scalar":
            arch_nnpack_objects = [
                # Transformations
                build.cc("scalar/2d-fourier-8x8.c"),
                build.cc("scalar/2d-fourier-16x16.c"),
                build.cc("scalar/2d-winograd-8x8-3x3.c"),
                # ReLU and Softmax
                build.cc("scalar/relu.c"),
                build.cc("scalar/softmax.c"),
                # FFT block accumulation
                # build.cc("scalar/fft-block-mac.c"),
                # Tuple GEMM
                build.cc("scalar/blas/s2gemm.c"),
                build.cc("scalar/blas/s2gemm-transc.c"),
                build.cc("scalar/blas/cgemm.c"),
                build.cc("scalar/blas/cgemm-conjb.c"),
                build.cc("scalar/blas/cgemm-conjb-transc.c"),
                # BLAS microkernels
                build.cc("scalar/blas/sgemm.c"),
                build.cc("scalar/blas/sdotxf.c"),
                build.cc("scalar/blas/shdotxf.c"),
            ]
        elif backend == "psimd":
            arch_nnpack_objects = [
                # Transformations
                build.cc("psimd/2d-fourier-8x8.c"),
                build.cc("psimd/2d-fourier-16x16.c"),
                build.cc("psimd/2d-winograd-8x8-3x3.c"),
                # ReLU and Softmax
                build.cc("psimd/relu.c"),
                build.cc("psimd/softmax.c"),
                # FFT block accumulation
                build.cc("psimd/fft-block-mac.c"),
                # Tuple GEMM
                build.cc("psimd/blas/s4gemm.c"),
                build.cc("psimd/blas/c4gemm.c"),
                build.cc("psimd/blas/s4c2gemm.c"),
                build.cc("psimd/blas/c4gemm-conjb.c"),
                build.cc("psimd/blas/s4c2gemm-conjb.c"),
                build.cc("psimd/blas/c4gemm-conjb-transc.c"),
                build.cc("psimd/blas/s4c2gemm-conjb-transc.c"),
                # BLAS microkernels
                build.cc("psimd/blas/sgemm.c"),
                build.cc("psimd/blas/sdotxf.c"),
                build.cc("psimd/blas/shdotxf.c"),
            ]

        reference_layer_objects = [
            build.cc("ref/convolution-output.c"),
            build.cc("ref/convolution-input-gradient.c"),
            build.cc("ref/convolution-kernel.c"),
            build.cc("ref/fully-connected-output.c"),
            build.cc("ref/max-pooling-output.c"),
            build.cc("ref/softmax-output.c"),
            build.cc("ref/relu-output.c"),
            build.cc("ref/relu-input-gradient.c"),
        ]

        reference_fft_objects = [
            build.cc("ref/fft/aos.c"),
            build.cc("ref/fft/soa.c"),
            build.cc("ref/fft/forward-real.c"),
            build.cc("ref/fft/forward-dualreal.c"),
            build.cc("ref/fft/inverse-real.c"),
            build.cc("ref/fft/inverse-dualreal.c"),
        ]

        if backend == "x86_64":
            arch_fft_stub_objects = [
                build.peachpy("x86_64-fma/fft-soa.py"),
                build.peachpy("x86_64-fma/fft-aos.py"),
                build.peachpy("x86_64-fma/fft-dualreal.py"),
                build.peachpy("x86_64-fma/ifft-dualreal.py"),
                build.peachpy("x86_64-fma/fft-real.py"),
                build.peachpy("x86_64-fma/ifft-real.py"),
            ]

            arch_winograd_stub_objects = [
                build.peachpy("x86_64-fma/winograd-f6k3.py"),
            ]

            arch_math_stub_objects = [
            ]
        elif backend == "scalar":
            arch_fft_stub_objects = [
                build.cc("scalar/fft-aos.c"),
                build.cc("scalar/fft-soa.c"),
                build.cc("scalar/fft-real.c"),
                build.cc("scalar/fft-dualreal.c"),
            ]
        elif backend == "psimd":
            arch_fft_stub_objects = [
                build.cc("psimd/fft-aos.c"),
                build.cc("psimd/fft-soa.c"),
                build.cc("psimd/fft-real.c"),
                build.cc("psimd/fft-dualreal.c"),
            ]

            arch_winograd_stub_objects = [
                build.cc("psimd/winograd-f6k3.c"),
            ]

            arch_math_stub_objects = [
                build.cc("psimd/exp.c"),
            ]

        fft_objects = reference_fft_objects + arch_fft_stub_objects

        reference_blockmac_objects = [
            build.cc("ref/fft/block-mac.c"),
        ]

        nnpack_objects = nnpack_objects + arch_nnpack_objects

        build.static_library("nnpack", nnpack_objects)

    # Build tests for micro-kernels. Link to the micro-kernels implementations
    with build.options(source_dir="test", extra_include_dirs="test", deps=build.deps.googletest.core):

        build.unittest("fourier-reference-test",
            reference_fft_objects + [build.cxx("fourier/reference.cc")])

        if backend == "x86_64":
            build.smoketest("fourier-x86_64-avx2-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/x86_64-avx2.cc")])

            build.smoketest("winograd-x86_64-fma3-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/x86_64-fma3.cc")])

            build.smoketest("sgemm-x86_64-fma3-test",
                arch_nnpack_objects + [build.cxx("sgemm/x86_64-fma3.cc")])
        elif backend == "psimd":
            build.smoketest("fourier-psimd-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/psimd.cc")])

            build.smoketest("winograd-psimd-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/psimd.cc")])

            build.smoketest("sgemm-psimd-test",
                arch_nnpack_objects + [build.cxx("sgemm/psimd.cc")])
        elif backend == "scalar":
            build.smoketest("fourier-scalar-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/scalar.cc")])

            build.smoketest("sgemm-scalar-test",
                arch_nnpack_objects + [build.cxx("sgemm/scalar.cc")])

    # Build test for layers. Link to the library.
    with build.options(source_dir="test", include_dirs="test", deps={
                (build, build.deps.pthreadpool, build.deps.googletest.core, build.deps.fp16): any,
                "rt": build.target.is_linux
            }):

        build.smoketest("convolution-output-smoketest",
            reference_layer_objects + [build.cxx("convolution-output/smoke.cc")])
        build.unittest("convolution-output-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-output/alexnet.cc")])
        build.unittest("convolution-output-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-output/vgg-a.cc")])
        build.unittest("convolution-output-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-output/overfeat-fast.cc")])

        build.smoketest("convolution-output+relu-smoketest",
            reference_layer_objects + [build.cxx("convolution-output/smoke+relu.cc")])
        build.unittest("convolution-output+relu-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-output/alexnet+relu.cc")])
        build.unittest("convolution-output+relu-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-output/vgg-a+relu.cc")])
        build.unittest("convolution-output+relu-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-output/overfeat-fast+relu.cc")])

        build.smoketest("convolution-input-gradient-smoketest",
            reference_layer_objects + [build.cxx("convolution-input-gradient/smoke.cc")])
        build.unittest("convolution-input-gradient-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-input-gradient/alexnet.cc")])
        build.unittest("convolution-input-gradient-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-input-gradient/vgg-a.cc")])
        build.unittest("convolution-input-gradient-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-input-gradient/overfeat-fast.cc")])

        build.smoketest("convolution-kernel-gradient-smoketest",
            reference_layer_objects + [build.cxx("convolution-kernel-gradient/smoke.cc")])
        build.unittest("convolution-kernel-gradient-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-kernel-gradient/alexnet.cc")])
        build.unittest("convolution-kernel-gradient-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-kernel-gradient/vgg-a.cc")])
        build.unittest("convolution-kernel-gradient-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-kernel-gradient/overfeat-fast.cc")])

        build.smoketest("convolution-inference-smoketest",
            reference_layer_objects + [build.cxx("convolution-inference/smoke.cc")])
        build.unittest("convolution-inference-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-inference/alexnet.cc")])
        build.unittest("convolution-inference-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-inference/vgg-a.cc")])
        build.unittest("convolution-inference-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-inference/overfeat-fast.cc")])

        build.smoketest("convolution-inference+relu-smoketest",
            reference_layer_objects + [build.cxx("convolution-inference/smoke+relu.cc")])
        build.unittest("convolution-inference+relu-alexnet-test",
            reference_layer_objects + [build.cxx("convolution-inference/alexnet+relu.cc")])
        build.unittest("convolution-inference+relu-vgg-a-test",
            reference_layer_objects + [build.cxx("convolution-inference/vgg-a+relu.cc")])
        build.unittest("convolution-inference+relu-overfeat-fast-test",
            reference_layer_objects + [build.cxx("convolution-inference/overfeat-fast+relu.cc")])

        build.smoketest("fully-connected-output-smoketest",
            reference_layer_objects + [build.cxx("fully-connected-output/smoke.cc")])
        build.unittest("fully-connected-output-alexnet-test",
            reference_layer_objects + [build.cxx("fully-connected-output/alexnet.cc")])
        build.unittest("fully-connected-output-vgg-a-test",
            reference_layer_objects + [build.cxx("fully-connected-output/vgg-a.cc")])
        build.unittest("fully-connected-output-overfeat-fast-test",
            reference_layer_objects + [build.cxx("fully-connected-output/overfeat-fast.cc")])

        build.unittest("fully-connected-inference-alexnet-test",
            reference_layer_objects + [build.cxx("fully-connected-inference/alexnet.cc")])
        build.unittest("fully-connected-inference-vgg-a-test",
            reference_layer_objects + [build.cxx("fully-connected-inference/vgg-a.cc")])
        build.unittest("fully-connected-inference-overfeat-fast-test",
            reference_layer_objects + [build.cxx("fully-connected-inference/overfeat-fast.cc")])

        build.smoketest("max-pooling-output-smoketest",
            reference_layer_objects + [build.cxx("max-pooling-output/smoke.cc")])
        build.unittest("max-pooling-output-vgg-a-test",
            reference_layer_objects + [build.cxx("max-pooling-output/vgg-a.cc")])
        build.unittest("max-pooling-output-overfeat-fast",
            reference_layer_objects + [build.cxx("max-pooling-output/overfeat-fast.cc")])

        build.unittest("relu-output-alexnet-test",
            reference_layer_objects + [build.cxx("relu-output/alexnet.cc")])
        build.unittest("relu-output-vgg-a-test",
            reference_layer_objects + [build.cxx("relu-output/vgg-a.cc")])
        build.unittest("relu-output-overfeat-fast-test",
            reference_layer_objects + [build.cxx("relu-output/overfeat-fast.cc")])

        build.unittest("relu-input-gradient-alexnet-test",
            reference_layer_objects + [build.cxx("relu-input-gradient/alexnet.cc")])
        build.unittest("relu-input-gradient-vgg-a-test",
            reference_layer_objects + [build.cxx("relu-input-gradient/vgg-a.cc")])
        build.unittest("relu-input-gradient-overfeat-fast-test",
            reference_layer_objects + [build.cxx("relu-input-gradient/overfeat-fast.cc")])

        build.smoketest("softmax-output-smoketest",
            reference_layer_objects + [build.cxx("softmax-output/smoke.cc")])
        build.unittest("softmax-output-imagenet-test",
            reference_layer_objects + [build.cxx("softmax-output/imagenet.cc")])

    # Build benchmarking utilities
    with build.options(source_dir="bench", extra_include_dirs="bench", deps={
            (build, build.deps.pthreadpool): all,
            "rt": build.target.is_linux}):

        support_objects = [build.cc("median.c")]
        if build.target.is_x86_64:
            support_objects += [build.peachpy("memread.py")]
        else:
            support_objects += [build.cc("memread.c")]
        if build.target.is_linux and build.target.is_x86_64:
            support_objects += [build.cc("perf_counter.c")]

        build.executable("transform-bench",
            [build.cc("transform.c")] + support_objects)

        build.executable("convolution-benchmark",
            [build.cc("convolution.c")] + support_objects)

        build.executable("fully-connected-bench",
            [build.cc("fully-connected.c")] + support_objects)

        build.executable("pooling-benchmark",
            [build.cc("pooling.c")] + support_objects)

        build.executable("relu-benchmark",
            [build.cc("relu.c")] + support_objects)

    return build

if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
