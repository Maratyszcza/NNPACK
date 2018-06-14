#!/usr/bin/env python


import confu
parser = confu.standard_parser()
parser.add_argument("--backend", dest="backend", default="auto",
                    choices=["auto", "psimd", "scalar"])
parser.add_argument("--inference-only", dest="inference_only", default=False,
                    action="store_true",
                    help="Build only inference/forward pass functions to reduce library size")
parser.add_argument("--convolution-only", dest="convolution_only", default=False,
                    action="store_true",
                    help="Build only convolution functions to reduce library size")


def main(args):
    options = parser.parse_args(args)

    backend = options.backend
    if backend == "auto":
        if options.target.is_x86_64:
            backend = "x86_64"
        elif options.target.is_arm or options.target.is_arm64:
            backend = "arm"
        elif options.target.is_emscripten:
            backend = "scalar"
        else:
            backend = "psimd"

    build = confu.Build.from_options(options)

    macros = dict()
    if backend == "psimd":
        macros["NNP_BACKEND_PSIMD"] = 1
    if backend == "scalar":
        macros["NNP_BACKEND_SCALAR"] = 1
    export_macros = dict()
    export_macros["NNP_CONVOLUTION_ONLY"] = int(options.convolution_only)
    export_macros["NNP_INFERENCE_ONLY"] = int(options.inference_only)
    macros.update(export_macros)

    build.export_cpath("include", ["nnpack.h"])

    with build.options(source_dir="src", macros=macros,
            deps={
                (build.deps.pthreadpool, build.deps.cpuinfo, build.deps.fxdiv, build.deps.fp16): any,
                build.deps.psimd: backend == "psimd" or backend == "arm",
            },
            extra_include_dirs={
                ("src", "src/ref"): any,
                "src/x86_64-fma": options.target.is_x86_64
            }):

        nnpack_objects = [
            build.cc("init.c"),
            build.cc("convolution-inference.c"),
        ]
        if not options.convolution_only:
            # Fully-connected, pooling, Softmax, ReLU layers
            nnpack_objects += [
                build.cc("fully-connected-inference.c"),
                build.cc("pooling-output.c"),
                build.cc("softmax-output.c"),
                build.cc("relu-output.c"),
            ]
            if not options.inference_only:
                # Training functions for fully-connected and ReLU layers
                nnpack_objects += [
                    build.cc("fully-connected-output.c"),
                    build.cc("relu-input-gradient.c"),
                ]

        if not options.inference_only:
            # Training functions for convolutional layer
            nnpack_objects += [
                build.cc("convolution-output.c"),
                build.cc("convolution-input-gradient.c"),
                build.cc("convolution-kernel-gradient.c"),
            ]

        if backend == "x86_64":
            arch_nnpack_objects = [
                # Transformations
                build.peachpy("x86_64-fma/2d-fourier-8x8.py"),
                build.peachpy("x86_64-fma/2d-fourier-16x16.py"),
                build.peachpy("x86_64-fma/2d-winograd-8x8-3x3.py"),
                # Tuple GEMM
                build.peachpy("x86_64-fma/blas/s8gemm.py"),
                build.peachpy("x86_64-fma/blas/c8gemm.py"),
                build.peachpy("x86_64-fma/blas/s4c6gemm.py"),
                # Direct convolution
                build.peachpy("x86_64-fma/blas/conv1x1.py"),
                # BLAS microkernels
                build.peachpy("x86_64-fma/blas/sgemm.py"),
            ]
            if not options.convolution_only:
                arch_nnpack_objects += [
                    # Activations
                    build.peachpy("x86_64-fma/softmax.py"),
                    build.cc("x86_64-fma/softmax.c"),
                    build.peachpy("x86_64-fma/relu.py"),
                    # Pooling
                    build.peachpy("x86_64-fma/max-pooling.py"),
                    # BLAS microkernels
                    build.peachpy("x86_64-fma/blas/sdotxf.py"),
                    build.peachpy("x86_64-fma/blas/shdotxf.py"),
                ]
        elif backend == "scalar":
            arch_nnpack_objects = [
                # Transformations
                build.cc("scalar/2d-fourier-8x8.c"),
                build.cc("scalar/2d-fourier-16x16.c"),
                build.cc("scalar/2d-winograd-8x8-3x3.c"),
                # Tuple GEMM
                build.cc("scalar/blas/s2gemm.c"),
                build.cc("scalar/blas/cgemm-conjb.c"),
                # Direct convolution
                build.cc("scalar/blas/conv1x1.c"),
                # BLAS microkernels
                build.cc("scalar/blas/sgemm.c"),
            ]
            if not options.inference_only:
                arch_nnpack_objects += [
                    # Tuple GEMM
                    build.cc("scalar/blas/s2gemm-transc.c"),
                    build.cc("scalar/blas/cgemm.c"),
                    build.cc("scalar/blas/cgemm-conjb-transc.c"),
                ]
            if not options.convolution_only:
                arch_nnpack_objects += [
                    # Activations
                    build.cc("scalar/relu.c"),
                    build.cc("scalar/softmax.c"),
                    # BLAS microkernels
                    build.cc("scalar/blas/sdotxf.c"),
                    build.cc("scalar/blas/shdotxf.c"),
                ]
        elif backend == "arm":
            from confu import arm
            with build.options(isa=arm.neon+arm.fp16 if options.target.is_arm else None):
                arch_nnpack_objects = [
                    # Transformations
                    build.cc("psimd/2d-fourier-8x8.c"),
                    build.cc("psimd/2d-fourier-16x16.c"),
                    build.cc("neon/2d-winograd-8x8-3x3.c"),
                    build.cc("neon/2d-winograd-8x8-3x3-fp16.c"),
                    # Tuple GEMM
                    build.cc("neon/blas/h4gemm.c"),
                    build.cc("neon/blas/s4gemm.c"),
                    build.cc("neon/blas/c4gemm-conjb.c"),
                    build.cc("neon/blas/s4c2gemm-conjb.c"),
                    # Direct convolution
                    build.cc("neon/blas/conv1x1.c"),
                    # BLAS microkernels
                    build.cc("neon/blas/sgemm.c"),
                ]
                if not options.inference_only:
                    arch_nnpack_objects += [
                        # Transformations
                        build.cc("psimd/2d-winograd-8x8-3x3.c"),
                        # Tuple GEMM
                        build.cc("neon/blas/c4gemm.c"),
                        build.cc("neon/blas/s4c2gemm.c"),
                        build.cc("neon/blas/c4gemm-conjb-transc.c"),
                        build.cc("neon/blas/s4c2gemm-conjb-transc.c"),
                    ]
                if not options.convolution_only:
                    arch_nnpack_objects += [
                        # ReLU and Softmax
                        build.cc("neon/relu.c"),
                        build.cc("psimd/softmax.c"),
                        # BLAS microkernels
                        build.cc("neon/blas/sdotxf.c"),
                        build.cc("psimd/blas/shdotxf.c"),
                    ]
            if options.target.is_arm:
                # Functions implemented in assembly
                arch_nnpack_objects += [
                    build.cc("neon/blas/h4gemm-aarch32.S"),
                    build.cc("neon/blas/s4gemm-aarch32.S"),
                    build.cc("neon/blas/sgemm-aarch32.S"),
                ]
        elif backend == "psimd":
            arch_nnpack_objects = [
                # Transformations
                build.cc("psimd/2d-fourier-8x8.c"),
                build.cc("psimd/2d-fourier-16x16.c"),
                build.cc("psimd/2d-winograd-8x8-3x3.c"),
                # Tuple GEMM
                build.cc("psimd/blas/s4gemm.c"),
                build.cc("psimd/blas/c4gemm-conjb.c"),
                build.cc("psimd/blas/s4c2gemm-conjb.c"),
                # Direct convolution
                build.cc("psimd/blas/conv1x1.c"),
                # BLAS microkernels
                build.cc("psimd/blas/sgemm.c"),
            ]
            if not options.inference_only:
                arch_nnpack_objects += [
                    # Tuple GEMM
                    build.cc("psimd/blas/c4gemm.c"),
                    build.cc("psimd/blas/s4c2gemm.c"),
                    build.cc("psimd/blas/c4gemm-conjb-transc.c"),
                    build.cc("psimd/blas/s4c2gemm-conjb-transc.c"),
                ]
            if not options.convolution_only:
                arch_nnpack_objects += [
                    # Activations
                    build.cc("psimd/relu.c"),
                    build.cc("psimd/softmax.c"),
                    # BLAS microkernels
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

            arch_winograd_stub_objects = [
                build.cc("scalar/winograd-f6k3.c"),
            ]
        elif backend == "psimd" or backend == "arm":
            arch_fft_stub_objects = [
                build.cc("psimd/fft-aos.c"),
                build.cc("psimd/fft-soa.c"),
                build.cc("psimd/fft-real.c"),
                build.cc("psimd/fft-dualreal.c"),
            ]

            if backend == "psimd":
                arch_winograd_stub_objects = [
                    build.cc("psimd/winograd-f6k3.c"),
                ]
            else:
                # ARM NEON Winograd transform optionally uses FP16 storage
                with build.options(isa=arm.neon+arm.fp16 if options.target.is_arm else None):
                    arch_winograd_stub_objects = [
                        build.cc("neon/winograd-f6k3.c"),
                    ]

            arch_math_stub_objects = [
                build.cc("psimd/exp.c"),
            ]

        fft_objects = reference_fft_objects + arch_fft_stub_objects

        nnpack_objects = nnpack_objects + arch_nnpack_objects

        build.static_library("nnpack", nnpack_objects)

    # Build tests for micro-kernels. Link to the micro-kernels implementations
    with build.options(source_dir="test", extra_include_dirs="test",
            deps={
                (build.deps.googletest, build.deps.cpuinfo, build.deps.clog, build.deps.fp16): any,
                "log": build.target.is_android}):

        build.unittest("fourier-reference-test",
            reference_fft_objects + [build.cxx("fourier/reference.cc")])

        if backend == "x86_64":
            build.smoketest("fourier-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/x86_64-avx2.cc")])

            build.smoketest("winograd-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/x86_64-fma3.cc")])

            build.smoketest("sgemm-test",
                arch_nnpack_objects + [build.cxx("sgemm/x86_64-fma3.cc")])
        elif backend == "psimd":
            build.smoketest("fourier-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/psimd.cc")])

            build.smoketest("winograd-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/psimd.cc")])

            build.smoketest("sgemm-test",
                arch_nnpack_objects + [build.cxx("sgemm/psimd.cc")])
        elif backend == "arm":
            # No ARM-specific Fourier implementation; use PSIMD
            build.smoketest("fourier-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/psimd.cc")])

            build.smoketest("winograd-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/neon.cc")])

            build.smoketest("sgemm-test",
                arch_nnpack_objects + [build.cxx("sgemm/neon.cc")])

            build.smoketest("sxgemm-test",
                arch_nnpack_objects + [build.cxx("sxgemm/neon.cc")])

            build.smoketest("hxgemm-test",
                arch_nnpack_objects + [build.cxx("hxgemm/neon.cc")])
        elif backend == "scalar":
            build.smoketest("fourier-test",
                reference_fft_objects + arch_fft_stub_objects + [build.cxx("fourier/scalar.cc")])

            build.smoketest("winograd-test",
                arch_winograd_stub_objects + arch_nnpack_objects + [build.cxx("winograd/scalar.cc")])

            build.smoketest("sgemm-test",
                arch_nnpack_objects + [build.cxx("sgemm/scalar.cc")])

    # Build test for layers. Link to the library.
    with build.options(source_dir="test", include_dirs="test", deps={
                (build, build.deps.pthreadpool, build.deps.cpuinfo, build.deps.clog, build.deps.googletest.core, build.deps.fp16): any,
                "rt": build.target.is_linux,
                "log": build.target.is_android,
            }):

        if not options.inference_only:
            build.smoketest("convolution-output-smoketest",
                reference_layer_objects + [build.cxx("convolution-output/smoke.cc")])
            build.unittest("convolution-output-alexnet-test",
                reference_layer_objects + [build.cxx("convolution-output/alexnet.cc")])
            build.unittest("convolution-output-vgg-a-test",
                reference_layer_objects + [build.cxx("convolution-output/vgg-a.cc")])
            build.unittest("convolution-output-overfeat-fast-test",
                reference_layer_objects + [build.cxx("convolution-output/overfeat-fast.cc")])

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

        if not options.convolution_only:
            build.unittest("fully-connected-inference-alexnet-test",
                reference_layer_objects + [build.cxx("fully-connected-inference/alexnet.cc")])
            build.unittest("fully-connected-inference-vgg-a-test",
                reference_layer_objects + [build.cxx("fully-connected-inference/vgg-a.cc")])
            build.unittest("fully-connected-inference-overfeat-fast-test",
                reference_layer_objects + [build.cxx("fully-connected-inference/overfeat-fast.cc")])

            if not options.inference_only:
                build.smoketest("fully-connected-output-smoketest",
                    reference_layer_objects + [build.cxx("fully-connected-output/smoke.cc")])
                build.unittest("fully-connected-output-alexnet-test",
                    reference_layer_objects + [build.cxx("fully-connected-output/alexnet.cc")])
                build.unittest("fully-connected-output-vgg-a-test",
                    reference_layer_objects + [build.cxx("fully-connected-output/vgg-a.cc")])
                build.unittest("fully-connected-output-overfeat-fast-test",
                    reference_layer_objects + [build.cxx("fully-connected-output/overfeat-fast.cc")])

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

            if not options.inference_only:
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

    # Build automatic benchmarks
    with build.options(source_dir="bench", extra_include_dirs=["bench", "test"], macros=macros, deps={
            (build, build.deps.pthreadpool, build.deps.cpuinfo, build.deps.clog, build.deps.fp16, build.deps.googlebenchmark): all,
            "rt": build.target.is_linux,
            "log": build.target.is_android}):

        build.benchmark("convolution-inference-bench", build.cxx("convolution-inference.cc"))
        build.benchmark("sgemm-bench", build.cxx("sgemm.cc"))
        build.benchmark("sxgemm-bench", build.cxx("sxgemm.cc"))
        build.benchmark("hxgemm-bench", build.cxx("hxgemm.cc"))
        build.benchmark("conv1x1-bench", build.cxx("conv1x1.cc"))
        build.benchmark("winograd-bench", build.cxx("winograd.cc"))

    # Build benchmarking utilities
    if not options.inference_only and not build.target.is_android:
        with build.options(source_dir="bench", extra_include_dirs="bench", macros=macros, deps={
                (build, build.deps.pthreadpool, build.deps.cpuinfo, build.deps.clog): all,
                "rt": build.target.is_linux,
                "log": build.target.is_android}):

            support_objects = [build.cc("median.c")]
            if build.target.is_x86_64:
                support_objects += [build.peachpy("memread.py")]
            else:
                support_objects += [build.cc("memread.c")]
            if build.target.is_linux and build.target.is_x86_64:
                support_objects += [build.cc("perf_counter.c")]

            build.executable("transform-benchmark",
                [build.cc("transform.c")] + support_objects)

            build.executable("convolution-benchmark",
                [build.cc("convolution.c")] + support_objects)

            if not options.convolution_only:
                build.executable("fully-connected-benchmark",
                    [build.cc("fully-connected.c")] + support_objects)

                build.executable("pooling-benchmark",
                    [build.cc("pooling.c")] + support_objects)

                build.executable("relu-benchmark",
                    [build.cc("relu.c")] + support_objects)

    return build

if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
