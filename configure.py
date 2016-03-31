#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import glob
import argparse
import ninja_syntax


class Configuration:

    def __init__(self, options, ninja_build_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.ninja")):
        self.output = open(ninja_build_file, "w")
        self.writer = ninja_syntax.Writer(self.output)
        self.source_dir = None
        self.build_dir = None
        self.artifact_dir = None
        self.include_dirs = []
        self.object_ext = ".o"

        # Variables
        self.build, self.host = Configuration.detect_system(options.host)

        cflags = ["-g", "-std=gnu11"]
        cxxflags = ["-g", "-std=gnu++11"]
        ldflags = ["-g"]
        if self.host in ["x86_64-linux-gnu", "x86_64-nacl-glibc", "x86_64-nacl-newlib"]:
            cflags.append("-pthread")
            cxxflags.append("-pthread")
            ldflags.append("-pthread")

        if self.host == "x86_64-linux-gnu":
            self.writer.variable("imageformat", "elf")
            self.writer.variable("abi", "sysv")
            self.writer.variable("ar", "ar")
            self.writer.variable("cc", "gcc")
            self.writer.variable("cxx", "g++")
            ldflags.append("-Wl,-fuse-ld=gold")
        elif self.host == "x86_64-windows-msvc":
            import _winreg
            with _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\VisualStudio\14.0_Config") as vs_key:
                vs_tools_dir, _ = _winreg.QueryValueEx(vs_key, "InstallDir")
                vs_dir = os.path.abspath(os.path.join(vs_tools_dir, "..", ".."))

            self.writer.variable("imageformat", "ms-coff")
            self.writer.variable("abi", "ms")
            self.writer.variable("ar", "lib")
            self.writer.variable("visual_studio_dir", vs_dir)
            self.writer.variable("cc", "\"" + os.path.join("$visual_studio_dir", "VC", "Clang 3.7", "bin", "x86", "clang.exe") + "\"")
            self.writer.variable("cxx", "\"" + os.path.join("$visual_studio_dir", "VC", "Clang 3.7", "bin", "x86", "clang.exe") + "\"")
        elif self.host in ["x86_64-nacl-newlib", "x86_64-nacl-glibc"]:
            self.writer.variable("imageformat", "elf")
            self.writer.variable("abi", "nacl")
            self.writer.variable("nacl_sdk_dir", os.getenv("NACL_SDK_ROOT"))
            self.writer.variable("pepper_include_dir", os.path.join("$nacl_sdk_dir", "include"))

            toolchain_library_subdir_map = {
                ("x86_64-linux-gnu", "x86_64-nacl-glibc"):  ("linux_x86_glibc", "glibc_x86_64"),
                ("x86_64-linux-gnu", "x86_64-nacl-newlib"): ("linux_pnacl", "clang-newlib_x86_64"),
                ("x86_64-osx",       "x86_64-nacl-glibc"):  ("mac_x86_glibc", "glibc_x86_64"),
                ("x86_64-osx",       "x86_64-nacl-newlib"): ("mac_pnacl", "clang-newlib_x86_64"),
            }
            if (self.build, self.host) in toolchain_library_subdir_map:
                toolchain_subdir, library_subdir = toolchain_library_subdir_map.get((self.build, self.host))
                self.writer.variable("nacl_toolchain_dir", os.path.join("$nacl_sdk_dir", "toolchain", toolchain_subdir))
                self.writer.variable("pepper_lib_dir", os.path.join("$nacl_sdk_dir", "lib", library_subdir, "Release"))
            else:
                print("Error: cross-compilation for %s is not supported on %s" % (self.host, self.build))
                sys.exit(1)
            toolchain_compiler_map = {
                "x86_64-nacl-glibc": ("x86_64-nacl-gcc", "x86_64-nacl-g++"),
                "x86_64-nacl-newlib": ("x86_64-nacl-clang", "x86_64-nacl-clang++"),
            }
            toolchain_cc, toolchain_cxx = toolchain_compiler_map[self.host]
            self.writer.variable("ar", os.path.join("$nacl_toolchain_dir", "bin", "x86_64-nacl-ar"))
            self.writer.variable("cc", os.path.join("$nacl_toolchain_dir", "bin", toolchain_cc))
            self.writer.variable("cxx", os.path.join("$nacl_toolchain_dir", "bin", toolchain_cxx))
            if self.build in ["x86_64-linux-gnu", "x86_64-osx"]:
                self.writer.variable("sel_ldr", os.path.join("$nacl_sdk_dir", "tools", "sel_ldr.py"))
        elif self.host == "x86_64-osx":
            self.writer.variable("imageformat", "mach-o")
            self.writer.variable("abi", "sysv")
            self.writer.variable("ar", "ar")
            self.writer.variable("cc", "clang")
            self.writer.variable("cxx", "clang++")
        else:
            print("Unsupported platform: %s" % sys.platform, file=sys.stdout)
            sys.exit(1)

        self.writer.variable("cflags", " ".join(cflags))
        self.writer.variable("cxxflags", " ".join(cxxflags))
        self.writer.variable("ldflags", " ".join(ldflags))
        self.writer.variable("optflags", "-O3")
        self.writer.variable("python2", "python")

        # Rules
        self.writer.rule("lib", "$ar rcs $out $in",
            description="LIB $descpath")
        self.writer.rule("cc", "$cc -o $out -c $in -MMD -MF $out.d $optflags $cflags $includes",
            deps="gcc", depfile="$out.d",
            description="CC $descpath")
        self.writer.rule("cxx", "$cxx -o $out -c $in -MMD -MF $out.d $optflags $cxxflags $includes",
            deps="gcc", depfile="$out.d",
            description="CXX $descpath")
        self.writer.rule("ccld", "$cc $ldflags $libdirs -o $out $in $libs",
            description="CCLD $descpath")
        self.writer.rule("cxxld", "$cxx $ldflags $libdirs -o $out $in $libs",
            description="CXXLD $descpath")
        self.writer.rule("peachpy", "$python2 -m peachpy.x86_64 -mabi=$abi -g4 -mimage-format=$imageformat -MMD -MF $object.d -emit-c-header $header -o $object $in",
            deps="gcc", depfile="$object.d",
            description="PEACHPY[x86-64] $descpath")
        if self.host in ["x86_64-nacl-newlib", "x86_64-nacl-glibc"]:
            self.writer.rule("run", "$sel_ldr -- $in $args",
                description="RUN $descpath", pool="console")
        else:
            self.writer.rule("run", "$in $args",
                description="RUN $descpath", pool="console")

    @staticmethod
    def detect_system(host):
        if sys.platform.startswith("linux"):
            build = "x86_64-linux-gnu"
        elif sys.platform == "darwin":
            build = "x86_64-osx"
        elif sys.platform == "win32":
            build = "x86_64-windows-msvc"
        else:
            print("Error: failed to detect build platform: sys.platform = %s" % sys.platform, file=sys.stdout)
            sys.exit(1)
        if host is not None:
            if os.getenv("NACL_SDK_ROOT") is None:
                print("Error: failed to find NaCl SDK: NACL_SDK_ROOT environment variable is not set", file=sys.stdout)
                sys.exit(1)
            return build, host
        else:
            return build, build


    def _compile(self, rule, source_file, object_file, header_file=None, extra_flags={}):
        if not os.path.isabs(source_file):
            source_file = os.path.join(self.source_dir, source_file)
        if object_file is None:
            object_file = os.path.join(self.build_dir, os.path.relpath(source_file, self.source_dir)) + self.object_ext
        elif not os.path.isabs(object_file):
            object_file = os.path.join(self.build_dir, object_file)
        variables = {
            "descpath": os.path.relpath(source_file, self.source_dir)
        }
        for key, values in extra_flags.items():
            if values:
                if isinstance(values, str):
                    values = [values]
                variables[key] = "$" + key + " " + " ".join(values)
        if rule != "peachpy":
            if self.include_dirs:
                variables["includes"] = " ".join(["-I" + i for i in self.include_dirs])
        else:
            variables["object"] = object_file
            if header_file is None:
                header_file = os.path.join(self.build_dir, os.path.relpath(source_file, self.source_dir)) + ".h"
            variables["header"] = header_file
        self.writer.build(object_file, rule, source_file, variables=variables)
        return object_file


    def cc(self, source_file, object_file=None, extra_cflags=[]):
        return self._compile("cc", source_file, object_file, extra_flags={"cflags": extra_cflags})


    def peachpy(self, source_file, object_file=None, header_file=None):
        return self._compile("peachpy", source_file, object_file)


    def cxx(self, source_file, object_file=None, extra_cxxflags=[]):
        return self._compile("cxx", source_file, object_file, extra_flags={"cxxflags": extra_cxxflags})


    def _link(self, rule, object_files, binary_file, binary_dir, lib_dirs, libs, extra_ldflags):
        if not os.path.isabs(binary_file):
            binary_file = os.path.join(binary_dir, binary_file)
        variables = {
            "descpath": os.path.relpath(binary_file, binary_dir)
        }
        if lib_dirs:
            variables["libdirs"] = " ".join(["-L" + l for l in lib_dirs])
        if libs:
            variables["libs"] = " ".join(["-l" + l for l in libs])
        if extra_ldflags:
            if isinstance(extra_ldflags, str):
                extra_ldflags = [extra_ldflags]
            variables["ldflags"] = "$ldflags " + " ".join(extra_ldflags)
        self.writer.build(binary_file, rule, object_files, variables=variables)
        return binary_file


    def ccld(self, object_files, binary_file, lib_dirs=[], libs=[], extra_ldflags=[]):
        return self._link("ccld", object_files, binary_file, self.artifact_dir, lib_dirs, libs, extra_ldflags)


    def cxxld(self, object_files, binary_file, lib_dirs=[], libs=[], extra_ldflags=[]):
        return self._link("cxxld", object_files, binary_file, self.artifact_dir, lib_dirs, libs, extra_ldflags)


    def lib(self, object_files, library_file):
        library_basename = os.path.basename(library_file)
        if not library_basename.startswith("lib"):
            library_basename = "lib" + library_basename
        if not library_basename.endswith(".a"):
            library_basename = library_basename + ".a"
        library_file = os.path.join(os.path.dirname(library_file), library_basename)
        if not os.path.isabs(library_file):
            library_file = os.path.join(self.artifact_dir, library_file)
        variables = {
            "descpath": os.path.relpath(library_file, self.artifact_dir)
        }
        self.writer.build(library_file, "lib", object_files, variables=variables)
        return library_file


    def run(self, executable_file, target, args=""):
        variables = {
            "descpath": os.path.relpath(executable_file, self.artifact_dir)
        }
        if args:
            variables["args"] = args
        self.writer.build(target, "run", executable_file, variables=variables)

    def default(self, targets):
        if not isinstance(targets, list):
            targets = [targets]
        self.writer.default([ninja_syntax.escape(target).replace(":", "$:") for target in targets])

    def phony(self, target, deps):
        self.writer.build(target, "phony", deps)


parser = argparse.ArgumentParser(description="NNPACK configuration script")
parser.add_argument("--host", dest="host", choices=("x86_64-linux-gnu", "x86_64-osx", "x86_64-nacl-newlib", "x86_64-nacl-glibc"))
parser.add_argument("--enable-mkl", dest="use_mkl", action="store_true")
parser.add_argument("--enable-openblas", dest="use_openblas", action="store_true")
parser.add_argument("--enable-blis", dest="use_blis", action="store_true")


def main():
    options = parser.parse_args()

    config = Configuration(options)

    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Build gtest
    gtest_dir = os.path.join(root_dir, "third-party", "gtest-1.7.0")
    config.source_dir = os.path.join(gtest_dir, "src")
    config.build_dir = os.path.join(gtest_dir, "lib")
    config.include_dirs = [os.path.join(gtest_dir, "include"), gtest_dir]

    gtest_objects = [config.cxx("gtest-all.cc")]

    # Build pthreadpool
    pthreadpool_dir = os.path.join(root_dir, "third-party", "pthreadpool")
    config.source_dir = os.path.join(pthreadpool_dir, "src")
    config.build_dir = os.path.join(pthreadpool_dir, "lib")
    config.include_dirs = [os.path.join(pthreadpool_dir, "include")]

    pthreadpool_objects = [config.cc("pthreadpool.c")]

    # Build the library
    config.source_dir = os.path.join(root_dir, "src")
    config.build_dir = os.path.join(root_dir, "build")
    config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(root_dir, "src", "ref"), os.path.join(pthreadpool_dir, "include")]

    nnpack_objects = [
        config.cc("init.c"),
        config.cc("convolution-output.c"),
        config.cc("convolution-input-gradient.c"),
        config.cc("convolution-kernel.c"),
        config.cc("convolution-inference.c"),
        config.cc("fully-connected-output.c"),
        config.cc("fully-connected-inference.c"),
        config.cc("pooling-output.c"),
        config.cc("relu-output.c"),
        config.cc("relu-input-gradient.c"),
    ]

    x86_64_nnpack_objects = [
        # Transformations
        config.peachpy("x86_64-fma/2d-fft-8x8.py"),
        config.peachpy("x86_64-fma/2d-fft-16x16.py"),
        config.peachpy("x86_64-fma/2d-wt-8x8-3x3.py"),
        # Pooling
        config.peachpy("x86_64-fma/max-pooling.py"),
        # ReLU
        config.peachpy("x86_64-fma/relu.py"),
        # FFT block accumulation
        config.peachpy("x86_64-fma/fft-block-mac.py"),
        # Tuple GEMM
        config.peachpy("x86_64-fma/c8gemm.py"),
        config.peachpy("x86_64-fma/s8gemm.py"),
        # BLAS microkernels
        config.peachpy("x86_64-fma/sgemm.py"),
        config.peachpy("x86_64-fma/sdotxf.py"),
    ]

    reference_layer_objects = [
        config.cc("ref/convolution-output.c"),
        config.cc("ref/convolution-input-gradient.c"),
        config.cc("ref/convolution-kernel.c"),
        config.cc("ref/fully-connected-output.c"),
        config.cc("ref/pooling-output.c"),
        config.cc("ref/softmax-output.c"),
        config.cc("ref/relu-output.c"),
        config.cc("ref/relu-input-gradient.c"),
    ]

    reference_fft_objects = [
        config.cc("ref/fft/aos.c"),
        config.cc("ref/fft/soa.c"),
        config.cc("ref/fft/forward-real.c"),
        config.cc("ref/fft/forward-dualreal.c"),
        config.cc("ref/fft/inverse-real.c"),
        config.cc("ref/fft/inverse-dualreal.c"),
    ]

    x86_64_fft_stub_objects = [
        config.peachpy("x86_64-fma/fft-soa.py"),
        config.peachpy("x86_64-fma/fft-aos.py"),
        config.peachpy("x86_64-fma/fft-dualreal.py"),
        config.peachpy("x86_64-fma/ifft-dualreal.py"),
        config.peachpy("x86_64-fma/fft-real.py"),
        config.peachpy("x86_64-fma/ifft-real.py"),
    ]

    x86_64_winograd_stub_objects = [
        config.peachpy("x86_64-fma/winograd-f6k3.py"),
    ]

    fft_objects = reference_fft_objects + x86_64_fft_stub_objects

    reference_blockmac_objects = [
        config.cc("ref/fft/block-mac.c"),
    ]

    nnpack_objects = nnpack_objects + x86_64_nnpack_objects + pthreadpool_objects

    # Build the library
    config.artifact_dir = os.path.join(root_dir, "lib")
    config.default(config.lib(nnpack_objects, "nnpack"))

    # Build Native Client module
    if config.host in ["x86_64-nacl-glibc", "x86_64-nacl-newlib"]:
        config.source_dir = os.path.join(root_dir, "src")
        config.build_dir = os.path.join(root_dir, "build")
        config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(root_dir, "src"), os.path.join(pthreadpool_dir, "include")]
        config.artifact_dir = os.path.join(root_dir, "web")

        module_source_files = ["nacl/entry.c", "nacl/instance.c", "nacl/interfaces.c", "nacl/messaging.c", "nacl/stringvars.c", "nacl/benchmark.c"]
        nacl_module_objects = [config.cc(source_file, extra_cflags="-I$pepper_include_dir") for source_file in module_source_files]
        nacl_module_binary = \
            config.ccld(nnpack_objects + nacl_module_objects, "nnpack.x86_64.nexe", lib_dirs=["$pepper_lib_dir"], libs=["m", "ppapi"])
        config.default(nacl_module_binary)

    # Build unit tests
    if config.host not in ["x86_64-nacl-glibc"]:
        config.source_dir = os.path.join(root_dir, "test")
        config.build_dir = os.path.join(root_dir, "build", "test")
        config.artifact_dir = os.path.join(root_dir, "bin")
        config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(pthreadpool_dir, "include"), os.path.join(root_dir, "test"), os.path.join(gtest_dir, "include")]
        unittest_libs = ["m"]

        fourier_reference_test_binary = \
            config.cxxld(reference_fft_objects + [config.cxx("fourier/reference.cc")] + gtest_objects,
                "fourier-reference-test", libs=unittest_libs)
        config.run(fourier_reference_test_binary, "fourier-reference-test")
        fourier_x86_64_avx2_test_binary = \
            config.cxxld(reference_fft_objects + x86_64_fft_stub_objects + [config.cxx("fourier/x86_64-avx2.cc")] + gtest_objects,
                "fourier-x86_64-avx2-test", libs=unittest_libs)
        config.run(fourier_x86_64_avx2_test_binary, "fourier-x86_64-avx2-test")

        winograd_x86_64_fma3_test_binary = \
            config.cxxld(x86_64_winograd_stub_objects + [config.cxx("winograd/x86_64-fma3.cc")] + gtest_objects,
                "winograd-x86_64-fma3-test", libs=unittest_libs)
        config.run(winograd_x86_64_fma3_test_binary, "winograd-x86_64-fma3-test")

        convolution_output_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/smoke.cc")] + gtest_objects,
                "convolution-output-smoketest", libs=unittest_libs)
        convolution_output_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/alexnet.cc")] + gtest_objects,
                "convolution-output-alexnet-test", libs=unittest_libs)
        convolution_output_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/vgg-a.cc")] + gtest_objects,
                "convolution-output-vgg-a-test", libs=unittest_libs)
        convolution_output_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/overfeat-fast.cc")] + gtest_objects,
                "convolution-output-overfeat-fast-test", libs=unittest_libs)
        config.run(convolution_output_smoke_test_binary, "convolution-output-smoketest")
        config.run(convolution_output_alexnet_test_binary, "convolution-output-alexnet-test")
        config.run(convolution_output_vgg_a_test_binary, "convolution-output-vgg-a-test")
        config.run(convolution_output_overfeat_fast_test_binary, "convolution-output-overfeat-fast-test")
        config.phony("convolution-output-test",
            ["convolution-output-smoketest", "convolution-output-alexnet-test", "convolution-output-vgg-a-test", "convolution-output-overfeat-fast-test"])

        convolution_input_gradient_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/smoke.cc")] + gtest_objects,
                "convolution-input-gradient-smoketest", libs=unittest_libs)
        convolution_input_gradient_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/alexnet.cc")] + gtest_objects,
                "convolution-input-gradient-alexnet-test", libs=unittest_libs)
        convolution_input_gradient_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/vgg-a.cc")] + gtest_objects,
                "convolution-input-gradient-vgg-a-test", libs=unittest_libs)
        convolution_input_gradient_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/overfeat-fast.cc")] + gtest_objects,
                "convolution-input-gradient-overfeat-fast-test", libs=unittest_libs)
        config.run(convolution_input_gradient_smoke_test_binary, "convolution-input-gradient-smoketest")
        config.run(convolution_input_gradient_alexnet_test_binary, "convolution-input-gradient-alexnet-test")
        config.run(convolution_input_gradient_vgg_a_test_binary, "convolution-input-gradient-vgg-a-test")
        config.run(convolution_input_gradient_overfeat_fast_test_binary, "convolution-input-gradient-overfeat-fast-test")
        config.phony("convolution-input-gradient-test",
            ["convolution-input-gradient-smoketest", "convolution-input-gradient-alexnet-test", "convolution-input-gradient-vgg-a-test", "convolution-input-gradient-overfeat-fast-test"])

        convolution_kernel_gradient_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/smoke.cc")] + gtest_objects,
                "convolution-kernel-gradient-smoketest", libs=unittest_libs)
        convolution_kernel_gradient_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/alexnet.cc")] + gtest_objects,
                "convolution-kernel-gradient-alexnet-test", libs=unittest_libs)
        convolution_kernel_gradient_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/vgg-a.cc")] + gtest_objects,
                "convolution-kernel-gradient-vgg-a-test", libs=unittest_libs)
        convolution_kernel_gradient_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/overfeat-fast.cc")] + gtest_objects,
                "convolution-kernel-gradient-overfeat-fast-test", libs=unittest_libs)
        config.run(convolution_kernel_gradient_smoke_test_binary, "convolution-kernel-gradient-smoketest")
        config.run(convolution_kernel_gradient_alexnet_test_binary, "convolution-kernel-gradient-alexnet-test")
        config.run(convolution_kernel_gradient_vgg_a_test_binary, "convolution-kernel-gradient-vgg-a-test")
        config.run(convolution_kernel_gradient_overfeat_fast_test_binary, "convolution-kernel-gradient-overfeat-fast-test")
        config.phony("convolution-kernel-gradient-test",
            ["convolution-kernel-gradient-smoketest", "convolution-kernel-gradient-alexnet-test", "convolution-kernel-gradient-vgg-a-test", "convolution-kernel-gradient-overfeat-fast-test"])

        convolution_inference_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/smoke.cc")] + gtest_objects,
                "convolution-inference-smoketest", libs=unittest_libs)
        convolution_inference_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/alexnet.cc")] + gtest_objects,
                "convolution-inference-alexnet-test", libs=unittest_libs)
        convolution_inference_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/vgg-a.cc")] + gtest_objects,
                "convolution-inference-vgg-a-test", libs=unittest_libs)
        convolution_inference_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/overfeat-fast.cc")] + gtest_objects,
                "convolution-inference-overfeat-fast-test", libs=unittest_libs)
        config.run(convolution_inference_smoke_test_binary, "convolution-inference-smoketest")
        config.run(convolution_inference_alexnet_test_binary, "convolution-inference-alexnet-test")
        config.run(convolution_inference_vgg_a_test_binary, "convolution-inference-vgg-a-test")
        config.run(convolution_inference_overfeat_fast_test_binary, "convolution-inference-overfeat-fast-test")
        config.phony("convolution-inference-test",
            ["convolution-inference-smoketest", "convolution-inference-alexnet-test", "convolution-inference-vgg-a-test", "convolution-inference-overfeat-fast-test"])

        fully_connected_output_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/smoke.cc")] + gtest_objects,
                "fully-connected-output-smoketest", libs=unittest_libs)
        fully_connected_output_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/alexnet.cc")] + gtest_objects,
                "fully-connected-output-alexnet-test", libs=unittest_libs)
        fully_connected_output_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/vgg-a.cc")] + gtest_objects,
                "fully-connected-output-vgg-a-test", libs=unittest_libs)
        fully_connected_output_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/overfeat-fast.cc")] + gtest_objects,
                "fully-connected-output-overfeat-fast-test", libs=unittest_libs)
        config.run(fully_connected_output_smoke_test_binary, "fully-connected-output-smoketest")
        config.run(fully_connected_output_alexnet_test_binary, "fully-connected-output-alexnet-test")
        config.run(fully_connected_output_vgg_a_test_binary, "fully-connected-output-vgg-a-test")
        config.run(fully_connected_output_overfeat_fast_test_binary, "fully-connected-output-overfeat-fast-test")
        config.phony("fully-connected-output-test",
            ["fully-connected-output-smoketest", "fully-connected-output-alexnet-test", "fully-connected-output-vgg-a-test", "fully-connected-output-overfeat-fast-test"])

        fully_connected_inference_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/alexnet.cc")] + gtest_objects, "fully-connected-inference-alexnet-test", libs=unittest_libs)
        fully_connected_inference_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/vgg-a.cc")] + gtest_objects, "fully-connected-inference-vgg-a-test", libs=unittest_libs)
        fully_connected_inference_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/overfeat-fast.cc")] + gtest_objects, "fully-connected-inference-overfeat-fast-test", libs=unittest_libs)
        config.run(fully_connected_inference_alexnet_test_binary, "fully-connected-inference-alexnet-test")
        config.run(fully_connected_inference_vgg_a_test_binary, "fully-connected-inference-vgg-a-test")
        config.run(fully_connected_inference_overfeat_fast_test_binary, "fully-connected-inference-overfeat-fast-test")
        config.phony("fully-connected-inference-test",
            ["fully-connected-inference-alexnet-test", "fully-connected-inference-vgg-a-test", "fully-connected-inference-overfeat-fast-test"])

        pooling_output_smoke_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/smoke.cc")] + gtest_objects, "pooling-output-smoketest", libs=unittest_libs)
        pooling_output_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/vgg-a.cc")] + gtest_objects, "pooling-output-vgg-a-test", libs=unittest_libs)
        pooling_output_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/overfeat-fast.cc")] + gtest_objects, "pooling-output-overfeat-fast", libs=unittest_libs)
        config.run(pooling_output_smoke_test_binary, "pooling-output-smoketest")
        config.run(pooling_output_vgg_a_test_binary, "pooling-output-vgg-a-test")
        config.run(pooling_output_overfeat_fast_test_binary, "pooling-output-overfeat-fast-test")
        config.phony("pooling-output-test",
            ["pooling-output-smoketest", "pooling-output-vgg-a-test", "pooling-output-overfeat-fast-test"])

        relu_output_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/alexnet.cc")] + gtest_objects, "relu-output-alexnet-test", libs=unittest_libs)
        relu_output_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/vgg-a.cc")] + gtest_objects, "relu-output-vgg-a-test", libs=unittest_libs)
        relu_output_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/overfeat-fast.cc")] + gtest_objects, "relu-output-overfeat-fast-test", libs=unittest_libs)
        config.run(relu_output_alexnet_test_binary, "relu-output-alexnet-test")
        config.run(relu_output_vgg_a_test_binary, "relu-output-vgg-a-test")
        config.run(relu_output_overfeat_fast_test_binary, "relu-output-overfeat-fast-test")
        config.phony("relu-output-test",
            ["relu-output-alexnet-test", "relu-output-vgg-a-test", "relu-output-overfeat-fast-test"])

        relu_input_gradient_alexnet_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/alexnet.cc")] + gtest_objects, "relu-input-gradient-alexnet-test", libs=unittest_libs)
        relu_input_gradient_vgg_a_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/vgg-a.cc")] + gtest_objects, "relu-input-gradient-vgg-a-test", libs=unittest_libs)
        relu_input_gradient_overfeat_fast_test_binary = \
            config.cxxld(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/overfeat-fast.cc")] + gtest_objects, "relu-input-gradient-overfeat-fast-test", libs=unittest_libs)
        config.run(relu_input_gradient_alexnet_test_binary, "relu-input-gradient-alexnet-test")
        config.run(relu_input_gradient_vgg_a_test_binary, "relu-input-gradient-vgg-a-test")
        config.run(relu_input_gradient_overfeat_fast_test_binary, "relu-input-gradient-overfeat-fast-test")
        config.phony("relu-input-gradient-test",
            ["relu-input-gradient-alexnet-test", "relu-input-gradient-vgg-a-test", "relu-input-gradient-overfeat-fast-test"])

        config.writer.default([
            fourier_reference_test_binary, fourier_x86_64_avx2_test_binary,
            winograd_x86_64_fma3_test_binary,
            convolution_output_smoke_test_binary, convolution_output_alexnet_test_binary, convolution_output_vgg_a_test_binary, convolution_output_overfeat_fast_test_binary,
            convolution_input_gradient_smoke_test_binary, convolution_input_gradient_alexnet_test_binary, convolution_input_gradient_vgg_a_test_binary, convolution_input_gradient_overfeat_fast_test_binary,
            convolution_kernel_gradient_smoke_test_binary, convolution_kernel_gradient_alexnet_test_binary, convolution_kernel_gradient_vgg_a_test_binary, convolution_kernel_gradient_overfeat_fast_test_binary,
            convolution_inference_smoke_test_binary, convolution_inference_alexnet_test_binary, convolution_inference_vgg_a_test_binary, convolution_inference_overfeat_fast_test_binary,
            fully_connected_output_smoke_test_binary, fully_connected_output_alexnet_test_binary, fully_connected_output_vgg_a_test_binary, fully_connected_output_overfeat_fast_test_binary,
            fully_connected_inference_alexnet_test_binary, fully_connected_inference_vgg_a_test_binary, fully_connected_inference_overfeat_fast_test_binary,
            pooling_output_smoke_test_binary, pooling_output_vgg_a_test_binary, pooling_output_overfeat_fast_test_binary,
            relu_output_alexnet_test_binary, relu_output_vgg_a_test_binary, relu_output_overfeat_fast_test_binary,
            relu_input_gradient_alexnet_test_binary, relu_input_gradient_vgg_a_test_binary, relu_input_gradient_overfeat_fast_test_binary])

        config.phony("test", [
            "convolution-output-test", "convolution-input-gradient-test", "convolution-kernel-gradient-test", "convolution-inference-test",
            "fully-connected-output-test", "fully-connected-inference-test",
            "pooling-output-test",
            "relu-output-test", "relu-input-gradient-test"])
        config.phony("smoketest", [
            "convolution-output-smoketest", "convolution-input-gradient-smoketest", "convolution-kernel-gradient-smoketest", "convolution-inference-smoketest",
            "fully-connected-output-smoketest",
            "pooling-output-smoketest"])

    # Build benchmarks
    config.source_dir = os.path.join(root_dir, "bench")
    config.build_dir = os.path.join(root_dir, "build", "bench")
    config.artifact_dir = os.path.join(root_dir, "bin")
    config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(pthreadpool_dir, "include"), os.path.join(root_dir, "bench")]
    bench_support_objects = [config.cc("median.c"), config.peachpy("memread.py")]
    if config.host == "x86_64-linux-gnu":
        bench_support_objects.append(config.cc("perf_counter.c"))
    extra_cflags = []
    extra_lib_dirs = []
    extra_libs = ["m"]
    extra_ldflags = []
    if options.use_mkl:
        extra_cflags.append("-DUSE_MKL")
        extra_cflags.append("-I/opt/intel/mkl/include")
        extra_lib_dirs.append("/opt/intel/mkl/lib/intel64")
        extra_libs = ["mkl_intel_lp64", "mkl_core", "mkl_sequential", "m"]
        extra_ldflags.append("-Wl,--no-as-needed")
    transform_bench_binary = config.ccld([config.cc("transform.c", extra_cflags=extra_cflags)] + nnpack_objects + bench_support_objects, "transform-bench",
        lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
    config.default(transform_bench_binary)

    convolution_bench_binary = config.ccld([config.cc("convolution.c")] + nnpack_objects + bench_support_objects,
        "convolution-benchmark", libs=["m"])
    fully_connected_bench_binary = config.ccld([config.cc("fully-connected.c")] + nnpack_objects + bench_support_objects,
        "fully-connected-benchmark", libs=["m"])
    pooling_bench_binary = config.ccld([config.cc("pooling.c")] + nnpack_objects + bench_support_objects,
        "pooling-benchmark", libs=["m"])
    relu_bench_binary = config.ccld([config.cc("relu.c")] + nnpack_objects + bench_support_objects,
        "relu-benchmark", libs=["m"])
    config.default([convolution_bench_binary, fully_connected_bench_binary, pooling_bench_binary, relu_bench_binary])

    ugemm_bench_binary = config.ccld([config.cc("ugemm.c")] + x86_64_nnpack_objects + bench_support_objects, "ugemm-bench", libs=["m"])
    if options.use_mkl:
        extra_cflags = ["-DUSE_MKL", "-I/opt/intel/mkl/include", "-pthread"]
        extra_lib_dirs = ["/opt/intel/mkl/lib/intel64"]
        extra_libs = ["mkl_intel_lp64", "mkl_core", "mkl_gnu_thread", "m", "pthread"]
        extra_ldflags = ["-Wl,--no-as-needed", "-pthread", "-fopenmp"]
        mkl_gemm_bench_binary = config.ccld([config.cc("gemm.c", "mkl-gemm.o", extra_cflags=extra_cflags)] + bench_support_objects, "mkl-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
        config.default(mkl_gemm_bench_binary)
    if options.use_openblas:
        extra_cflags = ["-DUSE_OPENBLAS", "-I/opt/OpenBLAS/include", "-pthread"]
        extra_lib_dirs = ["/opt/OpenBLAS/lib"]
        extra_libs = ["openblas", "m", "pthread"]
        extra_ldflags = []
        openblas_gemm_bench_binary = config.ccld([config.cc("gemm.c", "openblas-gemm.o", extra_cflags=extra_cflags)] + bench_support_objects, "openblas-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
        config.default(openblas_gemm_bench_binary)
    if options.use_blis:
        extra_cflags = ["-DUSE_BLIS", "-I/opt/BLIS/include", "-pthread", "-fopenmp"]
        extra_libs = ["m", "pthread"]
        extra_ldflags = ["-fopenmp"]
        blis_gemm_bench_binary = config.ccld([config.cc("gemm.c", "blis-gemm.o", extra_cflags=extra_cflags), "/opt/BLIS/lib/libblis.a"] + bench_support_objects, "blis-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
        config.default(blis_gemm_bench_binary)
    config.default(ugemm_bench_binary)

if __name__ == "__main__":
    sys.exit(main())
