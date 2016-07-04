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

        # Variables
        self.build, self.host = Configuration.detect_system(options.host)

        self.build_static = options.build_static
        self.build_shared = options.build_shared and self.host in ["x86_64-linux-gnu", "x86_64-osx"]
        if self.host == "pnacl-nacl-newlib":
            self.object_ext = ".bc"
            self.pic_object_ext = None
            self.executable_ext = ".pexe"
            self.static_library_ext = ".a"
            self.dynamic_library_ext = None
        elif self.host in ["x86_64-nacl-glibc", "x86_64-nacl-newlib"]:
            self.object_ext = ".o"
            self.pic_object_ext = None
            self.executable_ext = ".nexe"
            self.static_library_ext = ".a"
            self.dynamic_library_ext = None
        elif self.host == "x86_64-osx":
            self.object_ext = ".o"
            self.pic_object_ext = ".lo"
            self.executable_ext = ""
            self.static_library_ext = ".a"
            self.dynamic_library_ext = ".dylib"
        else:
            self.object_ext = ".o"
            self.pic_object_ext = ".lo"
            self.executable_ext = ""
            self.static_library_ext = ".a"
            self.dynamic_library_ext = ".so"

        cflags = ["-g", "-std=gnu11"]
        if options.use_psimd or self.host == "pnacl-nacl-newlib":
            cflags += ["-DNNP_ARCH_PSIMD"]
        cxxflags = ["-g", "-std=gnu++11"]
        ldflags = ["-g"]
        if self.host in ["x86_64-linux-gnu", "x86_64-nacl-glibc", "x86_64-nacl-newlib", "pnacl-nacl-newlib"]:
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
        elif self.host in ["x86_64-nacl-newlib", "x86_64-nacl-glibc", "pnacl-nacl-newlib"]:
            if self.host.startswith("x86_64-"):
                self.writer.variable("imageformat", "elf")
                self.writer.variable("abi", "nacl")
            self.writer.variable("nacl_sdk_dir", os.getenv("NACL_SDK_ROOT"))
            self.writer.variable("pepper_include_dir", os.path.join("$nacl_sdk_dir", "include"))

            toolchain_library_subdir_map = {
                ("x86_64-linux-gnu", "x86_64-nacl-glibc"):  ("linux_x86_glibc", "glibc_x86_64"),
                ("x86_64-linux-gnu", "x86_64-nacl-newlib"): ("linux_pnacl", "clang-newlib_x86_64"),
                ("x86_64-linux-gnu", "pnacl-nacl-newlib"):  ("linux_pnacl", "pnacl"),
                ("x86_64-osx",       "x86_64-nacl-glibc"):  ("mac_x86_glibc", "glibc_x86_64"),
                ("x86_64-osx",       "x86_64-nacl-newlib"): ("mac_pnacl", "clang-newlib_x86_64"),
                ("x86_64-osx",       "pnacl-nacl-newlib"):  ("mac_pnacl", "pnacl"),
            }
            if (self.build, self.host) in toolchain_library_subdir_map:
                toolchain_subdir, library_subdir = toolchain_library_subdir_map.get((self.build, self.host))
                self.writer.variable("nacl_toolchain_dir", os.path.join("$nacl_sdk_dir", "toolchain", toolchain_subdir))
                self.writer.variable("pepper_lib_dir", os.path.join("$nacl_sdk_dir", "lib", library_subdir, "Release"))
            else:
                print("Error: cross-compilation for %s is not supported on %s" % (self.host, self.build))
                sys.exit(1)
            toolchain_compiler_map = {
                "x86_64-nacl-glibc":  ("x86_64-nacl-gcc",   "x86_64-nacl-g++",     "x86_64-nacl-ar"),
                "x86_64-nacl-newlib": ("x86_64-nacl-clang", "x86_64-nacl-clang++", "x86_64-nacl-ar"),
                "pnacl-nacl-newlib":  ("pnacl-clang",       "pnacl-clang++",       "pnacl-ar"),
            }
            toolchain_cc, toolchain_cxx, toolchain_ar = toolchain_compiler_map[self.host]
            self.writer.variable("ar", os.path.join("$nacl_toolchain_dir", "bin", toolchain_ar))
            self.writer.variable("cc", os.path.join("$nacl_toolchain_dir", "bin", toolchain_cc))
            self.writer.variable("cxx", os.path.join("$nacl_toolchain_dir", "bin", toolchain_cxx))
            if self.host == "pnacl-nacl-newlib":
                self.writer.variable("finalize", os.path.join("$nacl_toolchain_dir", "bin", "pnacl-finalize"))
                self.writer.variable("translate", os.path.join("$nacl_toolchain_dir", "bin", "pnacl-translate"))
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
        if self.host.startswith("x86_64-"):
            self.writer.rule("peachpy", "$python2 -m peachpy.x86_64 -mabi=$abi -g4 -mimage-format=$imageformat $peachpyincludes -MMD -MF $object.d -emit-c-header $header -o $object $in",
                deps="gcc", depfile="$object.d",
                description="PEACHPY[x86-64] $descpath")
        if self.host == "pnacl-nacl-newlib":
            self.writer.rule("finalize", "$finalize --compress -o $out $in",
                description="FINALIZE $descpath")
            self.writer.rule("translate", "$translate --allow-llvm-bitcode-input -O3 -threads=auto -arch x86-64 $in -o $out",
                description="TRANSLATE $descpath")
        if self.host in ["x86_64-nacl-newlib", "x86_64-nacl-glibc", "pnacl-nacl-newlib"]:
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


    def _compile(self, rule, source_file, object_file, pic=False, header_file=None, extra_flags=None):
        if not os.path.isabs(source_file):
            source_file = os.path.join(self.source_dir, source_file)
        if object_file is None:
            object_file = os.path.join(self.build_dir, os.path.relpath(source_file, self.source_dir)) + (self.object_ext if not pic else self.pic_object_ext)
        elif not os.path.isabs(object_file):
            object_file = os.path.join(self.build_dir, object_file) + (self.object_ext if not pic else self.pic_object_ext)
        else:
            object_file = object_file + (self.object_ext if not pic else self.pic_object_ext)
        variables = {
            "descpath": os.path.relpath(source_file, self.source_dir)
        }
        if extra_flags is None:
            extra_flags = {}
        if pic:
            if "cflags" in extra_flags:
                extra_flags["cflags"] = ["-fPIC"] + extra_flags["cflags"]
            else:
                extra_flags["cflags"] = ["-fPIC"]
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
        if self.build_static:
            nonpic_object_file = self._compile("cc", source_file, object_file, extra_flags={"cflags": extra_cflags})
        else:
            nonpic_object_file = None
        if self.build_shared:
            pic_object_file = self._compile("cc", source_file, object_file, pic=True, extra_flags={"cflags": extra_cflags})
        else:
            pic_object_file = None
        return nonpic_object_file, pic_object_file


    def peachpy(self, source_file, object_file=None, header_file=None):
        object_file = self._compile("peachpy", source_file, object_file)
        return object_file, object_file


    def cxx(self, source_file, object_file=None, extra_cxxflags=[]):
        return self._compile("cxx", source_file, object_file, extra_flags={"cxxflags": extra_cxxflags}), None


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


    def ccld_executable(self, object_files, binary_file, lib_dirs=[], libs=[], extra_ldflags=[]):
        object_files = [nonpic_object for nonpic_object, pic_object in object_files]
        if self.host == "pnacl-nacl-newlib":
            bytecode_file = self._link("ccld", object_files, binary_file + ".bc", self.build_dir, lib_dirs, libs, extra_ldflags)
            portable_file = self.finalize(bytecode_file, os.path.join(self.build_dir, binary_file + ".pexe"))
            native_file = self.translate(portable_file, os.path.join(self.artifact_dir, binary_file + ".nexe"))
            return native_file
        else:
            return self._link("ccld", object_files, binary_file, self.artifact_dir, lib_dirs, libs, extra_ldflags)

    @staticmethod
    def _prepare_library_path(library_file, library_dir, library_ext):
        library_basename = os.path.basename(library_file)
        if not library_basename.startswith("lib"):
            library_basename = "lib" + library_basename
        if not library_basename.endswith(library_ext):
            library_basename = library_basename + library_ext
        library_file = os.path.join(os.path.dirname(library_file), library_basename)
        if not os.path.isabs(library_file):
            library_file = os.path.join(library_dir, library_file)
        return library_file

    def ccld_library(self, object_files, library_file, lib_dirs=[], libs=[], extra_ldflags=[]):
        assert self.dynamic_library_ext is not None

        library_file = Configuration._prepare_library_path(library_file, self.artifact_dir, self.dynamic_library_ext)

        if self.host == "x86_64-osx":
            extra_ldflags.insert(0, "-dynamiclib")
        else:
            extra_ldflags = ["-shared", "-Wl,-soname," + os.path.basename(library_file)] + extra_ldflags
        return self._link("ccld", object_files, library_file, self.artifact_dir, lib_dirs, libs, extra_ldflags)

    def module(self, object_files, module_file, lib_dirs=[], libs=[], extra_ldflags=[]):
        assert self.host in ["pnacl-nacl-newlib", "x86_64-nacl-newlib", "x86_64-nacl-glibc"]
        object_files = [nonpic_object for nonpic_object, pic_object in object_files]

        if self.host == "pnacl-nacl-newlib":
            bytecode_file = self._link("ccld", object_files, module_file + ".bc", self.build_dir, lib_dirs, libs, extra_ldflags)
            portable_file = self.finalize(bytecode_file, os.path.join(self.artifact_dir, module_file + ".pexe"))
            return portable_file
        else:
            return self._link("ccld", object_files, module_file + ".x86_64.nexe", self.artifact_dir, lib_dirs, libs, extra_ldflags)

    def unittest(self, object_files, test_name):
        object_files = [nonpic_object for nonpic_object, pic_object in object_files]
        if self.host == "pnacl-nacl-newlib":
            bytecode_file = self._link("cxxld", object_files, test_name + ".bc", self.build_dir, lib_dirs=[], libs=["m"], extra_ldflags=[])
            native_file = self.translate(bytecode_file, os.path.join(self.artifact_dir, test_name + ".nexe"))
        else:
            native_file = self._link("cxxld", object_files, test_name + self.executable_ext, self.artifact_dir, lib_dirs=[], libs=["m"], extra_ldflags=[])
        self.run(native_file, test_name)
        self.default(native_file)
        return test_name

    def translate(self, portable_file, native_file):
        assert self.host == "pnacl-nacl-newlib"

        assert os.path.isabs(portable_file)
        if not os.path.isabs(native_file):
            native_file = os.path.join(self.artifact_dir, native_file)
        variables = {
            "descpath": os.path.relpath(native_file, self.artifact_dir)
        }
        self.writer.build(native_file, "translate", portable_file, variables=variables)
        return native_file

    def finalize(self, bytecode_file, portable_file):
        assert self.host == "pnacl-nacl-newlib"

        if not os.path.isabs(portable_file):
            portable_file = os.path.join(self.build_dir, portable_file)
        if os.path.splitext(portable_file)[1] != ".pexe":
            portable_file = portable_file + ".pexe"
        variables = {
            "descpath": os.path.relpath(portable_file, self.artifact_dir)
        }
        self.writer.build(portable_file, "finalize", bytecode_file, variables=variables)
        return portable_file

    def lib(self, object_files, library_file):
        library_file = Configuration._prepare_library_path(library_file, self.artifact_dir, self.static_library_ext)
        variables = {
            "descpath": os.path.relpath(library_file, self.artifact_dir)
        }
        self.writer.build(library_file, "lib", object_files, variables=variables)
        return library_file

    def library(self, object_files, library_file):
        if self.build_static:
            static_library_file = self.lib([nonpic_object for nonpic_object, pic_object in object_files], library_file)
        else:
            static_library_file = None
        if self.build_shared:
            shared_library_file = self.ccld_library([pic_object for nonpic_object, pic_object in object_files], library_file)
        else:
            shared_library_file = None
        return static_library_file, shared_library_file

    def run(self, executable_file, target, args=""):
        variables = {
            "descpath": os.path.relpath(executable_file, self.artifact_dir)
        }
        if args:
            variables["args"] = args
        self.writer.build(target, "run", executable_file, variables=variables)

    def default(self, targets):
        if isinstance(targets, tuple):
            targets = list(targets)
        elif not isinstance(targets, list):
            targets = [targets]
        self.writer.default([ninja_syntax.escape(target).replace(":", "$:") for target in targets if target is not None])

    def variable(self, name, value):
        self.writer.variable(name, value)

    def phony(self, target, deps):
        self.writer.build(target, "phony", deps)


parser = argparse.ArgumentParser(description="NNPACK configuration script")
parser.add_argument("--host", dest="host", choices=("x86_64-linux-gnu", "x86_64-osx", "x86_64-nacl-newlib", "x86_64-nacl-glibc", "pnacl-nacl-newlib"))
parser.add_argument("--enable-mkl", dest="use_mkl", action="store_true")
parser.add_argument("--enable-openblas", dest="use_openblas", action="store_true")
parser.add_argument("--enable-blis", dest="use_blis", action="store_true")
parser.add_argument("--enable-psimd", dest="use_psimd", action="store_true")
parser.add_argument("--disable-static", dest="build_static", action="store_false", default=True)
parser.add_argument("--enable-shared", dest="build_shared", action="store_true", default=False)


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
    fxdiv_dir = os.path.join(root_dir, "third-party", "FXdiv")
    pthreadpool_dir = os.path.join(root_dir, "third-party", "pthreadpool")
    config.source_dir = os.path.join(pthreadpool_dir, "src")
    config.build_dir = os.path.join(pthreadpool_dir, "lib")
    config.include_dirs = [os.path.join(pthreadpool_dir, "include"), os.path.join(fxdiv_dir, "include")]

    pthreadpool_objects = [config.cc("pthreadpool.c")]

    # Build the library
    config.source_dir = os.path.join(root_dir, "src")
    config.build_dir = os.path.join(root_dir, "build")
    config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(root_dir, "src"), os.path.join(root_dir, "src", "ref"), os.path.join(pthreadpool_dir, "include"), os.path.join(fxdiv_dir, "include")]

    nnpack_objects = [
        config.cc("init.c"),
        config.cc("convolution-output.c"),
        config.cc("convolution-input-gradient.c"),
        config.cc("convolution-kernel.c"),
        config.cc("convolution-inference.c"),
        config.cc("fully-connected-output.c"),
        config.cc("fully-connected-inference.c"),
        config.cc("pooling-output.c"),
        config.cc("softmax-output.c"),
        config.cc("relu-output.c"),
        config.cc("relu-input-gradient.c"),
    ]

    if config.host.startswith("x86_64-") and not options.use_psimd:
        arch_nnpack_objects = [
            # Transformations
            config.peachpy("x86_64-fma/2d-fft-8x8.py"),
            config.peachpy("x86_64-fma/2d-fft-16x16.py"),
            config.peachpy("x86_64-fma/2d-wt-8x8-3x3.py"),
            # Pooling
            config.peachpy("x86_64-fma/max-pooling.py"),
            # ReLU and Softmax
            config.peachpy("x86_64-fma/relu.py"),
            config.peachpy("x86_64-fma/softmax.py"),
            config.cc("x86_64-fma/softmax.c"),
            # FFT block accumulation
            config.peachpy("x86_64-fma/fft-block-mac.py"),
            # Tuple GEMM
            config.peachpy("x86_64-fma/blas/s8gemm.py"),
            config.peachpy("x86_64-fma/blas/c8gemm.py"),
            config.peachpy("x86_64-fma/blas/s4c6gemm.py"),
            # BLAS microkernels
            config.peachpy("x86_64-fma/blas/sgemm.py"),
            config.peachpy("x86_64-fma/blas/sdotxf.py"),
        ]
    else:
        arch_nnpack_objects = [
            # Transformations
            config.cc("psimd/2d-fourier-8x8.c"),
            config.cc("psimd/2d-fourier-16x16.c"),
            config.cc("psimd/2d-wt-8x8-3x3.c"),
            # ReLU and Softmax
            config.cc("psimd/relu.c"),
            config.cc("psimd/softmax.c"),
            # FFT block accumulation
            config.cc("psimd/fft-block-mac.c"),
            # Tuple GEMM
            config.cc("psimd/blas/s4gemm.c"),
            config.cc("psimd/blas/c4gemm.c"),
            config.cc("psimd/blas/s4c2gemm.c"),
            config.cc("psimd/blas/c4gemm-conjb.c"),
            config.cc("psimd/blas/s4c2gemm-conjb.c"),
            config.cc("psimd/blas/c4gemm-conjb-transc.c"),
            config.cc("psimd/blas/s4c2gemm-conjb-transc.c"),
            # BLAS microkernels
            config.cc("psimd/blas/sgemm.c"),
            config.cc("psimd/blas/sdotxf.c"),
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

    if config.host.startswith("x86_64-") and not options.use_psimd:
        arch_fft_stub_objects = [
            config.peachpy("x86_64-fma/fft-soa.py"),
            config.peachpy("x86_64-fma/fft-aos.py"),
            config.peachpy("x86_64-fma/fft-dualreal.py"),
            config.peachpy("x86_64-fma/ifft-dualreal.py"),
            config.peachpy("x86_64-fma/fft-real.py"),
            config.peachpy("x86_64-fma/ifft-real.py"),
        ]

        arch_winograd_stub_objects = [
            config.peachpy("x86_64-fma/winograd-f6k3.py"),
        ]

        arch_math_stub_objects = [
        ]
    else:
        arch_fft_stub_objects = [
            config.cc("psimd/fft-aos.c"),
            config.cc("psimd/fft-soa.c"),
            config.cc("psimd/fft-real.c"),
            config.cc("psimd/fft-dualreal.c"),
        ]

        arch_winograd_stub_objects = [
            config.cc("psimd/winograd-f6k3.c"),
        ]

        arch_math_stub_objects = [
            config.cc("psimd/exp.c"),
        ]

    fft_objects = reference_fft_objects + arch_fft_stub_objects

    reference_blockmac_objects = [
        config.cc("ref/fft/block-mac.c"),
    ]

    nnpack_objects = nnpack_objects + arch_nnpack_objects + pthreadpool_objects

    # Build the library
    config.variable("peachpyincludes", "-I" + os.path.join(config.source_dir, "x86_64-fma"))
    config.artifact_dir = os.path.join(root_dir, "lib")
    config.default(config.library(nnpack_objects, "nnpack"))

    # Build Native Client module
    if config.host in ["x86_64-nacl-glibc", "x86_64-nacl-newlib", "pnacl-nacl-newlib"]:
        config.source_dir = os.path.join(root_dir, "src")
        config.build_dir = os.path.join(root_dir, "build")
        config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(root_dir, "src"), os.path.join(pthreadpool_dir, "include")]
        config.artifact_dir = os.path.join(root_dir, "web")

        module_source_files = ["nacl/entry.c", "nacl/instance.c", "nacl/interfaces.c", "nacl/messaging.c", "nacl/stringvars.c", "nacl/benchmark.c"]
        nacl_module_objects = [config.cc(source_file, extra_cflags="-I$pepper_include_dir") for source_file in module_source_files]
        nacl_module_binary = \
            config.module(nnpack_objects + nacl_module_objects, "nnpack", lib_dirs=["$pepper_lib_dir"], libs=["m", "ppapi"])
        config.default(nacl_module_binary)

    # Build unit tests
    if config.host not in ["x86_64-nacl-glibc"]:
        config.source_dir = os.path.join(root_dir, "test")
        config.build_dir = os.path.join(root_dir, "build", "test")
        config.artifact_dir = os.path.join(root_dir, "bin")
        config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(pthreadpool_dir, "include"), os.path.join(root_dir, "test"), os.path.join(gtest_dir, "include")]

        fourier_reference_test = \
            config.unittest(reference_fft_objects + [config.cxx("fourier/reference.cc")] + gtest_objects,
                "fourier-reference-test")

        if config.host.startswith("x86_64-") and not options.use_psimd:
            fourier_x86_64_avx2_test = \
                config.unittest(reference_fft_objects + arch_fft_stub_objects + [config.cxx("fourier/x86_64-avx2.cc")] + gtest_objects,
                    "fourier-x86_64-avx2-test")

            winograd_x86_64_fma3_test = \
                config.unittest(arch_winograd_stub_objects + nnpack_objects + [config.cxx("winograd/x86_64-fma3.cc")] + gtest_objects,
                    "winograd-x86_64-fma3-test")

            sgemm_x86_64_fma3_test = \
                config.unittest(nnpack_objects + [config.cxx("sgemm/x86_64-fma3.cc")] + gtest_objects,
                    "sgemm-x86_64-fma3-test")

        if config.host.startswith("pnacl-") or options.use_psimd:
            fourier_psimd_test = \
                config.unittest(reference_fft_objects + arch_fft_stub_objects + [config.cxx("fourier/psimd.cc")] + gtest_objects,
                    "fourier-psimd-test")

            winograd_psimd_test = \
                config.unittest(arch_winograd_stub_objects + nnpack_objects + [config.cxx("winograd/psimd.cc")] + gtest_objects,
                    "winograd-psimd-test")

            sgemm_psimd_test = \
                config.unittest(nnpack_objects + [config.cxx("sgemm/psimd.cc")] + gtest_objects,
                    "sgemm-psimd-test")

        convolution_output_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/smoke.cc")] + gtest_objects,
                "convolution-output-smoketest")
        convolution_output_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/alexnet.cc")] + gtest_objects,
                "convolution-output-alexnet-test")
        convolution_output_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/vgg-a.cc")] + gtest_objects,
                "convolution-output-vgg-a-test")
        convolution_output_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-output/overfeat-fast.cc")] + gtest_objects,
                "convolution-output-overfeat-fast-test")
        config.phony("convolution-output-test",
            [convolution_output_smoke_test, convolution_output_alexnet_test, convolution_output_vgg_a_test, convolution_output_overfeat_fast_test])

        convolution_input_gradient_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/smoke.cc")] + gtest_objects,
                "convolution-input-gradient-smoketest")
        convolution_input_gradient_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/alexnet.cc")] + gtest_objects,
                "convolution-input-gradient-alexnet-test")
        convolution_input_gradient_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/vgg-a.cc")] + gtest_objects,
                "convolution-input-gradient-vgg-a-test")
        convolution_input_gradient_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-input-gradient/overfeat-fast.cc")] + gtest_objects,
                "convolution-input-gradient-overfeat-fast-test")
        config.phony("convolution-input-gradient-test",
            [convolution_input_gradient_smoke_test, convolution_input_gradient_alexnet_test, convolution_input_gradient_vgg_a_test, convolution_input_gradient_overfeat_fast_test])

        convolution_kernel_gradient_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/smoke.cc")] + gtest_objects,
                "convolution-kernel-gradient-smoketest")
        convolution_kernel_gradient_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/alexnet.cc")] + gtest_objects,
                "convolution-kernel-gradient-alexnet-test")
        convolution_kernel_gradient_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/vgg-a.cc")] + gtest_objects,
                "convolution-kernel-gradient-vgg-a-test")
        convolution_kernel_gradient_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-kernel-gradient/overfeat-fast.cc")] + gtest_objects,
                "convolution-kernel-gradient-overfeat-fast-test")
        config.phony("convolution-kernel-gradient-test",
            [convolution_kernel_gradient_smoke_test, convolution_kernel_gradient_alexnet_test, convolution_kernel_gradient_vgg_a_test, convolution_kernel_gradient_overfeat_fast_test])

        convolution_inference_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/smoke.cc")] + gtest_objects,
                "convolution-inference-smoketest")
        convolution_inference_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/alexnet.cc")] + gtest_objects,
                "convolution-inference-alexnet-test")
        convolution_inference_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/vgg-a.cc")] + gtest_objects,
                "convolution-inference-vgg-a-test")
        convolution_inference_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("convolution-inference/overfeat-fast.cc")] + gtest_objects,
                "convolution-inference-overfeat-fast-test")
        config.phony("convolution-inference-test",
            [convolution_inference_smoke_test, convolution_inference_alexnet_test, convolution_inference_vgg_a_test, convolution_inference_overfeat_fast_test])

        fully_connected_output_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/smoke.cc")] + gtest_objects,
                "fully-connected-output-smoketest")
        fully_connected_output_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/alexnet.cc")] + gtest_objects,
                "fully-connected-output-alexnet-test")
        fully_connected_output_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/vgg-a.cc")] + gtest_objects,
                "fully-connected-output-vgg-a-test")
        fully_connected_output_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-output/overfeat-fast.cc")] + gtest_objects,
                "fully-connected-output-overfeat-fast-test")
        config.phony("fully-connected-output-test",
            [fully_connected_output_smoke_test, fully_connected_output_alexnet_test, fully_connected_output_vgg_a_test, fully_connected_output_overfeat_fast_test])

        fully_connected_inference_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/alexnet.cc")] + gtest_objects,
                "fully-connected-inference-alexnet-test")
        fully_connected_inference_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/vgg-a.cc")] + gtest_objects,
                "fully-connected-inference-vgg-a-test")
        fully_connected_inference_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("fully-connected-inference/overfeat-fast.cc")] + gtest_objects,
                "fully-connected-inference-overfeat-fast-test")
        config.phony("fully-connected-inference-test",
            [fully_connected_inference_alexnet_test, fully_connected_inference_vgg_a_test, fully_connected_inference_overfeat_fast_test])

        pooling_output_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/smoke.cc")] + gtest_objects,
                "pooling-output-smoketest")
        pooling_output_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/vgg-a.cc")] + gtest_objects,
                "pooling-output-vgg-a-test")
        pooling_output_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("pooling-output/overfeat-fast.cc")] + gtest_objects,
                "pooling-output-overfeat-fast")
        config.phony("pooling-output-test",
            [pooling_output_smoke_test, pooling_output_vgg_a_test, pooling_output_overfeat_fast_test])

        relu_output_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/alexnet.cc")] + gtest_objects,
                "relu-output-alexnet-test")
        relu_output_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/vgg-a.cc")] + gtest_objects,
                "relu-output-vgg-a-test")
        relu_output_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-output/overfeat-fast.cc")] + gtest_objects,
                "relu-output-overfeat-fast-test")
        config.phony("relu-output-test",
            [relu_output_alexnet_test, relu_output_vgg_a_test, relu_output_overfeat_fast_test])

        relu_input_gradient_alexnet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/alexnet.cc")] + gtest_objects,
                "relu-input-gradient-alexnet-test")
        relu_input_gradient_vgg_a_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/vgg-a.cc")] + gtest_objects,
                "relu-input-gradient-vgg-a-test")
        relu_input_gradient_overfeat_fast_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("relu-input-gradient/overfeat-fast.cc")] + gtest_objects,
                "relu-input-gradient-overfeat-fast-test")
        config.phony("relu-input-gradient-test",
            [relu_input_gradient_alexnet_test, relu_input_gradient_vgg_a_test, relu_input_gradient_overfeat_fast_test])

        softmax_output_smoke_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("softmax-output/smoke.cc")] + gtest_objects,
                "softmax-output-smoketest")
        softmax_output_imagenet_test = \
            config.unittest(nnpack_objects + reference_layer_objects + [config.cxx("softmax-output/imagenet.cc")] + gtest_objects,
                "softmax-output-imagenet-test")
        config.phony("softmax-output-test",
            [softmax_output_smoke_test, softmax_output_imagenet_test])

        config.phony("test", [
            "convolution-output-test", "convolution-input-gradient-test", "convolution-kernel-gradient-test", "convolution-inference-test",
            "fully-connected-output-test", "fully-connected-inference-test",
            "pooling-output-test",
            "relu-output-test", "relu-input-gradient-test",
            "softmax-output-test"])
        config.phony("smoketest", [
            convolution_output_smoke_test, convolution_input_gradient_smoke_test, convolution_kernel_gradient_smoke_test, convolution_inference_smoke_test,
            fully_connected_output_smoke_test,
            pooling_output_smoke_test,
            softmax_output_smoke_test])

    # Build benchmarks
    config.source_dir = os.path.join(root_dir, "bench")
    config.build_dir = os.path.join(root_dir, "build", "bench")
    config.artifact_dir = os.path.join(root_dir, "bin")
    config.include_dirs = [os.path.join(root_dir, "include"), os.path.join(pthreadpool_dir, "include"), os.path.join(root_dir, "bench")]
    bench_support_objects = [config.cc("median.c")]
    if config.host.startswith("x86_64-"):
        bench_support_objects.append(config.peachpy("memread.py"))

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
    transform_bench_binary = config.ccld_executable([config.cc("transform.c", extra_cflags=extra_cflags)] + nnpack_objects + bench_support_objects, "transform-bench",
        lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
    config.phony("transform-bench", transform_bench_binary)
    config.default(transform_bench_binary)

    convolution_bench_binary = config.ccld_executable([config.cc("convolution.c")] + nnpack_objects + bench_support_objects,
        "convolution-benchmark", libs=["m"])
    config.phony("convolution-bench", convolution_bench_binary)
    fully_connected_bench_binary = config.ccld_executable([config.cc("fully-connected.c")] + nnpack_objects + bench_support_objects,
        "fully-connected-benchmark", libs=["m"])
    config.phony("fully-connected-bench", fully_connected_bench_binary)
    pooling_bench_binary = config.ccld_executable([config.cc("pooling.c")] + nnpack_objects + bench_support_objects,
        "pooling-benchmark", libs=["m"])
    config.phony("pooling-bench", pooling_bench_binary)
    relu_bench_binary = config.ccld_executable([config.cc("relu.c")] + nnpack_objects + bench_support_objects,
        "relu-benchmark", libs=["m"])
    config.phony("relu-bench", relu_bench_binary)
    config.default([convolution_bench_binary, fully_connected_bench_binary, pooling_bench_binary, relu_bench_binary])

    vgg_bench_binary = config.ccld_executable([config.cc("vgg.c")] + nnpack_objects + bench_support_objects,
        "vgg-benchmark", libs=["m"])
    config.phony("vgg-bench", vgg_bench_binary)
    config.default([vgg_bench_binary])

    if config.host.startswith("x86_64-") and not options.use_psimd:
        #ugemm_bench_binary = config.ccld_executable([config.cc("ugemm.c")] + arch_nnpack_objects + bench_support_objects, "ugemm-bench", libs=["m"])
        #config.default(ugemm_bench_binary)
        if options.use_mkl:
            extra_cflags = ["-DUSE_MKL", "-I/opt/intel/mkl/include", "-pthread"]
            extra_lib_dirs = ["/opt/intel/mkl/lib/intel64"]
            extra_libs = ["mkl_intel_lp64", "mkl_core", "mkl_gnu_thread", "m", "pthread"]
            extra_ldflags = ["-Wl,--no-as-needed", "-pthread", "-fopenmp"]
            mkl_gemm_bench_binary = config.ccld_executable([config.cc("gemm.c", "mkl-gemm", extra_cflags=extra_cflags)] + bench_support_objects, "mkl-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
            config.default(mkl_gemm_bench_binary)
        if options.use_openblas:
            extra_cflags = ["-DUSE_OPENBLAS", "-I/opt/OpenBLAS/include", "-pthread"]
            extra_lib_dirs = ["/opt/OpenBLAS/lib"]
            extra_libs = ["openblas", "m", "pthread"]
            extra_ldflags = []
            openblas_gemm_bench_binary = config.ccld_executable([config.cc("gemm.c", "openblas-gemm", extra_cflags=extra_cflags)] + bench_support_objects, "openblas-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
            config.default(openblas_gemm_bench_binary)
        if options.use_blis:
            extra_cflags = ["-DUSE_BLIS", "-I/opt/BLIS/include", "-pthread", "-fopenmp"]
            extra_libs = ["m", "pthread"]
            extra_ldflags = ["-fopenmp"]
            blis_gemm_bench_binary = config.ccld_executable([config.cc("gemm.c", "blis-gemm", extra_cflags=extra_cflags), "/opt/BLIS/lib/libblis.a"] + bench_support_objects, "blis-gemm-bench", lib_dirs=extra_lib_dirs, libs=extra_libs, extra_ldflags=extra_ldflags)
            config.default(blis_gemm_bench_binary)

if __name__ == "__main__":
    sys.exit(main())
