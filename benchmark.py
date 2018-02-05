#!/usr/bin/env python

from __future__ import print_function


def extract_time(line, prefix):
	if line.startswith(prefix):
		line = line[len(prefix):].lstrip()
		line = line[:line.index(" ms")].rstrip()
		return line


def convolution(mode, batch_size, input_channels, output_channels, image_size, kernel_size, padding, algorithm, transform_strategy=None, threads=None, verbose=False, use_selldr=False):
	import subprocess
	if use_selldr:
		import os
		import sys
		nacl_sdk_dir = os.getenv("NACL_SDK_ROOT")
		if nacl_sdk_dir is None:
			print("Error: can not find Native Client SDK: set NACL_SDK_ROOT envorinment variable and try again", file=sys.stderr)
			sys.exit(1)
		benchmark_args = [os.path.join(nacl_sdk_dir, "tools", "sel_ldr.py"), "--",
			"bin/convolution-benchmark"]
	else:
		benchmark_args = ["bin/convolution-benchmark"]
	benchmark_args += [
		"-m", mode,
		"-b", str(batch_size),
		"-ic", str(input_channels),
		"-oc", str(output_channels),
		"-is", str(image_size[0]), str(image_size[1]),
		"-ip", str(padding),
		"-ks", str(kernel_size[0]), str(kernel_size[1]),
		"-a", algorithm
	]
	if mode == "inference" and transform_strategy is not None:
		benchmark_args += ["-ts", transform_strategy]
	if threads is not None:
		benchmark_args += ["-t", str(threads)]
	benchmark = subprocess.Popen(benchmark_args, stdout=subprocess.PIPE)
	benchmark_stdout, _ = benchmark.communicate()
	if benchmark.returncode == 0:
		output_lines = [line for line in benchmark_stdout.splitlines() if len(line)]
		total, input_transform, kernel_transform, output_transform, block_multiplication, overhead = None, None, None, None, None, None
		for output_line in output_lines:
			total = total or extract_time(output_line, "Time:")
			input_transform = input_transform or extract_time(output_line, "Input transform:")
			kernel_transform = kernel_transform or extract_time(output_line, "Kernel transform:")
			output_transform = output_transform or extract_time(output_line, "Output transform:")
			block_multiplication = block_multiplication or extract_time(output_line, "Block multiplication:")
			overhead = overhead or extract_time(output_line, "Overhead:")
		if verbose:
			return (total, input_transform, kernel_transform, output_transform, block_multiplication, overhead)
		else:
			return (total,)

def fully_connected(mode, batch_size, input_channels, output_channels, threads=None, verbose=False, use_selldr=False):
	import subprocess
	if use_selldr:
		import os
		import sys
		nacl_sdk_dir = os.getenv("NACL_SDK_ROOT")
		if nacl_sdk_dir is None:
			print("Error: can not find Native Client SDK: set NACL_SDK_ROOT envorinment variable and try again", file=sys.stderr)
			sys.exit(1)
		benchmark_args = [os.path.join(nacl_sdk_dir, "tools", "sel_ldr.py"), "--",
			"bin/fully-connected-benchmark"]
	else:
		benchmark_args = ["bin/fully-connected-benchmark"]
	benchmark_args += [
		"-m", mode,
		"-b", str(batch_size),
		"-ic", str(input_channels),
		"-oc", str(output_channels)
	]
	if threads is not None:
		benchmark_args += ["-t", str(threads)]
	benchmark = subprocess.Popen(benchmark_args, stdout=subprocess.PIPE)
	benchmark_stdout, _ = benchmark.communicate()
	if benchmark.returncode == 0:
		output_lines = [line for line in benchmark_stdout.splitlines() if len(line)]
		total, input_transform, kernel_transform, block_multiplication, overhead = None, None, None, None, None
		for output_line in output_lines:
			total = total or extract_time(output_line, "Time:")
			input_transform = input_transform or extract_time(output_line, "Input packing:")
			kernel_transform = kernel_transform or extract_time(output_line, "Kernel packing:")
			block_multiplication = block_multiplication or extract_time(output_line, "Block multiplication:")
			overhead = overhead or extract_time(output_line, "Overhead:")
		if verbose:
			return (total, input_transform, kernel_transform, block_multiplication, overhead)
		else:
			return (total,)

overfeat_fast_layers = [
	("conv2",   96,  256, (24, 24), (5, 5), 0),
	("conv3",  256,  512, (12, 12), (3, 3), 1),
	("conv4",  512, 1024, (12, 12), (3, 3), 1),
	("conv5", 1024, 1024, (12, 12), (3, 3), 1),
	("fc6", 36864, 3072),
	("fc7",  3072, 4096),
	("fc8",  4096, 1000),
]

alexnet_layers = [
	("conv2",  64, 192, (27, 27), (5, 5), 2),
	("conv3", 192, 384, (13, 13), (3, 3), 1),
	("conv4", 384, 256, (13, 13), (3, 3), 1),
	("conv5", 256, 256, (13, 13), (3, 3), 1),
	("fc6", 12544, 4096),
	("fc7",  4096, 4096),
	("fc8",  4096, 1000),
]

vgg_a_layers = [
	("conv1",     3,  64, (224, 224), (3, 3), 1),
	("conv2",    64, 128, (112, 112), (3, 3), 1),
	("conv3.1", 128, 256,   (56, 56), (3, 3), 1),
	("conv3.2", 256, 256,   (56, 56), (3, 3), 1),
	("conv4.1", 256, 512,   (28, 28), (3, 3), 1),
	("conv4.2", 512, 512,   (28, 28), (3, 3), 1),
	("conv5",   512, 512,   (14, 14), (3, 3), 1),
	("fc6", 25088, 4096),
	("fc7",  4096, 4096),
	("fc8",  4096, 1000),
]
		
if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="NNPACK benchmarking script")
	parser.add_argument("--enable-selldr", dest="use_selldr", action="store_true")
	parser.add_argument("-l", "--layer", dest="layer", required=True, choices=["convolution", "fully-connected", "pooling"])
	parser.add_argument("-n", "--network", dest="network", required=True, choices=["vgg-a", "alexnet", "overfeat-fast"])
	parser.add_argument("-m", "--mode", dest="mode", required=True, choices=["inference", "output", "input-gradient", "kernel-gradient"])
	parser.add_argument("--transform-strategy", dest="transform_strategy", default="compute", choices=["compute", "precompute"])
	parser.add_argument("-b", "--batch", dest="batch", type=int)
	parser.add_argument("-t", "--threads", dest="threads")
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)

	options = parser.parse_args()

	network_layers, default_batch = {
		"vgg-a": (vgg_a_layers, 64),
		"alexnet": (alexnet_layers, 128),
		"overfeat-fast": (overfeat_fast_layers, 128)
	}[options.network]
	layer_prefix = {
		"convolution": "conv",
		"fully-connected": "fc",
		"pooling": "pool"
	}[options.layer]
	network_layers = [layer for layer in network_layers if layer[0].startswith(layer_prefix)]

	batch = default_batch
	if options.batch is not None:
		batch = options.batch
		if batch != 1 and options.mode == "inference":
			raise ValueError("Non-unit batch {batch} is not allowed in inference mode".format(batch=batch))
	elif options.mode == "inference":
		batch = 1
	if options.transform_strategy is not None:
		if options.layer != "convolution":
			raise ValueError("Transform strategy {transform_strategy} is meaningless for non-convolutional layers".format(transform_strategy=transform_strategy))
		elif options.mode != "inference":
			raise ValueError("Transform strategy {transform_strategy} is meaningless in non-inference mode".format(transform_strategy=transform_strategy))

	if options.layer == "convolution":
		for name, input_channels, output_channels, image_size, kernel_size, padding in network_layers:
			measurements = [name]
			for algorithm in ["implicit-gemm", "ft8x8", "ft16x16", "wt8x8"]:
				if algorithm.startswith("wt") and kernel_size != (3, 3):
					continue

				measurements += list(convolution(options.mode, batch, input_channels, output_channels,
					image_size, kernel_size, padding, algorithm,
					transform_strategy=options.transform_strategy,
					threads=options.threads, verbose=options.verbose, use_selldr=options.use_selldr))
			print("\t".join(map(str, measurements)))
	elif options.layer == "fully-connected":
		for name, input_channels, output_channels in network_layers:
			measurements = fully_connected(options.mode, batch, input_channels, output_channels,
				threads=options.threads, verbose=options.verbose, use_selldr=options.use_selldr)
			print("{name}\t{measurements}".format(name=name, measurements="\t".join(measurements)))
