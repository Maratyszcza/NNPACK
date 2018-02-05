#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include <perf_counter.h>

#include <nnpack.h>

extern unsigned long long median(unsigned long long array[], size_t length);
extern struct nnp_profile median_profile(struct nnp_profile array[], size_t length);
extern void read_memory(const void* memory, size_t length);

static void* malloc_with_alignment(size_t size, size_t alignment) {
	#if defined(__ANDROID__)
		return memalign(alignment, size);
	#else
		void* memory_block = NULL;
		if (posix_memalign(&memory_block, alignment, size) != 0) {
			return NULL;
		}

		return memory_block;
	#endif
}

enum mode {
	mode_output,
	mode_input_gradient,
	mode_kernel_gradient,
	mode_inference,
};

struct nnp_profile benchmark_convolution(
	enum mode mode,
	const void* memory, size_t cache_size,
	enum nnp_convolution_algorithm algorithm,
	enum nnp_convolution_transform_strategy transform_strategy,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size kernel_size,
	struct nnp_size output_subsampling,
	float input[],
	float kernel[],
	const float bias[],
	float output[],
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	struct nnp_profile computation_profile[max_iterations];
	enum nnp_status status = nnp_status_success;
	void* memory_block = NULL;
	void* transformed_kernel = NULL;
	size_t memory_size = 0, transformed_kernel_size = 0;
	switch (mode) {
		case mode_output:
			status = nnp_convolution_output(
				algorithm,
				batch_size, input_channels, output_channels,
				input_size, input_padding, kernel_size,
				NULL, NULL, NULL, NULL, NULL, &memory_size,
				nnp_activation_identity, NULL,
				threadpool, NULL);
			break;
		case mode_input_gradient:
			status = nnp_convolution_input_gradient(
				algorithm,
				batch_size, input_channels, output_channels,
				input_size, input_padding, kernel_size,
				NULL, NULL, NULL, NULL, &memory_size,
				nnp_activation_identity, NULL,
				threadpool, NULL);
			break;
		case mode_kernel_gradient:
			status = nnp_convolution_kernel_gradient(
				algorithm,
				batch_size, input_channels, output_channels,
				input_size, input_padding, kernel_size,
				NULL, NULL, NULL, NULL, &memory_size,
				nnp_activation_identity, NULL,
				threadpool, NULL);
			break;
		case mode_inference:
			if (transform_strategy == nnp_convolution_transform_strategy_precompute) {
				status = nnp_convolution_inference(
					algorithm, transform_strategy,
					input_channels, output_channels,
					input_size, input_padding, kernel_size, output_subsampling,
					NULL, NULL, NULL, NULL, NULL, &transformed_kernel_size,
					nnp_activation_identity, NULL,
					threadpool, NULL);
				switch (status) {
					case nnp_status_success:
						break;
					case nnp_status_invalid_algorithm:
					case nnp_status_unsupported_algorithm:
						return (struct nnp_profile) { nanf("") };
						break;
					case nnp_status_unsupported_transform_strategy:
						/* Fall back to compute strategy */
						transform_strategy = nnp_convolution_transform_strategy_compute;
						break;
					default:
						fprintf(stderr, "Error: failed to detect transformed kernel size: status %d\n", status);
						exit(EXIT_FAILURE);
				}
			}
			if (transform_strategy == nnp_convolution_transform_strategy_precompute) {
				transformed_kernel = malloc_with_alignment(transformed_kernel_size, 64);
				if (transformed_kernel == NULL) {
					fprintf(stderr, "Error: failed to allocate %zu bytes for transformed kernel\n", memory_size);
					exit(EXIT_FAILURE);
				}

				status = nnp_convolution_inference(
					algorithm, transform_strategy,
					input_channels, output_channels,
					input_size, input_padding, kernel_size, output_subsampling,
					NULL, kernel, NULL, NULL, transformed_kernel, &transformed_kernel_size,
					nnp_activation_identity, NULL,
					threadpool, NULL);
				if (status != nnp_status_success) {
					fprintf(stderr, "Error: failed to pre-compute kernel transform: status %d\n", status);
					exit(EXIT_FAILURE);
				}
				transform_strategy = nnp_convolution_transform_strategy_reuse;
			}

			status = nnp_convolution_inference(
				algorithm, transform_strategy,
				input_channels, output_channels,
				input_size, input_padding, kernel_size, output_subsampling,
				NULL, NULL, NULL, NULL, NULL, &memory_size,
				nnp_activation_identity, NULL,
				threadpool, NULL);
			break;
	}
	switch (status) {
		case nnp_status_success:
			break;
		case nnp_status_invalid_algorithm:
		case nnp_status_unsupported_algorithm:
			return (struct nnp_profile) { nanf("") };
			break;
		default:
			fprintf(stderr, "Error: failed to detect workspace memory size: status %d\n", status);
			exit(EXIT_FAILURE);
	}
	if (memory_size != 0) {
		memory_block = malloc_with_alignment(memory_size, 64);
		if (memory_block == NULL) {
			fprintf(stderr, "Error: failed to allocate %zu bytes for workspace\n", memory_size);
			exit(EXIT_FAILURE);
		}
	}

	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		read_memory(memory, cache_size);
		switch (mode) {
			case mode_output:
				nnp_convolution_output(
					algorithm,
					batch_size, input_channels, output_channels,
					input_size, input_padding, kernel_size,
					input, kernel, bias, output,
					memory_block, memory_size == 0 ? NULL : &memory_size,
					nnp_activation_identity, NULL,
					threadpool,
					&computation_profile[iteration]);
				break;
			case mode_input_gradient:
				nnp_convolution_input_gradient(
					algorithm,
					batch_size, input_channels, output_channels,
					input_size, input_padding, kernel_size,
					output, kernel, input,
					memory_block, memory_size == 0 ? NULL : &memory_size,
					nnp_activation_identity, NULL,
					threadpool,
					&computation_profile[iteration]);
				break;
			case mode_kernel_gradient:
				nnp_convolution_kernel_gradient(
					algorithm,
					batch_size, input_channels, output_channels,
					input_size, input_padding, kernel_size,
					input, output, kernel,
					memory_block, memory_size == 0 ? NULL : &memory_size,
					nnp_activation_identity, NULL,
					threadpool,
					&computation_profile[iteration]);
				break;
			case mode_inference:
				nnp_convolution_inference(
					algorithm, transform_strategy,
					input_channels, output_channels,
					input_size, input_padding, kernel_size, output_subsampling,
					input, transformed_kernel == NULL ? kernel : transformed_kernel, bias, output,
					memory_block, memory_size == 0 ? NULL : &memory_size,
					nnp_activation_identity, NULL,
					threadpool,
					&computation_profile[iteration]);
				break;
		}
	}
	free(memory_block);

	return median_profile(computation_profile, max_iterations);
}

struct options {
	enum mode mode;
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_subsampling;
	enum nnp_convolution_algorithm algorithm;
	enum nnp_convolution_transform_strategy transform_strategy;
	size_t threads;
	size_t iterations;
	bool threadpool;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -ic  --input-channels     The number of input channels\n"
"  -oc  --output-channels    The number of output channels\n"
"  -is  --input-size         Input height and width\n"
"  -ks  --kernel-size        Kernel height and width\n"
"Optional parameters:\n"
"  -m   --mode               The convolution mode (output, inference, input-gradient, kernel-gradient)\n"
"  -a   --algorithm          The algorithm (auto, ft8x8, ft16x16, wt8x8, implicit-gemm, or direct) for computing convolution (default: auto)\n"
"  -ts  --transform-strategy The transformation strategy (compute, or precompute) for kernel transformation (default: compute)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -s   --output-subsampling The size of a output subsampling region, AKA stride (default: 1x1)\n"
"  -ip  --input-padding      Implicit input padding (default: 0)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 3)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.mode = mode_inference,
		.batch_size = 1,
		.input_channels = 0,
		.output_channels = 0,
		.input_size = { 0, 0 },
		.input_padding = 0,
		.kernel_size = { 0, 0 },
		.output_subsampling = { 1, 1 },
		.algorithm = nnp_convolution_algorithm_auto,
		.transform_strategy = nnp_convolution_transform_strategy_compute,
		.threads = 0,
		.iterations = 3,
		.threadpool = true,
	};
	for (int argi = 1; argi < argc; argi += 1) {
		if ((strcmp(argv[argi], "--batch") == 0) || (strcmp(argv[argi], "-b") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected batch value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.batch_size) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.batch_size == 0) {
				fprintf(stderr, "Error: invalid value %s for the batch size: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--input-channels") == 0) || (strcmp(argv[argi], "-ic") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected input channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of input channels: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--output-channels") == 0) || (strcmp(argv[argi], "-oc") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected output channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.output_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of output channels: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--input-size") == 0) || (strcmp(argv[argi], "-is") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two input size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the input height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.input_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the input width: positive value expected\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--kernel-size") == 0) || (strcmp(argv[argi], "-ks") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two kernel size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.kernel_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.kernel_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel width: positive value expected\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--input-padding") == 0) || (strcmp(argv[argi], "-ip") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected padding value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_padding) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--output-subsampling") == 0) || (strcmp(argv[argi], "-s") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two output subsampling values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.output_subsampling.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.output_subsampling.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling width: positive value expected\n", argv[argi + 2]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--algorithm") == 0) || (strcmp(argv[argi], "-a") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected convolution algorithm name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "auto") == 0) {
				options.algorithm = nnp_convolution_algorithm_auto;
			} else if (strcmp(argv[argi + 1], "ft8x8") == 0) {
				options.algorithm = nnp_convolution_algorithm_ft8x8;
			} else if (strcmp(argv[argi + 1], "ft16x16") == 0) {
				options.algorithm = nnp_convolution_algorithm_ft16x16;
			} else if (strcmp(argv[argi + 1], "wt8x8") == 0) {
				options.algorithm = nnp_convolution_algorithm_wt8x8;
			} else if (strcmp(argv[argi + 1], "implicit-gemm") == 0) {
				options.algorithm = nnp_convolution_algorithm_implicit_gemm;
			} else if (strcmp(argv[argi + 1], "direct") == 0) {
				options.algorithm = nnp_convolution_algorithm_direct;
			} else {
				fprintf(stderr, "Error: invalid convolution algorithm name %s\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--transform-strategy") == 0) || (strcmp(argv[argi], "-ts") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected transformation strategy name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "compute") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_compute;
			} else if (strcmp(argv[argi + 1], "precompute") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_precompute;
			} else {
				fprintf(stderr, "Error: invalid trasnformation strategy name %s\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--mode") == 0) || (strcmp(argv[argi], "-m") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected convolution mode name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "output") == 0) {
				options.mode = mode_output;
			} else if (strcmp(argv[argi + 1], "input-gradient") == 0) {
				options.mode = mode_input_gradient;
			} else if (strcmp(argv[argi + 1], "kernel-gradient") == 0) {
				options.mode = mode_kernel_gradient;
			} else if (strcmp(argv[argi + 1], "inference") == 0) {
				options.mode = mode_inference;
			} else {
				fprintf(stderr, "Error: invalid value %s for the convolution mode\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--threads") == 0) || (strcmp(argv[argi], "-t") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected number of threads value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.threads) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.threads == 0) {
				options.threadpool = false;
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--iterations") == 0) || (strcmp(argv[argi], "-i") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected iterations value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.iterations) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.iterations == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of iterations: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--help") == 0) || (strcmp(argv[argi], "-h") == 0)) {
			print_options_help(argv[0]);
			exit(EXIT_SUCCESS);
		} else {
			fprintf(stderr, "Error: unknown argument '%s'\n", argv[argi]);
			print_options_help(argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if (options.mode == mode_inference && options.batch_size != 1) {
		fprintf(stderr, "Error: inference requires unit batch size\n");
		exit(EXIT_FAILURE);
	}
	if (options.transform_strategy == nnp_convolution_transform_strategy_precompute && options.mode != mode_inference) {
		fprintf(stderr, "Error: \"precompute\" transform strategy requires inference mode\n");
		exit(EXIT_FAILURE);
	}
	if (options.input_channels == 0) {
		fprintf(stderr, "Error: the number of input channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.output_channels == 0) {
		fprintf(stderr, "Error: the number of output channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.input_size.width == 0) {
		fprintf(stderr, "Error: the input size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.kernel_size.width == 0) {
		fprintf(stderr, "Error: the kernel size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

int main(int argc, char** argv) {
	enum nnp_status init_status = nnp_initialize();
	if (init_status != nnp_status_success) {
		fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
		exit(EXIT_FAILURE);
	}

	const struct options options = parse_options(argc, argv);

	const size_t batch_size = options.batch_size;
	const size_t input_channels = options.input_channels;
	const size_t output_channels = options.output_channels;
	const struct nnp_padding input_padding = { options.input_padding, options.input_padding, options.input_padding, options.input_padding };
	const struct nnp_size input_size = options.input_size;
	const struct nnp_size kernel_size = options.kernel_size;
	const struct nnp_size output_subsampling = options.output_subsampling;
	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};
	struct nnp_size tile_size;
	double flops_per_element;

	printf("Batch size: %zu\n", batch_size);
	printf("Input channels: %zu\n", input_channels);
	printf("Output channels: %zu\n", output_channels);
	printf("Input: %zux%zu with implicit padding %zu\n", input_size.height, input_size.width, options.input_padding);
	printf("Kernel: %zux%zu\n", kernel_size.height, kernel_size.width);
	printf("Subsampling: %zux%zu\n", output_subsampling.height, output_subsampling.width);
	switch (options.algorithm) {
		case nnp_convolution_algorithm_auto:
			/* To avoid failure in the next phases */
			tile_size = kernel_size;
			printf("Algorithm: auto\n");
			break;
		case nnp_convolution_algorithm_ft8x8:
			tile_size = (struct nnp_size) { 8, 8 };
			flops_per_element = 4.0;
			printf("Algorithm: FT8x8\n");
			break;
		case nnp_convolution_algorithm_ft16x16:
			tile_size = (struct nnp_size) { 16, 16 };
			flops_per_element = 4.0;
			printf("Algorithm: FT16x16\n");
			break;
		case nnp_convolution_algorithm_wt8x8:
			tile_size = (struct nnp_size) { 8, 8 };
			flops_per_element = 2.0;
			printf("Algorithm: WT8x8\n");
			break;
		case nnp_convolution_algorithm_wt8x8_fp16:
			tile_size = (struct nnp_size) { 8, 8 };
			flops_per_element = 2.0;
			printf("Algorithm: WT8x8 (FP16)\n");
			break;
		case nnp_convolution_algorithm_implicit_gemm:
			tile_size = (struct nnp_size) { 1, 1 };
			flops_per_element = 2.0 * kernel_size.height * kernel_size.width;
			printf("Algorithm: Implicit GEMM\n");
			break;
		case nnp_convolution_algorithm_direct:
			tile_size = (struct nnp_size) { 1, 1 };
			flops_per_element = 2.0 * kernel_size.height * kernel_size.width;
			printf("Algorithm: direct\n");
			break;
	}
	const struct nnp_size output_tile_size = {
		.height = tile_size.height - kernel_size.height + 1,
		.width = tile_size.width - kernel_size.width + 1
	};
	const size_t tile_count =
		(output_size.height / output_tile_size.height + !!(output_size.height % output_tile_size.height)) *
		(output_size.width / output_tile_size.width + !!(output_size.width % output_tile_size.width));

	#ifdef __ANDROID__
		const size_t cache_size = 4 * 1024 * 1024;
	#else
		const size_t cache_size = 128 * 1024 * 1024;
	#endif
	void* memory = malloc_with_alignment(cache_size, 64);
	if (memory == NULL) {
		fprintf(stderr, "Error: failed to allocate memory for cache flushing buffer\n");
		exit(EXIT_FAILURE);
	}
	void* input = malloc(batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	void* kernel = malloc(input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	void* output = malloc(batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	void* bias = malloc(output_channels * sizeof(float));

	memset(input, 0, batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	memset(kernel, 0, input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	memset(output, 0, batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	memset(bias, 0, output_channels * sizeof(float));

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	const struct nnp_profile convolution_profile =
		benchmark_convolution(
			options.mode,
			memory, cache_size,
			options.algorithm,
			options.transform_strategy,
			batch_size, input_channels, output_channels,
			input_size, input_padding, kernel_size, output_subsampling,
			input, kernel, bias, output,
			threadpool, options.iterations);
	const double convolution_time = convolution_profile.total;

	const size_t input_transform_footprint = sizeof(float) * batch_size * input_channels *
		(input_size.height * input_size.width + tile_count * tile_size.height * tile_size.width);
	const size_t kernel_transform_footprint = sizeof(float) * output_channels * input_channels *
		(kernel_size.height * kernel_size.width + tile_size.height * tile_size.width);
	const size_t output_transform_footprint = sizeof(float)* batch_size * output_channels *
		(output_size.height * output_size.width + tile_count * tile_size.height * tile_size.width);

	printf("Time: %5.3f ms\n", convolution_time * 1.0e+3);
	printf("Input transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.input_transform * 1.0e+3,
		(convolution_profile.input_transform / convolution_time) * 100.0,
		((double) input_transform_footprint) * 1.0e-9 / convolution_profile.input_transform);
	printf("Kernel transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.kernel_transform * 1.0e+3,
		(convolution_profile.kernel_transform / convolution_time) * 100.0,
		((double) kernel_transform_footprint) * 1.0e-9 / convolution_profile.kernel_transform);
	printf("Output transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.output_transform * 1.0e+3,
		(convolution_profile.output_transform / convolution_time) * 100.0,
		((double) output_transform_footprint) * 1.0e-9 / convolution_profile.output_transform);
	if (convolution_profile.block_multiplication != 0.0) {
		if (options.algorithm == nnp_convolution_algorithm_auto) {
			/* We don't know which algorithm is actually used, and thus can't compute FLOPS */
			printf("Block multiplication: %5.3f ms (%.1f%%)\n",
				convolution_profile.block_multiplication * 1.0e+3,
				(convolution_profile.block_multiplication / convolution_time) * 100.0);
		} else {
			printf("Block multiplication: %5.3f ms (%.1f%%) [%.1f GFLOPS]\n",
				convolution_profile.block_multiplication * 1.0e+3,
				(convolution_profile.block_multiplication / convolution_time) * 100.0,
				(flops_per_element * tile_size.height * tile_size.width * tile_count * batch_size * output_channels * input_channels * 1.0e-9) /
					convolution_profile.block_multiplication);
		}
	}
	const double overhead_time = convolution_profile.total -
		(convolution_profile.input_transform +
			convolution_profile.kernel_transform +
			convolution_profile.output_transform +
			convolution_profile.block_multiplication);
	printf("Overhead: %5.3f ms (%.1f%%)\n",
		overhead_time * 1.0e+3, (overhead_time / convolution_time) * 100.0);

	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
