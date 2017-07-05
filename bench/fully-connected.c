#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <perf_counter.h>

#include <nnpack.h>

extern unsigned long long median(unsigned long long array[], size_t length);
extern struct nnp_profile median_profile(struct nnp_profile array[], size_t length);
extern void read_memory(const void* memory, size_t length);

enum mode {
	mode_output,
	mode_inference,
	mode_inference_mixed,
};

struct nnp_profile benchmark_fully_connected(
	enum mode mode,
	const void* memory, size_t cache_size,
	size_t batch_size,
	size_t input_channels,
	size_t output_channels,
	const float* input,
	const void* kernel,
	float* output,
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	switch (mode) {
		case mode_inference:
		case mode_inference_mixed:
		{
			unsigned long long computation_time[max_iterations];
			size_t computation_samples = 0;
			for (size_t iteration = 0; iteration < max_iterations; iteration++) {
				read_memory(memory, cache_size);

				unsigned long long start_time, end_time;
				if (!read_timer(&start_time))
					continue;

				switch (mode) {
					case mode_inference:
						nnp_fully_connected_inference(
							input_channels,
							output_channels,
							input,
							kernel,
							output,
							threadpool);
						break;
					case mode_inference_mixed:
						nnp_fully_connected_inference_f16f32(
							input_channels,
							output_channels,
							input,
							kernel,
							output,
							threadpool);
						break;
					case mode_output:
						break;
				}

				if (!read_timer(&end_time))
					continue;

				computation_time[computation_samples++] = end_time - start_time;
			}
			unsigned long long median_computation_time = median(computation_time, max_iterations);
			return (struct nnp_profile) {
				.total = median_computation_time * 1.0e-9,
				.block_multiplication = median_computation_time * 1.0e-9
			};
			break;
		}
		case mode_output:
		{
			struct nnp_profile computation_profile[max_iterations];
			for (size_t iteration = 0; iteration < max_iterations; iteration++) {
				read_memory(memory, cache_size);
				nnp_fully_connected_output(
					batch_size,
					input_channels,
					output_channels,
					input,
					kernel,
					output,
					threadpool,
					&computation_profile[iteration]);
			}
			return median_profile(computation_profile, max_iterations);
		}
	}
}

struct options {
	enum mode mode;
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
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
"Optional parameters:\n"
"  -m   --mode               The fully connected layer mode (output, inference, inference-mixed)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 3)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.mode = mode_output,
		.batch_size = 1,
		.input_channels = 0,
		.output_channels = 0,
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
		} else if ((strcmp(argv[argi], "--mode") == 0) || (strcmp(argv[argi], "-m") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected mode name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "output") == 0) {
				options.mode = mode_output;
			} else if (strcmp(argv[argi + 1], "inference-mixed") == 0) {
				options.mode = mode_inference_mixed;
			} else if (strcmp(argv[argi + 1], "inference") == 0) {
				options.mode = mode_inference;
			} else {
				fprintf(stderr, "Error: invalid value %s for the mode\n", argv[argi + 1]);
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
	printf("Batch size: %zu\n", batch_size);
	printf("Input channels: %zu\n", input_channels);
	printf("Output channels: %zu\n", output_channels);

	#ifdef __ANDROID__
		const size_t cache_size = 4 * 1024 * 1024;
	#else
		const size_t cache_size = 128 * 1024 * 1024;
	#endif
	void* memory = NULL;
	#if defined(__ANDROID__)
		memory = memalign(64, cache_size);
		if (memory == NULL) {
			fprintf(stderr, "Error: failed to allocate memory for cache flushing buffer\n");
			exit(EXIT_FAILURE);
		}
	#else
		if (posix_memalign(&memory, 64, cache_size) != 0) {
			fprintf(stderr, "Error: failed to allocate memory for cache flushing buffer\n");
			exit(EXIT_FAILURE);
		}
	#endif

	void* input = malloc(batch_size * input_channels * sizeof(float));
	void* kernel = malloc(input_channels * output_channels * sizeof(float));
	void* output = malloc(batch_size * output_channels * sizeof(float));

	memset(input, 0, batch_size * input_channels * sizeof(float));
	memset(kernel, 0, input_channels * output_channels * sizeof(float));
	memset(output, 0, batch_size * output_channels * sizeof(float));

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	const struct nnp_profile output_profile =
		benchmark_fully_connected(
			options.mode,
			memory, cache_size,
			batch_size, input_channels, output_channels,
			input, kernel, output,
			threadpool, options.iterations);

	printf("Time: %5.3f ms [%.1f GFLOPS]\n", output_profile.total * 1.0e+3,
		(2.0 * batch_size * output_channels * input_channels * 1.0e-9) / output_profile.total);
	printf("Input packing: %5.3f ms (%.1f%%)\n",
		output_profile.input_transform * 1.0e+3,
		(output_profile.input_transform / output_profile.total) * 100.0);
	printf("Kernel packing: %5.3f ms (%.1f%%)\n",
		output_profile.kernel_transform * 1.0e+3,
		(output_profile.kernel_transform / output_profile.total) * 100.0);
	printf("Block multiplication: %5.3f ms (%.1f%%) [%.1f GFLOPS]\n",
		output_profile.block_multiplication * 1.0e+3,
		(output_profile.block_multiplication / output_profile.total) * 100.0,
		(2.0 * batch_size * output_channels * input_channels * 1.0e-9) / output_profile.block_multiplication);
	const double overhead = output_profile.total - output_profile.input_transform - output_profile.kernel_transform - output_profile.block_multiplication;
	printf("Overhead: %5.3f ms (%.1f%%)\n",
		overhead * 1.0e+3,
		(overhead / output_profile.total) * 100.0);

	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
