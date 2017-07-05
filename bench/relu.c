#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include <perf_counter.h>

#include <nnpack.h>
#include <nnpack/utils.h>

extern unsigned long long median(unsigned long long array[], size_t length);
extern void read_memory(const void* memory, size_t length);

enum mode {
	mode_output,
	mode_output_inplace,
	mode_input_gradient,
};

unsigned long long benchmark_relu(
	enum mode mode,
	const void* memory, size_t cache_size,
	size_t batch_size, size_t channels,
	const float gradient[],
	const float input[],
	float output[],
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	unsigned long long computation_time[max_iterations];
	size_t computation_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		read_memory(memory, cache_size);

		unsigned long long start_time, end_time;
		if (!read_timer(&start_time))
			continue;

		switch (mode) {
			case mode_output:
				nnp_relu_output(
					batch_size, channels,
					input, output,
					0.0f,
					threadpool);
				break;
			case mode_output_inplace:
				nnp_relu_output(
					batch_size, channels,
					output, output,
					0.0f,
					threadpool);
				break;
			case mode_input_gradient:
				nnp_relu_input_gradient(
					batch_size, channels,
					gradient, input, output,
					0.0f,
					threadpool);
				break;
		}

		if (!read_timer(&end_time))
			continue;

		computation_time[computation_samples++] = end_time - start_time;
	}
	return median(computation_time, max_iterations);
}

struct options {
	enum mode mode;
	size_t batch_size;
	size_t channels;
	size_t threads;
	size_t iterations;
	bool threadpool;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -c   --channels           The number of channels\n"
"Optional parameters:\n"
"  -m   --mode               The fully connected layer mode (output, output-inplace, input-gradient)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 15)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.mode = mode_output,
		.batch_size = 1,
		.channels = 0,
		.threads = 0,
		.iterations = 15,
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
		} else if ((strcmp(argv[argi], "--channels") == 0) || (strcmp(argv[argi], "-c") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of channels: positive value expected\n", argv[argi + 1]);
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
			} else if (strcmp(argv[argi + 1], "output-inplace") == 0) {
				options.mode = mode_output_inplace;
			} else if (strcmp(argv[argi + 1], "input-gradient") == 0) {
				options.mode = mode_input_gradient;
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
	if (options.channels == 0) {
		fprintf(stderr, "Error: the number of channels is not specified\n");
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

	printf("Batch size: %zu\n", options.batch_size);
	printf("Channels: %zu\n", options.channels);

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

	const size_t layer_bytes = options.batch_size * options.channels * sizeof(float);
	void* gradient = NULL;
	void* input = NULL;
	void* output = malloc(layer_bytes);
	memset(output, 0, layer_bytes);
	if (options.mode == mode_input_gradient) {
		gradient = malloc(layer_bytes);
		memset(gradient, 0, layer_bytes);
	}
	if (options.mode != mode_output_inplace) {
		input = malloc(layer_bytes);
		memset(input, 0, layer_bytes);
	}

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	const unsigned long long relu_nanoseconds = benchmark_relu(
		options.mode,
		memory, cache_size,
		options.batch_size, options.channels,
		gradient, input, output,
		threadpool, options.iterations);

	const double transferred_bytes =
		(options.mode == mode_input_gradient) ?
			3.0 * layer_bytes : 2.0 * layer_bytes;
	printf("Time: %5.3f ms [%.1f GB/s]\n",
		((double) relu_nanoseconds) * 1.0e-6,
		transferred_bytes / ((double) relu_nanoseconds));
	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
