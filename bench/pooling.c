#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <perf_counter.h>

#include <nnpack.h>
#include <nnpack/utils.h>

extern unsigned long long median(unsigned long long array[], size_t length);
extern void read_memory(const void* memory, size_t length);

unsigned long long benchmark_pooling(
	const void* memory, size_t cache_size,
	size_t batch_size,
	size_t channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size pooling_size,
	struct nnp_size pooling_stride,
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

		nnp_max_pooling_output(
			batch_size,
			channels,
			input_size,
			input_padding,
			pooling_size,
			pooling_stride,
			input,
			output,
			threadpool);

		if (!read_timer(&end_time))
			continue;

		computation_time[computation_samples++] = end_time - start_time;
	}
	return median(computation_time, max_iterations);
}

struct options {
	size_t batch_size;
	size_t channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size pooling_size;
	struct nnp_size pooling_stride;
	size_t threads;
	size_t iterations;
	bool threadpool;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -c   --channels           The number of channels\n"
"  -is  --input-size         Input height and width\n"
"Optional parameters:\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -ip  --input-padding      Implicit input padding (default: 0)\n"
"  -ps  --pooling-size       Vertical and horizontal pooling size (default: 2x2)\n"
"  -pt  --pooling-stride     Vertical and horizontal pooling stride (default: 2x2)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 15)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.batch_size = 1,
		.channels = 0,
		.input_size = { 0, 0 },
		.input_padding = 0,
		.pooling_size = { 2, 2 },
		.pooling_stride = { 2, 2 },
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
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the input width: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--input-padding") == 0) || (strcmp(argv[argi], "-ip") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected input padding value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_padding) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_padding == 0) {
				fprintf(stderr, "Error: invalid value %s for the input padding: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--pooling-size") == 0) || (strcmp(argv[argi], "-ps") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two pooling size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.pooling_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.pooling_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the pooling height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.pooling_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.pooling_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel width: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--pooling-stride") == 0) || (strcmp(argv[argi], "-pt") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two pooling stride values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.pooling_stride.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.pooling_stride.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the vertical pooling stride: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.pooling_stride.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.pooling_stride.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the horizontal pooling stride: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
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
	if (options.input_size.width == 0) {
		fprintf(stderr, "Error: the input size is not specified\n");
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
	const size_t channels = options.channels;
	const struct nnp_size input_size = options.input_size;
	const struct nnp_padding input_padding = { options.input_padding, options.input_padding, options.input_padding, options.input_padding };
	const struct nnp_size pooling_size = options.pooling_size;
	const struct nnp_size pooling_stride = options.pooling_stride;
	const struct nnp_size output_size = {
		.height = divide_round_up(input_padding.top + input_size.height + input_padding.bottom, pooling_stride.height),
		.width = divide_round_up(input_padding.left + input_size.width + input_padding.right, pooling_stride.width),
	};

	printf("Batch size: %zu\n", batch_size);
	printf("Channels: %zu\n", channels);
	printf("Input: %zux%zu with implicit padding %zu\n", input_size.height, input_size.width, options.input_padding);
	printf("Pooling: %zux%zu with %zux%zu stride\n",
		pooling_size.height, pooling_size.width, pooling_stride.height, pooling_stride.width);

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

	const size_t input_bytes = batch_size * channels * input_size.width * input_size.height * sizeof(float);
	const size_t output_bytes = batch_size * channels * output_size.width * output_size.height * sizeof(float);
	void* input = malloc(input_bytes);
	void* output = malloc(output_bytes);
	memset(input, 0, input_bytes);
	memset(output, 0, output_bytes);

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	const unsigned long long pooling_output_nanoseconds =
		benchmark_pooling(
			memory, cache_size,
			batch_size, channels,
			input_size, input_padding, pooling_size, pooling_stride,
			input, output,
			threadpool, options.iterations);

	printf("Time: %5.3f ms [%.1f GB/s]\n",
		((double) pooling_output_nanoseconds) * 1.0e-6,
		((double) (input_bytes + output_bytes)) / ((double) pooling_output_nanoseconds));
	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
