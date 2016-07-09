#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include <perf_counter.h>

#include <nnpack.h>

extern unsigned long long median(unsigned long long array[], size_t length);
extern void read_memory(const void* memory, size_t length);

enum mode {
	mode_output,
};

struct convolutional_layer {
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size kernel_size;
	float* kernel;
	float* bias;
};

struct fully_connected_layer {
	size_t input_channels;
	size_t output_channels;
	float* kernel;
};

struct relu_layer {
	size_t channels;
	float negative_slope;
};

struct max_pooling_layer {
	size_t channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size pooling_size;
	struct nnp_size pooling_stride;
};

struct softmax_layer {
	size_t channels;
};

enum layer_type {
	layer_type_convolutional,
	layer_type_fully_connected,
	layer_type_relu,
	layer_type_max_pooling,
	layer_type_softmax,
};

struct layer {
	enum layer_type type;
	float* input;
	float* output;
	union {
		struct convolutional_layer convolutional_layer;
		struct fully_connected_layer fully_connected_layer;
		struct relu_layer relu_layer;
		struct max_pooling_layer max_pooling_layer;
		struct softmax_layer softmax_layer;
	};
};

double benchmark_vgg(
	enum mode mode,
	size_t batch_size,
	const float* kernel,
	const float* bias,
	size_t layers_count,
	struct layer layers[],
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	unsigned long long computation_time[max_iterations];
	size_t computation_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		unsigned long long start_time, end_time;
		if (!read_timer(&start_time))
			continue;

		enum nnp_status status;
		switch (mode) {
			case mode_output:
				for (size_t layer_index = 0; layer_index < layers_count; layer_index++) {
					switch (layers[layer_index].type) {
						case layer_type_convolutional:
							status = nnp_convolution_output(nnp_convolution_algorithm_auto,
								batch_size,
								layers[layer_index].convolutional_layer.input_channels,
								layers[layer_index].convolutional_layer.output_channels,
								layers[layer_index].convolutional_layer.input_size,
								layers[layer_index].convolutional_layer.input_padding,
								layers[layer_index].convolutional_layer.kernel_size,
								layers[layer_index].input,
								layers[layer_index].convolutional_layer.kernel,
								layers[layer_index].convolutional_layer.bias,
								layers[layer_index].output,
								threadpool, NULL);
							break;
						case layer_type_fully_connected:
							status = nnp_fully_connected_output(
								batch_size,
								layers[layer_index].fully_connected_layer.input_channels,
								layers[layer_index].fully_connected_layer.output_channels,
								layers[layer_index].input,
								layers[layer_index].fully_connected_layer.kernel,
								layers[layer_index].output,
								threadpool, NULL);
							break;
						case layer_type_relu:
							status = nnp_relu_output(
								batch_size,
								layers[layer_index].relu_layer.channels,
								layers[layer_index].input,
								layers[layer_index].output,
								layers[layer_index].relu_layer.negative_slope,
								threadpool);
							break;
						case layer_type_max_pooling:
							status = nnp_max_pooling_output(
								batch_size,
								layers[layer_index].max_pooling_layer.channels,
								layers[layer_index].max_pooling_layer.input_size,
								layers[layer_index].max_pooling_layer.input_padding,
								layers[layer_index].max_pooling_layer.pooling_size,
								layers[layer_index].max_pooling_layer.pooling_stride,
								layers[layer_index].input,
								layers[layer_index].output,
								threadpool);
							break;
						case layer_type_softmax:
							status = nnp_softmax_output(
								batch_size,
								layers[layer_index].softmax_layer.channels,
								layers[layer_index].input,
								layers[layer_index].output,
								threadpool);
							break;
					}
				}
				break;
		}
		assert(status == nnp_status_success);

		if (!read_timer(&end_time))
			continue;

		computation_time[computation_samples++] = end_time - start_time;
	}
	return median(computation_time, max_iterations);
}

struct options {
	enum mode mode;
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size kernel_size;
	enum nnp_convolution_algorithm algorithm;
	enum nnp_convolution_transform_strategy transform_strategy;
	size_t threads;
	size_t iterations;
	bool threadpool;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Optional parameters:\n"
"  -m   --mode               The convolution mode (output, inference)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 3)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.mode = mode_output,
		.batch_size = 1,
		.threads = 0,
		.iterations = 1,
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
		} else if ((strcmp(argv[argi], "--mode") == 0) || (strcmp(argv[argi], "-m") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected convolution mode name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "output") == 0) {
				options.mode = mode_output;
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

	float* data    = malloc(batch_size *   3 * 224 * 224 * sizeof(float));
	/* CONV 1.1 + ReLU */
	float* conv1_1 = malloc(batch_size *  64 * 224 * 224 * sizeof(float));
	/* CONV 1.2 + ReLU */
	float* conv1_2 = malloc(batch_size *  64 * 224 * 224 * sizeof(float));
	/* POOL 1 */
	float* pool1   = malloc(batch_size *  64 * 112 * 112 * sizeof(float));
	/* CONV 2.1 + ReLU */
	float* conv2_1 = malloc(batch_size * 128 * 112 * 112 * sizeof(float));
	/* CONV 2.2 + ReLU */
	float* conv2_2 = malloc(batch_size * 128 * 112 * 112 * sizeof(float));
	/* POOL 2 */
	float* pool2   = malloc(batch_size * 128 *  56 *  56 * sizeof(float));
	/* CONV 3.1 + ReLU */
	float* conv3_1 = malloc(batch_size * 256 *  56 *  56 * sizeof(float));
	/* CONV 3.2 + ReLU */
	float* conv3_2 = malloc(batch_size * 256 *  56 *  56 * sizeof(float));
	/* CONV 3.3 + ReLU */
	float* conv3_3 = malloc(batch_size * 256 *  56 *  56 * sizeof(float));
	/* POOL 3 */
	float* pool3   = malloc(batch_size * 256 *  28 *  28 * sizeof(float));
	/* CONV 4.1 + ReLU */
	float* conv4_1 = malloc(batch_size * 512 *  28 *  28 * sizeof(float));
	/* CONV 4.2 + ReLU */
	float* conv4_2 = malloc(batch_size * 512 *  28 *  28 * sizeof(float));
	/* CONV 4.3 + ReLU */
	float* conv4_3 = malloc(batch_size * 512 *  28 *  28 * sizeof(float));
	/* POOL 4 */
	float* pool4   = malloc(batch_size * 512 *  14 *  14 * sizeof(float));
	/* CONV 5.1 + ReLU */
	float* conv5_1 = malloc(batch_size * 512 *  14 *  14 * sizeof(float));
	/* CONV 5.2 + ReLU */
	float* conv5_2 = malloc(batch_size * 512 *  14 *  14 * sizeof(float));
	/* CONV 5.3 + ReLU */
	float* conv5_3 = malloc(batch_size * 512 *  14 *  14 * sizeof(float));
	/* POOL 5 */
	float* pool5   = malloc(batch_size * 512 *   7 *   7 * sizeof(float));
	/* FC 6 + ReLU */
	float* fc6     = malloc(batch_size * 25088 * sizeof(float));
	/* FC 7 + ReLU */
	float* fc7     = malloc(batch_size *  4096 * sizeof(float));
	/* FC 8 + ReLU */
	float* fc8     = malloc(batch_size *  1000 * sizeof(float));
	/* Softmax */
	float* prob    = malloc(batch_size *  1000 * sizeof(float));

	struct layer vgg16[] = {
		{
			.type = layer_type_convolutional,
			.input = data,
			.output = conv1_1,
			.convolutional_layer = {
				.input_channels  = 3,
				.output_channels = 64,
				.input_size = { 224, 224 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv1_1,
			.output = conv1_1,
		},
		{
			.type = layer_type_convolutional,
			.input = conv1_1,
			.output = conv1_2,
			.convolutional_layer = {
				.input_channels  = 64,
				.output_channels = 64,
				.input_size = { 224, 224 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv1_2,
			.output = conv1_2,
		},
		{
			.type = layer_type_max_pooling,
			.input = conv1_2,
			.output = pool1,
			.max_pooling_layer = {
				.channels = 64,
				.input_size = { 224, 224 },
				.pooling_size = { 2, 2 },
				.pooling_stride = { 2, 2 },
			},
		},
		{
			.type = layer_type_convolutional,
			.input = pool1,
			.output = conv2_1,
			.convolutional_layer = {
				.input_channels  = 64,
				.output_channels = 128,
				.input_size = { 112, 112 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv2_1,
			.output = conv2_1,
		},
		{
			.type = layer_type_convolutional,
			.input = conv2_1,
			.output = conv2_2,
			.convolutional_layer = {
				.input_channels  = 128,
				.output_channels = 128,
				.input_size = { 112, 112 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv2_2,
			.output = conv2_2,
		},
		{
			.type = layer_type_max_pooling,
			.input = conv2_2,
			.output = pool2,
			.max_pooling_layer = {
				.channels = 128,
				.input_size = { 112, 112 },
				.pooling_size = { 2, 2 },
				.pooling_stride = { 2, 2 },
			},
		},
		{
			.type = layer_type_convolutional,
			.input = pool2,
			.output = conv3_1,
			.convolutional_layer = {
				.input_channels  = 128,
				.output_channels = 256,
				.input_size = { 56, 56 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv3_1,
			.output = conv3_1,
		},
		{
			.type = layer_type_convolutional,
			.input = conv3_1,
			.output = conv3_2,
			.convolutional_layer = {
				.input_channels  = 256,
				.output_channels = 256,
				.input_size = { 56, 56 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv3_2,
			.output = conv3_2,
		},
		{
			.type = layer_type_convolutional,
			.input = conv3_2,
			.output = conv3_3,
			.convolutional_layer = {
				.input_channels  = 256,
				.output_channels = 256,
				.input_size = { 56, 56 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv3_3,
			.output = conv3_3,
		},
		{
			.type = layer_type_max_pooling,
			.input = conv3_3,
			.output = pool3,
			.max_pooling_layer = {
				.channels = 256,
				.input_size = { 56, 56 },
				.pooling_size = { 2, 2 },
				.pooling_stride = { 2, 2 },
			},
		},
		{
			.type = layer_type_convolutional,
			.input = pool3,
			.output = conv4_1,
			.convolutional_layer = {
				.input_channels  = 256,
				.output_channels = 512,
				.input_size = { 28, 28 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv4_1,
			.output = conv4_1,
		},
		{
			.type = layer_type_convolutional,
			.input = conv4_1,
			.output = conv4_2,
			.convolutional_layer = {
				.input_channels  = 512,
				.output_channels = 512,
				.input_size = { 28, 28 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv4_2,
			.output = conv4_2,
		},
		{
			.type = layer_type_convolutional,
			.input = conv4_2,
			.output = conv4_3,
			.convolutional_layer = {
				.input_channels  = 512,
				.output_channels = 512,
				.input_size = { 28, 28 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv4_3,
			.output = conv4_3,
		},
		{
			.type = layer_type_max_pooling,
			.input = conv4_3,
			.output = pool4,
			.max_pooling_layer = {
				.channels = 512,
				.input_size = { 28, 28 },
				.pooling_size = { 2, 2 },
				.pooling_stride = { 2, 2 },
			},
		},
		{
			.type = layer_type_convolutional,
			.input = pool4,
			.output = conv5_1,
			.convolutional_layer = {
				.input_channels  = 512,
				.output_channels = 512,
				.input_size = { 14, 14 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv5_1,
			.output = conv5_1,
		},
		{
			.type = layer_type_convolutional,
			.input = conv5_1,
			.output = conv5_2,
			.convolutional_layer = {
				.input_channels  = 512,
				.output_channels = 512,
				.input_size = { 14, 14 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv5_2,
			.output = conv5_2,
		},
		{
			.type = layer_type_convolutional,
			.input = conv5_2,
			.output = conv5_3,
			.convolutional_layer = {
				.input_channels  = 512,
				.output_channels = 512,
				.input_size = { 14, 14 },
				.input_padding = { 1, 1, 1, 1 },
				.kernel_size = { 3, 3 },
			},
		},
		{
			.type = layer_type_relu,
			.input = conv5_3,
			.output = conv5_3,
		},
		{
			.type = layer_type_max_pooling,
			.input = conv5_3,
			.output = pool5,
			.max_pooling_layer = {
				.channels = 512,
				.input_size = { 14, 14 },
				.pooling_size = { 2, 2 },
				.pooling_stride = { 2, 2 },
			},
		},
		{
			.type = layer_type_fully_connected,
			.input = pool5,
			.output = fc6,
			.fully_connected_layer = {
				.input_channels = 512 * 7 * 7,
				.output_channels = 4096,
			},
		},
		{
			.type = layer_type_relu,
			.input = fc6,
			.output = fc6,
		},
		{
			.type = layer_type_fully_connected,
			.input = fc6,
			.output = fc7,
			.fully_connected_layer = {
				.input_channels = 4096,
				.output_channels = 4096,
			},
		},
		{
			.type = layer_type_relu,
			.input = fc7,
			.output = fc7,
		},
		{
			.type = layer_type_fully_connected,
			.input = fc7,
			.output = fc8,
			.fully_connected_layer = {
				.input_channels = 4096,
				.output_channels = 1000,
			},
		},
		{
			.type = layer_type_relu,
			.input = fc8,
			.output = fc8,
		},
		{
			.type = layer_type_softmax,
			.input = fc8,
			.output = prob,
			.softmax_layer = {
				.channels = 1000,
			},
		},
	};

	const size_t vgg16_layers = sizeof(vgg16) / sizeof(vgg16[0]);
	for (size_t i = 0; i < vgg16_layers; i++) {
		switch (vgg16[i].type) {
			case layer_type_convolutional:
				vgg16[i].convolutional_layer.kernel = malloc(
					vgg16[i].convolutional_layer.output_channels * vgg16[i].convolutional_layer.input_channels *
					vgg16[i].convolutional_layer.kernel_size.height * vgg16[i].convolutional_layer.kernel_size.width *
					sizeof(float));
				vgg16[i].convolutional_layer.bias = malloc(
					vgg16[i].convolutional_layer.output_channels * sizeof(float));
				break;
			case layer_type_fully_connected:
				vgg16[i].fully_connected_layer.kernel = malloc(
					vgg16[i].fully_connected_layer.output_channels * vgg16[i].fully_connected_layer.input_channels * sizeof(float));
				break;
			case layer_type_relu:
			case layer_type_softmax:
			case layer_type_max_pooling:
				break;
		}
	}

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	const float* kernel = malloc(batch_size * 512 * 512 * sizeof(float));
	const float* bias = malloc(512 * sizeof(float));
	const unsigned long long vgg_nanoseconds = benchmark_vgg(
		options.mode,
		options.batch_size,
		kernel, bias,
		sizeof(vgg16) / sizeof(vgg16[0]), vgg16,
		threadpool, options.iterations);
	printf("Time: %5.3f ms\n", ((double) vgg_nanoseconds) * 1.0e-6);
	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
