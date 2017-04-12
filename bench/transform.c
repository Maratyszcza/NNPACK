#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include <perf_counter.h>

#include <nnpack/hwinfo.h>
#include <nnpack/macros.h>
#include <pthreadpool.h>

#ifdef USE_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

enum transform_type {
	transform_type_unknown,
	transform_type_nnpack_forward_8x8,
	transform_type_nnpack_forward_16x16,
	transform_type_nnpack_inverse_8x8,
	transform_type_nnpack_inverse_16x16,
#ifdef USE_MKL
	transform_type_mkl_perm_forward_8x8,
	transform_type_mkl_perm_forward_16x16,
	transform_type_mkl_perm_inverse_8x8,
	transform_type_mkl_perm_inverse_16x16,
#endif
	transform_type_nnpack_winograd6x6of3x3_input,
	transform_type_nnpack_winograd6x6of3x3_kernel,
	transform_type_nnpack_winograd6x6of3x3_output,
};

extern unsigned long long median(unsigned long long array[], size_t length);

struct nnpack_context {
	enum transform_type type;
	nnp_transform_2d_with_offset fft_function;
	nnp_transform_2d_with_offset ifft_function;
	nnp_transform_2d_with_offset winograd_input_transform_function;
	nnp_transform_2d_with_offset winograd_kernel_transform_function;
	nnp_transform_2d_with_offset winograd_output_transform_function;
	const float* input;
	float* output;
	size_t transform_size;
	size_t input_elements;
	size_t output_elements;
};

static void compute_nnpack_transform(struct nnpack_context* context, size_t index) {
	switch (context->type) {
		case transform_type_nnpack_forward_8x8:
		case transform_type_nnpack_forward_16x16:
			context->fft_function(
				context->input + index * context->input_elements,
				context->output + index * context->output_elements,
				context->transform_size, 64,
				context->transform_size, context->transform_size, 0, 0);
			break;
		case transform_type_nnpack_inverse_8x8:
		case transform_type_nnpack_inverse_16x16:
			context->ifft_function(
				context->input + index * context->input_elements,
				context->output + index * context->output_elements,
				64, context->transform_size,
				context->transform_size, context->transform_size, 0, 0);
			break;
		case transform_type_nnpack_winograd6x6of3x3_input:
			context->winograd_input_transform_function(
				context->input + index * context->input_elements,
				context->output + index * context->output_elements,
				context->transform_size, 32,
				context->transform_size, context->transform_size, 0, 0);
			break;
		case transform_type_nnpack_winograd6x6of3x3_kernel:
			context->winograd_kernel_transform_function(
				context->input + index * context->input_elements,
				context->output + index * context->output_elements,
				3,
				32,
				3, 3, 0, 0);
			break;
		case transform_type_nnpack_winograd6x6of3x3_output:
			context->winograd_output_transform_function(
				context->input + index * context->input_elements,
				context->output + index * context->output_elements,
				32,
				6,
				6, 6, 0, 0);
			break;
		default:
			NNP_UNREACHABLE;
	}
}

#ifdef USE_MKL
struct mkl_fft_context {
	DFTI_DESCRIPTOR_HANDLE descriptor;
	bool forward_fft;
	const float* input;
	float* output;
	size_t input_elements;
	size_t output_elements;
};

static void compute_mkl_fft(struct mkl_fft_context* context, size_t index) {
	if (context->forward_fft) {
		MKL_LONG status = DftiComputeForward(context->descriptor,
			(void*) (context->input + index * context->input_elements),
			context->output + index * context->output_elements);
		assert(status == DFTI_NO_ERROR);
	} else {
		MKL_LONG status = DftiComputeBackward(context->descriptor,
			(void*) (context->input + index * context->input_elements),
			context->output + index * context->output_elements);
		assert(status == DFTI_NO_ERROR);
	}
}
#endif

unsigned long long benchmark_batch_transform(
	enum transform_type type,
	size_t batch_size,
	size_t input_elements,
	size_t output_elements,
	size_t transform_size,
	const float input[],
	float output[],
	pthreadpool_t threadpool,
	size_t max_iterations)
{
	struct nnpack_context nnpack_context = {
		.type = type,
		.input = input,
		.output = output,
		.transform_size = transform_size,
		.input_elements = input_elements,
		.output_elements = output_elements
	};
#ifdef USE_MKL
	struct mkl_fft_context mkl_context = {
		.descriptor = NULL,
		.forward_fft = false,
		.input = input,
		.output = output,
		.input_elements = input_elements,
		.output_elements = output_elements
	};
#endif
	pthreadpool_function_1d_t compute_function = NULL;
	void* compute_context = NULL;

	switch (type) {
		case transform_type_nnpack_forward_8x8:
			nnpack_context.fft_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_nnpack_forward_16x16:
			nnpack_context.fft_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_nnpack_inverse_8x8:
			nnpack_context.ifft_function = nnp_hwinfo.transforms.ifft8x8_with_offset;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_nnpack_inverse_16x16:
			nnpack_context.ifft_function = nnp_hwinfo.transforms.ifft16x16_with_offset;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
#ifdef USE_MKL
		case transform_type_mkl_perm_forward_8x8:
		case transform_type_mkl_perm_forward_16x16:
			mkl_context.forward_fft = true;
		case transform_type_mkl_perm_inverse_8x8:
		case transform_type_mkl_perm_inverse_16x16:
		{
			MKL_LONG dfti_fft_size[2] = { transform_size, transform_size };
			MKL_LONG dfti_status = DftiCreateDescriptor(&mkl_context.descriptor, DFTI_SINGLE, DFTI_REAL, 2, dfti_fft_size);
			assert(dfti_status == DFTI_NO_ERROR);

			dfti_status = DftiSetValue(mkl_context.descriptor, DFTI_PACKED_FORMAT, DFTI_PERM_FORMAT);
			assert(dfti_status == DFTI_NO_ERROR);

			dfti_status = DftiSetValue(mkl_context.descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
			assert(dfti_status == DFTI_NO_ERROR);

			dfti_status = DftiSetValue(mkl_context.descriptor, DFTI_NUMBER_OF_USER_THREADS,
				pthreadpool_get_threads_count(threadpool));
			assert(dfti_status == DFTI_NO_ERROR);

			dfti_status = DftiCommitDescriptor(mkl_context.descriptor);
			assert(dfti_status == DFTI_NO_ERROR);

			compute_function = (pthreadpool_function_1d_t) compute_mkl_fft;
			compute_context = &mkl_context;
			break;
		}
#endif
		case transform_type_nnpack_winograd6x6of3x3_input:
			nnpack_context.winograd_input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_nnpack_winograd6x6of3x3_kernel:
			nnpack_context.winograd_input_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_nnpack_winograd6x6of3x3_output:
			nnpack_context.winograd_input_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3;
			compute_function = (pthreadpool_function_1d_t) compute_nnpack_transform;
			compute_context = &nnpack_context;
			break;
		case transform_type_unknown:
		default:
			NNP_UNREACHABLE;
	}

	unsigned long long computation_time[max_iterations];
	size_t computation_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		unsigned long long start_time, end_time;
		if (!read_timer(&start_time))
			continue;

		pthreadpool_compute_1d(threadpool, compute_function, compute_context, batch_size);

		if (!read_timer(&end_time))
			continue;

		computation_time[computation_samples++] = end_time - start_time;
	}

#ifdef USE_MKL
	if (mkl_context.descriptor != NULL) {
		MKL_LONG dfti_status = DftiFreeDescriptor(&mkl_context.descriptor);
		assert(dfti_status == DFTI_NO_ERROR);
	}
#endif
	return median(computation_time, computation_samples);
}

unsigned long long profile_batch_fft(
	enum transform_type type,
	size_t batch_size,
	size_t input_elements, 
	size_t output_elements,
	size_t transform_size,
	const float input[],
	float output[],
	int perf_counter_file_descriptor,
	size_t max_iterations)
{
	struct nnpack_context context = {
		.type = type,
		.input = input,
		.output = output,
		.transform_size = transform_size,
		.input_elements = input_elements,
		.output_elements = output_elements
	};

	switch (type) {
		case transform_type_nnpack_forward_8x8:
			context.fft_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			break;
		case transform_type_nnpack_forward_16x16:
			context.fft_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			break;
		case transform_type_nnpack_inverse_8x8:
			context.ifft_function = nnp_hwinfo.transforms.ifft8x8_with_offset;
			break;
		case transform_type_nnpack_inverse_16x16:
			context.ifft_function = nnp_hwinfo.transforms.ifft16x16_with_offset;
			break;
		case transform_type_nnpack_winograd6x6of3x3_input:
			context.winograd_input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
			break;
		case transform_type_nnpack_winograd6x6of3x3_kernel:
			context.winograd_kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			break;
		case transform_type_nnpack_winograd6x6of3x3_output:
			context.winograd_output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3;
			break;
		case transform_type_unknown:
		default:
			NNP_UNREACHABLE;
	}

	unsigned long long overhead_count[max_iterations];
	size_t overhead_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		unsigned long long start_count = 0, end_count = 0;
		if (!read_perf_counter(perf_counter_file_descriptor, &start_count))
			continue;

		if (!read_perf_counter(perf_counter_file_descriptor, &end_count))
			continue;

		overhead_count[overhead_samples++] = end_count - start_count;
	}

	/* Performance counters aren't working */
	if (overhead_samples == 0)
		return ULLONG_MAX;

	unsigned long long computation_count[max_iterations];
	size_t computation_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		unsigned long long start_count = 0, end_count = 0;
		if (!read_perf_counter(perf_counter_file_descriptor, &start_count))
			continue;

		pthreadpool_compute_1d(NULL, (pthreadpool_function_1d_t) compute_nnpack_transform, &context, batch_size);

		if (!read_perf_counter(perf_counter_file_descriptor, &end_count))
			continue;

		computation_count[computation_samples++] = end_count - start_count;
	}

	const unsigned long long median_overhead_count = median(overhead_count, overhead_samples);
	const unsigned long long median_computation_count = median(computation_count, computation_samples);

	if (median_computation_count > median_overhead_count)
		return median_computation_count - median_overhead_count;
	else
		return 0;
}

struct options {
	size_t batch_size;
	enum transform_type type;
	size_t threads;
	size_t iterations;
	bool hardware_events;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -k   --kind               The kind of transformation operation (fft8x8, fft16x16, ifft8x8, ifft16x16, winograd6x6-3x3-input, winograd6x6-3x3-kernel, winograd6x6-3x3-output)\n"
"  -b   --batch              The size of batch of transformation operations\n"
"Optional parameters:\n"
"  -t   --threads            The number of threads (default: all)\n"
"  -i   --iterations         # iterations (default: 7)\n"
"  -e   --hardware-events    Collect hardware events for the kernel\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.type = transform_type_unknown,
		.batch_size = 0,
		.threads = 0,
		.iterations = 7,
		.hardware_events = false,
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
		} else if ((strcmp(argv[argi], "--kind") == 0) || (strcmp(argv[argi], "-k") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected transformation kind\n");
				exit(EXIT_FAILURE);
			}
			if ((strcmp(argv[argi + 1], "nnpack-fft8x8") == 0) || (strcmp(argv[argi + 1], "fft8x8") == 0)) {
				options.type = transform_type_nnpack_forward_8x8;
			} else if ((strcmp(argv[argi + 1], "nnpack-fft16x16") == 0) || (strcmp(argv[argi + 1], "fft16x16") == 0)) {
				options.type = transform_type_nnpack_forward_16x16;
			} else if ((strcmp(argv[argi + 1], "nnpack-ifft8x8") == 0) || (strcmp(argv[argi + 1], "ifft8x8") == 0)) {
				options.type = transform_type_nnpack_inverse_8x8;
			} else if ((strcmp(argv[argi + 1], "nnpack-ifft16x16") == 0) || (strcmp(argv[argi + 1], "ifft16x16") == 0)) {
				options.type = transform_type_nnpack_inverse_16x16;
#ifdef USE_MKL
			} else if (strcmp(argv[argi + 1], "mkl-perm-fft8x8") == 0) {
				options.type = transform_type_mkl_perm_forward_8x8;
			} else if (strcmp(argv[argi + 1], "mkl-perm-fft16x16") == 0) {
				options.type = transform_type_mkl_perm_forward_16x16;
			} else if (strcmp(argv[argi + 1], "mkl-perm-ifft8x8") == 0) {
				options.type = transform_type_mkl_perm_inverse_8x8;
			} else if (strcmp(argv[argi + 1], "mkl-perm-ifft16x16") == 0) {
				options.type = transform_type_mkl_perm_inverse_16x16;
#endif
			} else if ((strcmp(argv[argi + 1], "nnpack-winograd6x6-3x3-input") == 0) || (strcmp(argv[argi + 1], "winograd6x6-3x3-input") == 0)) {
				options.type = transform_type_nnpack_winograd6x6of3x3_input;
			} else if ((strcmp(argv[argi + 1], "nnpack-winograd6x6-3x3-kernel") == 0) || (strcmp(argv[argi + 1], "winograd6x6-3x3-kernel") == 0)) {
				options.type = transform_type_nnpack_winograd6x6of3x3_kernel;
			} else if ((strcmp(argv[argi + 1], "nnpack-winograd6x6-3x3-output") == 0) || (strcmp(argv[argi + 1], "winograd6x6-3x3-output") == 0)) {
				options.type = transform_type_nnpack_winograd6x6of3x3_output;
			} else {
				fprintf(stderr, "Error: invalid transformation kind %s\n", argv[argi + 1]);
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
				fprintf(stderr, "Error: invalid value %s for the number of threads: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
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
		} else if ((strcmp(argv[argi], "--hardware-events") == 0) || (strcmp(argv[argi], "-e") == 0)) {
			options.hardware_events = true;
		} else if ((strcmp(argv[argi], "--help") == 0) || (strcmp(argv[argi], "-h") == 0)) {
			print_options_help(argv[0]);
			exit(EXIT_SUCCESS);
		} else {
			fprintf(stderr, "Error: unknown argument '%s'\n", argv[argi]);
			print_options_help(argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if (options.type == transform_type_unknown) {
		fprintf(stderr, "Error: transformation type is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.batch_size == 0) {
		fprintf(stderr, "Error: batch size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

int main(int argc, char** argv) {
	const struct options options = parse_options(argc, argv);
	size_t input_elements = 0;
	size_t output_elements = 0;
	size_t transform_size = 0;

	switch (options.type) {
		case transform_type_nnpack_forward_8x8:
		case transform_type_nnpack_inverse_8x8:
#ifdef USE_MKL
		case transform_type_mkl_perm_forward_8x8:
		case transform_type_mkl_perm_inverse_8x8:
#endif
			input_elements = 8 * 8;
			output_elements = 8 * 8;
			transform_size = 8;
			break;
		case transform_type_nnpack_forward_16x16:
		case transform_type_nnpack_inverse_16x16:
#ifdef USE_MKL
		case transform_type_mkl_perm_forward_16x16:
		case transform_type_mkl_perm_inverse_16x16:
#endif
			input_elements = 16 * 16;
			output_elements = 16 * 16;
			transform_size = 16;
			break;
		case transform_type_nnpack_winograd6x6of3x3_input:
			input_elements = 8 * 8;
			output_elements = 8 * 8;
			transform_size = 8;
			break;
		case transform_type_nnpack_winograd6x6of3x3_kernel:
			input_elements = 3 * 3;
			output_elements = 8 * 8;
			transform_size = 8;
			break;
		case transform_type_nnpack_winograd6x6of3x3_output:
			input_elements = 8 * 8;
			output_elements = 6 * 6;
			transform_size = 8;
			break;
		default:
			NNP_UNREACHABLE;
	}

	void* input = NULL;
	void* output = NULL;
	if (posix_memalign(&input, 64, input_elements * options.batch_size * sizeof(float)) != 0) {
		fprintf(stderr, "Error: failed to allocate %zu bytes for input array\n",
			input_elements * options.batch_size * sizeof(float));
		exit(EXIT_FAILURE);
	}
	if (posix_memalign(&output, 64, output_elements * options.batch_size * sizeof(float)) != 0) {
		fprintf(stderr, "Error: failed to allocate %zu bytes for output array\n",
			output_elements * options.batch_size * sizeof(float));
		exit(EXIT_FAILURE);
	}

	memset(input, 0, input_elements * options.batch_size * sizeof(float));
	memset(output, 0, output_elements * options.batch_size * sizeof(float));

	{
		pthreadpool_t threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
		printf("Iterations: %zu\n", options.iterations);

		const unsigned long long transformation_nanoseconds = benchmark_batch_transform(
			options.type, options.batch_size,
			input_elements, output_elements, transform_size,
			input, output,
			threadpool, options.iterations);
		const size_t transformation_bytes = (input_elements * options.batch_size + output_elements * options.batch_size) * sizeof(float);
		const double transformation_gbps = ((double) transformation_bytes) / transformation_nanoseconds;

		printf("Time: %5.3lf ms\n", transformation_nanoseconds * 1.0e-6);
		printf("Bandwidth: %5.3lf GB/s\n", transformation_gbps);
		pthreadpool_destroy(threadpool);
	}

	int failures = 0;
	if (options.hardware_events) {
		size_t performance_counters_count = 0;
		const struct performance_counter* performance_counters =
			init_performance_counters(&performance_counters_count);

		for (size_t i = 0; i < performance_counters_count; i++) {
			if (!enable_perf_counter(performance_counters[i].file_descriptor)) {
				failures++;
				continue;
			}

			unsigned long long count = profile_batch_fft(
				options.type, options.batch_size,
				input_elements, output_elements, transform_size,
				input, output,
				performance_counters[i].file_descriptor,
				options.iterations);
			if (count == ULLONG_MAX) {
				failures++;
				continue;
			}

			if (!disable_perf_counter(performance_counters[i].file_descriptor)) {
				failures++;
				continue;
			}

			printf("%s: %llu\n", performance_counters[i].name, count);
		}
	}
	return failures;
}
