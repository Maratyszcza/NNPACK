#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include <cpuid.h>

#include <perf_counter.h>

#include <nnpack/blas.h>
#include <nnpack/macros.h>

extern unsigned long long median(unsigned long long array[], size_t length);

enum gemm_type {
	gemm_type_unknown,
	gemm_type_c8gemm2x2,
	gemm_type_s4c6gemm2x2,
	gemm_type_s8gemm3x3,
};

unsigned long long profile_gemm(
	void (*gemm_function)(size_t, size_t, const float*, const float*, float*, size_t, size_t),
	size_t kc, size_t mc, size_t nc, size_t mr, size_t nr, size_t row_stride, size_t column_stride,
	const float a[], const float b[], float c[],
	int perf_counter_file_descriptor, size_t max_iterations)
{
	unsigned long long overhead_count[max_iterations];
	size_t overhead_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		unsigned long long start_count = 0, end_count = 0;
		if (!read_perf_counter(perf_counter_file_descriptor, &start_count))
			continue;

		unsigned int eax, ebx, ecx, edx;
		__cpuid(0, eax, ebx, ecx, edx);
		__cpuid(0, eax, ebx, ecx, edx);

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

		unsigned int eax, ebx, ecx, edx;
		__cpuid(0, eax, ebx, ecx, edx);

		for (size_t m = 0; m < mc; m += 1) {
			for (size_t n = 0; n < nc; n += 1) {
				gemm_function(kc, 0,
					&a[m * kc * mr * column_stride],
					&b[n * kc * nr * column_stride],
					&c[(m * mr * nc * nr + n * nr) * column_stride],
					row_stride, column_stride);
			}
		}

		__cpuid(0, eax, ebx, ecx, edx);

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
	enum gemm_type type;
	size_t kc;
	size_t mc;
	size_t nc;
	size_t iterations;
	bool hardware_events;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -g   --gemm               The type of GEMM ukernel (s8gemm3x3, s4c6gemm2x2, c8gemm2x2)\n"
"  -k   --kc                 The KC parameter\n"
"  -n   --nc                 The NC parameter\n"
"  -m   --mc                 The MC parameter\n"
"Optional parameters:\n"
"  -i   --iterations         # iterations (default: 151)\n"
"  -e   --hardware-events    Collect hardware events for the kernel\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.type = gemm_type_unknown,
		.kc = 0,
		.mc = 0,
		.nc = 0,
		.iterations = 7,
		.hardware_events = true,
	};
	for (int argi = 1; argi < argc; argi += 1) {
		if ((strcmp(argv[argi], "-g") == 0) || (strcmp(argv[argi], "--gemm") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected k value\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "s8gemm3x3") == 0) {
				options.type = gemm_type_s8gemm3x3;
			} else if (strcmp(argv[argi + 1], "c8gemm2x2") == 0) {
				options.type = gemm_type_c8gemm2x2;
			} else if (strcmp(argv[argi + 1], "s4c6gemm2x2") == 0) {
				options.type = gemm_type_s4c6gemm2x2;
			} else {
				fprintf(stderr, "Error: invalid value %s for the gemm type: expected \"s8gemm3x3\", \"c8gemm2x2\", or \"s4c6gemm2x2\"\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "-k") == 0) || (strcmp(argv[argi], "--kc") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected KC value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.kc) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.kc == 0) {
				fprintf(stderr, "Error: invalid value %s for the KC parameter: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "-m") == 0) || (strcmp(argv[argi], "--mc") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected MC value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.mc) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.mc == 0) {
				fprintf(stderr, "Error: invalid value %s for the MC parameter: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "-n") == 0) || (strcmp(argv[argi], "--nc") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected NC value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.nc) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.nc == 0) {
				fprintf(stderr, "Error: invalid value %s for the NC parameter: positive value expected\n", argv[argi + 1]);
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
	if (options.type == gemm_type_unknown) {
		fprintf(stderr, "Error: gemm ukernel type is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.kc == 0) {
		fprintf(stderr, "Error: KC is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.mc == 0) {
		fprintf(stderr, "Error: MC is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.nc == 0) {
		fprintf(stderr, "Error: NC is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

int main(int argc, char** argv) {
	const struct options options = parse_options(argc, argv);
	void (*gemm_function)(size_t, size_t, const float*, const float*, float*, size_t, size_t) = NULL;
	size_t mr = 0;
	size_t nr = 0;
	size_t components = 0;
	const size_t simd_width = 8;
	switch (options.type) {
		case gemm_type_s4c6gemm2x2:
			mr = 2;
			nr = 2;
			components = 2;
			gemm_function = nnp_s4c6gemm2x2__fma3;
			break;
		case gemm_type_c8gemm2x2:
			mr = 2;
			nr = 2;
			components = 2;
			gemm_function = nnp_c8gemm2x2__fma3;
			break;
		case gemm_type_s8gemm3x3:
			mr = 3;
			nr = 3;
			components = 1;
			gemm_function = nnp_s8gemm3x3__fma3;
			break;
		default:
			NNP_UNREACHABLE;
	}

	const size_t a_size = simd_width * components * mr * options.mc * options.kc * sizeof(float);
	const size_t b_size = simd_width * components * nr * options.nc * options.kc * sizeof(float);
	const size_t c_size = simd_width * components * mr * nr * options.mc * options.nc * sizeof(float);

	float *a = NULL, *b = NULL, *c = NULL;
	if (posix_memalign((void**) &a, 64, a_size) != 0) {
		fprintf(stderr, "Error: failed to allocate %zu bytes for A matrix\n", a_size);
		exit(EXIT_FAILURE);
	}
	if (posix_memalign((void**) &b, 64, b_size) != 0) {
		fprintf(stderr, "Error: failed to allocate %zu bytes for B matrix\n", b_size);
		exit(EXIT_FAILURE);
	}
	if (posix_memalign((void**) &c, 64, c_size) != 0) {
		fprintf(stderr, "Error: failed to allocate %zu bytes for C matrix\n", c_size);
		exit(EXIT_FAILURE);
	}

	memset(c, 0, sizeof(simd_width * components * mr * nr * options.mc * options.nc * sizeof(float)));
	memset(a, 0, sizeof(simd_width * components * mr * options.mc * options.kc * sizeof(float)));
	memset(b, 0, sizeof(simd_width * components * nr * options.nc * options.kc * sizeof(float)));

	const size_t column_stride = simd_width * components;
	const size_t row_stride = options.nc * nr * column_stride;

	double cycles = __builtin_nan("");
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

			unsigned long long count = profile_gemm(
				gemm_function,
				options.kc, options.mc, options.nc,
				mr, nr, row_stride, column_stride,
				a, b, c,
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

			if (strcmp(performance_counters[i].name, "Cycles") == 0) {
				cycles = count;
			}
			if (strcmp(performance_counters[i].name, "FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE") == 0){
				const double simd_flops = count;
				printf("%s: %llu (%.1lf%% peak)\n", performance_counters[i].name, count,
					(simd_flops / (cycles * 4.0)) * 100.0);
			} else if (strcmp(performance_counters[i].name, "Instructions") == 0) {
				const double instructions = count;
				printf("%s: %llu (%.2lf IPC)\n", performance_counters[i].name, count,
					(instructions / cycles));
			} else {
				printf("%s: %llu\n", performance_counters[i].name, count);
			}
		}
	}
	return failures;
}
