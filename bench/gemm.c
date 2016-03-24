#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include <cpuid.h>

#include <perf_counter.h>

#if defined(USE_MKL)
	#include <mkl_cblas.h>
#elif defined(USE_OPENBLAS)
	#include <cblas.h>
#elif defined(USE_BLIS)
	#include <blis/blis.h>
#else
	#error No BLAS library enabled
#endif

extern unsigned long long median(unsigned long long array[], size_t length);
extern void read_memory(const void* memory, size_t length);

enum gemm_type {
	gemm_type_unknown,
	gemm_type_cgemm,
	gemm_type_sgemm,
};

unsigned long long benchmark_gemm(
	const void* memory, size_t cache_size,
	enum gemm_type type,
	size_t m, size_t n, size_t k,
	const float a[], const float b[], float c[],
	size_t max_iterations)
{
	unsigned long long computation_time[max_iterations];
	size_t computation_samples = 0;
	for (size_t iteration = 0; iteration < max_iterations; iteration++) {
		read_memory(memory, cache_size);

		unsigned long long start_time, end_time;
		if (!read_timer(&start_time))
			continue;

		switch (type) {
			case gemm_type_sgemm:
			{
#if defined(USE_MKL) || defined(USE_OPENBLAS)
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					m, n, k,
					1.0f, a, k, b, k, 0.0f, c, n);
#elif defined(USE_BLIS)
				float alpha = 1.0f;
				float beta = 0.0f;
				bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE,
					m, n, k,
					&alpha,
					(float*) a, k, 1,
					(float*) b, k, 1,
					&beta,
					c, n, 1);
#endif
				break;
			}
			case gemm_type_cgemm:
			{
#if defined(USE_MKL) || defined(USE_OPENBLAS)
				float alpha[2] = { 1.0f, 0.0f };
				float beta[2] = { 0.0f, 0.0f };
				cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					m, n, k,
					alpha, a, k, b, k, beta, c, n);
#elif defined(USE_BLIS)
				scomplex alpha = { 1.0f, 0.0f };
				scomplex beta = { 0.0f, 0.0f };
				bli_cgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE,
					m, n, k,
					&alpha,
					(scomplex*) a, k, 1,
					(scomplex*) b, k, 1,
					&beta,
					(scomplex*) c, n, 1);
#endif
				break;
			}
		}

		if (!read_timer(&end_time))
			continue;

		computation_time[computation_samples++] = end_time - start_time;
	}

	return median(computation_time, computation_samples);
}

struct options {
	enum gemm_type type;
	size_t m;
	size_t n;
	size_t k;
	size_t iterations;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -g   --gemm               The type of GEMM operation (sgemm, cgemm)\n"
"  -m                        The M dimension\n"
"  -n                        The N dimension\n"
"  -k                        The K dimension\n"
"Optional parameters:\n"
"  -i   --iterations         # iterations (default: 151)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.type = gemm_type_unknown,
		.k = 0,
		.m = 0,
		.n = 0,
		.iterations = 151
	};
	for (int argi = 1; argi < argc; argi += 1) {
		if ((strcmp(argv[argi], "-g") == 0) || (strcmp(argv[argi], "--gemm") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected k value\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "sgemm") == 0) {
				options.type = gemm_type_sgemm;
			} else if (strcmp(argv[argi + 1], "cgemm") == 0) {
				options.type = gemm_type_cgemm;
			} else {
				fprintf(stderr, "Error: invalid value %s for the gemm type: expected \"sgemm\", \"cgemm\"\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if (strcmp(argv[argi], "-k") == 0) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected K value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.k) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.k == 0) {
				fprintf(stderr, "Error: invalid value %s for the K parameter: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if (strcmp(argv[argi], "-m") == 0) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected M value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.m) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.m == 0) {
				fprintf(stderr, "Error: invalid value %s for the M parameter: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if (strcmp(argv[argi], "-n") == 0) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected N value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.n) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.n == 0) {
				fprintf(stderr, "Error: invalid value %s for the N parameter: positive value expected\n", argv[argi + 1]);
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
		fprintf(stderr, "Error: gemm kernel type is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.m == 0) {
		fprintf(stderr, "Error: M is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.n == 0) {
		fprintf(stderr, "Error: N is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.k == 0) {
		fprintf(stderr, "Error: K is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

int main(int argc, char** argv) {
	const struct options options = parse_options(argc, argv);
	size_t components = 0;
	switch (options.type) {
		case gemm_type_sgemm:
			components = 1;
			break;
		case gemm_type_cgemm:
			components = 2;
			break;
		default:
			__builtin_unreachable();
	}

	const size_t cache_size = 128 * 1024 * 1024;
	void* memory = valloc(cache_size);

#ifdef USE_BLIS
	err_t blis_status = bli_init();
	if (blis_status != BLIS_SUCCESS) {
		fprintf(stderr, "BLIS initialization failed: error code %d\n", blis_status);
		exit(EXIT_FAILURE);
	}
#endif

	float* c = (float*) valloc(options.m * options.n * components * sizeof(float));
	float* a = (float*) valloc(options.m * options.k * components * sizeof(float));
	float* b = (float*) valloc(options.k * options.n * components * sizeof(float));

	memset(c, 0, options.m * options.n * components * sizeof(float));
	memset(a, 0, options.m * options.k * components * sizeof(float));
	memset(b, 0, options.k * options.n * components * sizeof(float));

	{
		printf("Iterations: %zu\n", options.iterations);

		const unsigned long long gemm_nanoseconds = benchmark_gemm(
			memory, cache_size,
			options.type,
			options.m, options.n, options.k,
			a, b, c,
			options.iterations);
		const double gemm_gflops = 2.0 * components * components * options.m * options.n * options.k / ((double) gemm_nanoseconds);

		printf("Time: %5.3lf ms\n", gemm_nanoseconds * 1.0e-6);
		printf("Performance: %5.3lf GFLOPs/s\n", gemm_gflops);
	}
}
