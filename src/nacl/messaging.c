#include <stddef.h>
#include <string.h>

#include <ppapi/c/pp_var.h>

#include <nnpack/macros.h>

#include <nacl/interfaces.h>
#include <nacl/stringvars.h>
#include <nacl/benchmark.h>

static void handle_message(PP_Instance instance, struct PP_Var message_var) {
	struct PP_Var benchmark_var = PP_MakeUndefined();
	struct PP_Var iterations_var = PP_MakeUndefined();
	struct PP_Var threads_var = PP_MakeUndefined();
	struct PP_Var algorithm_var = PP_MakeUndefined();
	struct PP_Var batch_size_var = PP_MakeUndefined();
	struct PP_Var input_channels_var = PP_MakeUndefined();
	struct PP_Var output_channels_var = PP_MakeUndefined();
	struct PP_Var input_width_var = PP_MakeUndefined();
	struct PP_Var input_height_var = PP_MakeUndefined();
	struct PP_Var input_padding_var = PP_MakeUndefined();
	struct PP_Var kernel_width_var = PP_MakeUndefined();
	struct PP_Var kernel_height_var = PP_MakeUndefined();
	pthreadpool_t threadpool = NULL;

	/* Check that message is a dictionary */
	if (message_var.type != PP_VARTYPE_DICTIONARY) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_message_type);
		goto cleanup;
	}

	/* Check that message.benchmark is a string */
	benchmark_var = dictionary_interface->Get(message_var, string_var_benchmark);
	if (benchmark_var.type == PP_VARTYPE_UNDEFINED) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_benchmark_unspecified);
		goto cleanup;
	} else if (benchmark_var.type != PP_VARTYPE_STRING) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_benchmark_type);
		goto cleanup;
	}

	/* Convert benchmark string into benchmark_type enumeration value */
	uint32_t benchmark_length = 0;
	const char* benchmark_pointer = var_interface->VarToUtf8(benchmark_var, &benchmark_length);
	if (benchmark_pointer == NULL) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_benchmark_type);
		goto cleanup;
	}
	enum benchmark_type benchmark_type;
	switch (benchmark_length) {
		case sizeof("convolution-forward") - 1:
			if (strncmp(benchmark_pointer, "convolution-forward", benchmark_length) == 0) {
				benchmark_type = benchmark_type_convolution_forward;
			} else {
				goto benchmark_value_error;
			}
			break;
		case sizeof("batch-transform") - 1:
			if (strncmp(benchmark_pointer, "batch-transform", benchmark_length) == 0) {
				benchmark_type = benchmark_type_batch_transform;
			} else {
				goto benchmark_value_error;
			}
			break;
		default:
		benchmark_value_error:
			console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_benchmark_value);
			goto cleanup;
	}

	/* Check that message.iterations is a positive integer number */
	iterations_var = dictionary_interface->Get(message_var, string_var_iterations);
	if (iterations_var.type == PP_VARTYPE_UNDEFINED) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_iterations_unspecified);
		goto cleanup;
	} else if (iterations_var.type != PP_VARTYPE_INT32) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_iterations_type);
		goto cleanup;
	}
	const int32_t iterations = iterations_var.value.as_int;
	if (iterations <= 0) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_iterations_value);
		goto cleanup;
	}

	/* Check that message.threads, if present, is a non-negative integer number, and create a threadpool */
	threads_var = dictionary_interface->Get(message_var, string_var_threads);
	if (threads_var.type == PP_VARTYPE_UNDEFINED) {
		threadpool = pthreadpool_create(0);
	} else if (iterations_var.type != PP_VARTYPE_INT32) {
		console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_threads_type);
		goto cleanup;
	} else {
		const int32_t threads = iterations_var.value.as_int;
		if (threads < 0) {
			console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_threads_value);
			goto cleanup;
		} else if (threads != 0) {
			threadpool = pthreadpool_create((size_t) threads);
		}
	}

	switch (benchmark_type) {
		case benchmark_type_convolution_forward:
		{
			/* Check that message.algorithm is a string */
			algorithm_var = dictionary_interface->Get(message_var, string_var_algorithm);
			if (algorithm_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_algorithm_unspecified);
				goto cleanup;
			} else if (algorithm_var.type != PP_VARTYPE_STRING) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_algorithm_type);
				goto cleanup;
			}

			/* Convert algorithm string into nnp_convolution_method enumeration value */
			uint32_t algorithm_length = 0;
			const char* algorithm_pointer = var_interface->VarToUtf8(algorithm_var, &algorithm_length);
			if (algorithm_pointer == NULL) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_algorithm_type);
				goto cleanup;
			}

			enum nnp_convolution_algorithm convolution_algorithm;
			switch (algorithm_length) {
				case sizeof("wt-8x8") - 1:
					if (strncmp(algorithm_pointer, "wt-8x8", algorithm_length) == 0) {
						convolution_algorithm = nnp_convolution_algorithm_wt8x8;
					} else if (strncmp(algorithm_pointer, "ft-8x8", algorithm_length) == 0) {
						convolution_algorithm = nnp_convolution_algorithm_ft8x8;
					} else {
						goto algorithm_value_error;
					}
					break;
				case sizeof("ft-16x16") - 1:
					if (strncmp(algorithm_pointer, "ft-16x16", algorithm_length) == 0) {
						convolution_algorithm = nnp_convolution_algorithm_ft16x16;
					} else {
						goto algorithm_value_error;
					}
					break;
				default:
				algorithm_value_error:
					console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_algorithm_value);
					goto cleanup;
			}

			/* Check that message.batch_size is a positive integer number */
			batch_size_var = dictionary_interface->Get(message_var, string_var_batch_size);
			if (batch_size_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_batch_size_unspecified);
				goto cleanup;
			} else if (batch_size_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_batch_size_type);
				goto cleanup;
			}
			const int32_t batch_size = batch_size_var.value.as_int;
			if (batch_size <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_batch_size_value);
				goto cleanup;
			}

			/* Check that message.input_channels is a positive integer number */
			input_channels_var = dictionary_interface->Get(message_var, string_var_input_channels);
			if (input_channels_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_channels_unspecified);
				goto cleanup;
			} else if (input_channels_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_channels_type);
				goto cleanup;
			}
			const int32_t input_channels = input_channels_var.value.as_int;
			if (input_channels <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_channels_value);
				goto cleanup;
			}

			/* Check that message.output_channels is a positive integer number */
			output_channels_var = dictionary_interface->Get(message_var, string_var_output_channels);
			if (output_channels_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_output_channels_unspecified);
				goto cleanup;
			} else if (output_channels_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_output_channels_type);
				goto cleanup;
			}
			const int32_t output_channels = output_channels_var.value.as_int;
			if (output_channels <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_output_channels_value);
				goto cleanup;
			}

			/* Check that message.input_height is a positive integer number */
			input_height_var = dictionary_interface->Get(message_var, string_var_input_height);
			if (input_height_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_height_unspecified);
				goto cleanup;
			} else if (input_height_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_height_type);
				goto cleanup;
			}
			const int32_t input_height = input_height_var.value.as_int;
			if (input_height <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_height_value);
				goto cleanup;
			}

			/* Check that message.input_width is a positive integer number */
			input_width_var = dictionary_interface->Get(message_var, string_var_input_width);
			if (input_width_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_width_unspecified);
				goto cleanup;
			} else if (input_width_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_width_type);
				goto cleanup;
			}
			const int32_t input_width = input_width_var.value.as_int;
			if (input_width <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_width_value);
				goto cleanup;
			}

			/* Check that message.input_padding is a non-negative integer number */
			input_padding_var = dictionary_interface->Get(message_var, string_var_input_padding);
			if (input_padding_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_padding_unspecified);
				goto cleanup;
			} else if (input_padding_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_padding_type);
				goto cleanup;
			}
			const int32_t input_padding = input_padding_var.value.as_int;
			if (input_padding < 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_input_padding_value);
				goto cleanup;
			}

			/* Check that message.kernel_height is a positive integer number */
			kernel_height_var = dictionary_interface->Get(message_var, string_var_kernel_height);
			if (kernel_height_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_height_unspecified);
				goto cleanup;
			} else if (kernel_height_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_height_type);
				goto cleanup;
			}
			const int32_t kernel_height = kernel_height_var.value.as_int;
			if (kernel_height <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_height_value);
				goto cleanup;
			}

			/* Check that message.kernel_width is a positive integer number */
			kernel_width_var = dictionary_interface->Get(message_var, string_var_kernel_width);
			if (kernel_width_var.type == PP_VARTYPE_UNDEFINED) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_width_unspecified);
				goto cleanup;
			} else if (kernel_width_var.type != PP_VARTYPE_INT32) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_width_type);
				goto cleanup;
			}
			const int32_t kernel_width = kernel_width_var.value.as_int;
			if (kernel_width <= 0) {
				console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_kernel_width_value);
				goto cleanup;
			}

			struct nnp_profile convolution_forward_profile = benchmark_convolution_output(
				convolution_algorithm,
				(size_t) batch_size,
				(size_t) input_channels,
				(size_t) output_channels,
				(struct nnp_size) { .height = (size_t) input_height, .width = (size_t) input_width },
				(struct nnp_padding) { input_padding, input_padding, input_padding, input_padding },
				(struct nnp_size) { .height = (size_t) kernel_height, .width = (size_t) kernel_width },
				threadpool,
				(size_t) iterations);

			dictionary_interface->Set(message_var, string_var_total_time, PP_MakeDouble(convolution_forward_profile.total));
			dictionary_interface->Set(message_var, string_var_input_transform_time, PP_MakeDouble(convolution_forward_profile.input_transform));
			dictionary_interface->Set(message_var, string_var_kernel_transform_time, PP_MakeDouble(convolution_forward_profile.kernel_transform));
			dictionary_interface->Set(message_var, string_var_output_transform_time, PP_MakeDouble(convolution_forward_profile.output_transform));
			dictionary_interface->Set(message_var, string_var_block_multiplication_time, PP_MakeDouble(convolution_forward_profile.block_multiplication));

			break;
		}
		case benchmark_type_batch_transform:
			console_interface->Log(instance, PP_LOGLEVEL_WARNING, string_var_error_benchmark_value);
			break;
	}

	messaging_interface->PostMessage(instance, message_var);

cleanup:
	var_interface->Release(message_var);
	var_interface->Release(benchmark_var);
	var_interface->Release(iterations_var);
	var_interface->Release(threads_var);
	var_interface->Release(algorithm_var);
	var_interface->Release(batch_size_var);
	var_interface->Release(input_channels_var);
	var_interface->Release(output_channels_var);
	var_interface->Release(input_width_var);
	var_interface->Release(input_height_var);
	var_interface->Release(input_padding_var);
	var_interface->Release(kernel_width_var);
	var_interface->Release(kernel_height_var);
	if (threadpool != NULL) {
		pthreadpool_destroy(threadpool);
	}
}

const struct PPP_Messaging_1_0 plugin_messaging_interface = {
	.HandleMessage = handle_message
};

