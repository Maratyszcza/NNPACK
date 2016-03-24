#include <ppapi/c/pp_var.h>

#include <nacl/interfaces.h>

struct PP_Var string_var_error_message_type;
struct PP_Var string_var_error_benchmark_unspecified;
struct PP_Var string_var_error_benchmark_type;
struct PP_Var string_var_error_benchmark_value;
struct PP_Var string_var_error_iterations_unspecified;
struct PP_Var string_var_error_iterations_type;
struct PP_Var string_var_error_iterations_value;
struct PP_Var string_var_error_threads_type;
struct PP_Var string_var_error_threads_value;
struct PP_Var string_var_error_algorithm_unspecified;
struct PP_Var string_var_error_algorithm_type;
struct PP_Var string_var_error_algorithm_value;
struct PP_Var string_var_error_batch_size_unspecified;
struct PP_Var string_var_error_batch_size_type;
struct PP_Var string_var_error_batch_size_value;
struct PP_Var string_var_error_input_channels_unspecified;
struct PP_Var string_var_error_input_channels_type;
struct PP_Var string_var_error_input_channels_value;
struct PP_Var string_var_error_output_channels_unspecified;
struct PP_Var string_var_error_output_channels_type;
struct PP_Var string_var_error_output_channels_value;
struct PP_Var string_var_error_input_height_unspecified;
struct PP_Var string_var_error_input_height_type;
struct PP_Var string_var_error_input_height_value;
struct PP_Var string_var_error_input_width_unspecified;
struct PP_Var string_var_error_input_width_type;
struct PP_Var string_var_error_input_width_value;
struct PP_Var string_var_error_input_padding_unspecified;
struct PP_Var string_var_error_input_padding_type;
struct PP_Var string_var_error_input_padding_value;
struct PP_Var string_var_error_kernel_height_unspecified;
struct PP_Var string_var_error_kernel_height_type;
struct PP_Var string_var_error_kernel_height_value;
struct PP_Var string_var_error_kernel_width_unspecified;
struct PP_Var string_var_error_kernel_width_type;
struct PP_Var string_var_error_kernel_width_value;

struct PP_Var string_var_benchmark;
struct PP_Var string_var_iterations;
struct PP_Var string_var_threads;
struct PP_Var string_var_algorithm;
struct PP_Var string_var_batch_size;
struct PP_Var string_var_input_channels;
struct PP_Var string_var_output_channels;
struct PP_Var string_var_input_height;
struct PP_Var string_var_input_width;
struct PP_Var string_var_input_padding;
struct PP_Var string_var_kernel_height;
struct PP_Var string_var_kernel_width;

struct PP_Var string_var_total_time;
struct PP_Var string_var_input_transform_time;
struct PP_Var string_var_kernel_transform_time;
struct PP_Var string_var_output_transform_time;
struct PP_Var string_var_block_multiplication_time;

#define VAR_FROM_UTF8_LITERAL(interface, literal) \
	interface->VarFromUtf8(literal, sizeof(literal) - 1);

void init_string_vars(void) {
	string_var_error_message_type = VAR_FROM_UTF8_LITERAL(var_interface, "Message is not of dictionary type");
	string_var_error_benchmark_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Benchmark is not specified");
	string_var_error_benchmark_type = VAR_FROM_UTF8_LITERAL(var_interface, "Benchmark is not a string");
	string_var_error_benchmark_value = VAR_FROM_UTF8_LITERAL(var_interface, "Invalid benchmark name");
	string_var_error_iterations_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Number of iterations is not specified");
	string_var_error_iterations_type = VAR_FROM_UTF8_LITERAL(var_interface, "Number of iterations is not an integer");
	string_var_error_iterations_value = VAR_FROM_UTF8_LITERAL(var_interface, "Number of iterations is not a positive number");
	string_var_error_threads_type = VAR_FROM_UTF8_LITERAL(var_interface, "Number of threads is not an integer");
	string_var_error_threads_value = VAR_FROM_UTF8_LITERAL(var_interface, "Number of threads is negative");
	string_var_error_algorithm_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Algorithm is not specified");
	string_var_error_algorithm_type = VAR_FROM_UTF8_LITERAL(var_interface, "Algorithm is not a string");
	string_var_error_algorithm_value = VAR_FROM_UTF8_LITERAL(var_interface, "Invalid algorithm name");
	string_var_error_batch_size_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Batch size is not specified");
	string_var_error_batch_size_type = VAR_FROM_UTF8_LITERAL(var_interface, "Batch size is not an integer");
	string_var_error_batch_size_value = VAR_FROM_UTF8_LITERAL(var_interface, "Batch size is not a positive number");
	string_var_error_input_channels_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Number of input channels is not specified");
	string_var_error_input_channels_type = VAR_FROM_UTF8_LITERAL(var_interface, "Number of input channels is not an integer");
	string_var_error_input_channels_value = VAR_FROM_UTF8_LITERAL(var_interface, "Number of input channels is not a positive number");
	string_var_error_output_channels_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Number of output channels is not specified");
	string_var_error_output_channels_type = VAR_FROM_UTF8_LITERAL(var_interface, "Number of output channels is not an integer");
	string_var_error_output_channels_value = VAR_FROM_UTF8_LITERAL(var_interface, "Number of output channels is not a positive number");
	string_var_error_input_height_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Input height is not specified");
	string_var_error_input_height_type = VAR_FROM_UTF8_LITERAL(var_interface, "Input height is not an integer");
	string_var_error_input_height_value = VAR_FROM_UTF8_LITERAL(var_interface, "Input height is not a positive number");
	string_var_error_input_width_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Input width is not specified");
	string_var_error_input_width_type = VAR_FROM_UTF8_LITERAL(var_interface, "Input width is not an integer");
	string_var_error_input_width_value = VAR_FROM_UTF8_LITERAL(var_interface, "Input width is not a positive number");
	string_var_error_input_padding_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Input padding is not specified");
	string_var_error_input_padding_type = VAR_FROM_UTF8_LITERAL(var_interface, "Input padding is not an integer");
	string_var_error_input_padding_value = VAR_FROM_UTF8_LITERAL(var_interface, "Input padding is a negative number");
	string_var_error_kernel_height_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel height is not specified");
	string_var_error_kernel_height_type = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel height is not an integer");
	string_var_error_kernel_height_value = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel height is not a positive number");
	string_var_error_kernel_width_unspecified = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel width is not specified");
	string_var_error_kernel_width_type = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel width is not an integer");
	string_var_error_kernel_width_value = VAR_FROM_UTF8_LITERAL(var_interface, "Kernel width is not a positive number");

	string_var_benchmark = VAR_FROM_UTF8_LITERAL(var_interface, "benchmark");
	string_var_iterations = VAR_FROM_UTF8_LITERAL(var_interface, "iterations");
	string_var_threads = VAR_FROM_UTF8_LITERAL(var_interface, "threads");
	string_var_algorithm = VAR_FROM_UTF8_LITERAL(var_interface, "algorithm");
	string_var_batch_size = VAR_FROM_UTF8_LITERAL(var_interface, "batch-size");
	string_var_input_channels = VAR_FROM_UTF8_LITERAL(var_interface, "input-channels");
	string_var_output_channels = VAR_FROM_UTF8_LITERAL(var_interface, "output-channels");
	string_var_input_height = VAR_FROM_UTF8_LITERAL(var_interface, "input-height");
	string_var_input_width = VAR_FROM_UTF8_LITERAL(var_interface, "input-width");
	string_var_input_padding = VAR_FROM_UTF8_LITERAL(var_interface, "input-padding");
	string_var_kernel_height = VAR_FROM_UTF8_LITERAL(var_interface, "kernel-height");
	string_var_kernel_width = VAR_FROM_UTF8_LITERAL(var_interface, "kernel-width");

	string_var_total_time = VAR_FROM_UTF8_LITERAL(var_interface, "total-time");
	string_var_input_transform_time = VAR_FROM_UTF8_LITERAL(var_interface, "input-transform-time");
	string_var_kernel_transform_time = VAR_FROM_UTF8_LITERAL(var_interface, "kernel-transform-time");
	string_var_output_transform_time = VAR_FROM_UTF8_LITERAL(var_interface, "output-transform-time");
	string_var_block_multiplication_time = VAR_FROM_UTF8_LITERAL(var_interface, "block-multiplication-time");
}

void release_string_vars(void) {
	var_interface->Release(string_var_error_message_type);
	var_interface->Release(string_var_error_benchmark_unspecified);
	var_interface->Release(string_var_error_benchmark_type);
	var_interface->Release(string_var_error_benchmark_value);
	var_interface->Release(string_var_error_iterations_unspecified);
	var_interface->Release(string_var_error_iterations_type);
	var_interface->Release(string_var_error_iterations_value);
	var_interface->Release(string_var_error_threads_type);
	var_interface->Release(string_var_error_threads_value);
	var_interface->Release(string_var_error_algorithm_unspecified);
	var_interface->Release(string_var_error_algorithm_type);
	var_interface->Release(string_var_error_algorithm_value);
	var_interface->Release(string_var_error_batch_size_unspecified);
	var_interface->Release(string_var_error_batch_size_type);
	var_interface->Release(string_var_error_batch_size_value);
	var_interface->Release(string_var_error_input_channels_unspecified);
	var_interface->Release(string_var_error_input_channels_type);
	var_interface->Release(string_var_error_input_channels_value);
	var_interface->Release(string_var_error_output_channels_unspecified);
	var_interface->Release(string_var_error_output_channels_type);
	var_interface->Release(string_var_error_output_channels_value);
	var_interface->Release(string_var_error_input_height_unspecified);
	var_interface->Release(string_var_error_input_height_type);
	var_interface->Release(string_var_error_input_height_value);
	var_interface->Release(string_var_error_input_width_unspecified);
	var_interface->Release(string_var_error_input_width_type);
	var_interface->Release(string_var_error_input_width_value);
	var_interface->Release(string_var_error_input_padding_unspecified);
	var_interface->Release(string_var_error_input_padding_type);
	var_interface->Release(string_var_error_input_padding_value);
	var_interface->Release(string_var_error_kernel_height_unspecified);
	var_interface->Release(string_var_error_kernel_height_type);
	var_interface->Release(string_var_error_kernel_height_value);
	var_interface->Release(string_var_error_kernel_width_unspecified);
	var_interface->Release(string_var_error_kernel_width_type);
	var_interface->Release(string_var_error_kernel_width_value);

	var_interface->Release(string_var_benchmark);
	var_interface->Release(string_var_iterations);
	var_interface->Release(string_var_threads);
	var_interface->Release(string_var_algorithm);
	var_interface->Release(string_var_batch_size);
	var_interface->Release(string_var_input_channels);
	var_interface->Release(string_var_output_channels);
	var_interface->Release(string_var_input_height);
	var_interface->Release(string_var_input_width);
	var_interface->Release(string_var_input_padding);
	var_interface->Release(string_var_kernel_height);
	var_interface->Release(string_var_kernel_width);

	var_interface->Release(string_var_total_time);
	var_interface->Release(string_var_input_transform_time);
	var_interface->Release(string_var_kernel_transform_time);
	var_interface->Release(string_var_output_transform_time);
	var_interface->Release(string_var_block_multiplication_time);
}
