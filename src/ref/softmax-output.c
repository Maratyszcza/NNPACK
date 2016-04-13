#include <nnpack.h>
#include <nnpack/reference.h>
#include <nnpack/utils.h>

#include <float.h>
#include <math.h>

static inline float vector_maxf(size_t length, const float array[restrict static length]) {
    float max_element = -FLT_MAX;
    for (size_t i = 0; i < length; i++) {
        max_element = maxf(max_element, array[i]);
    }
    return max_element;
}

static inline float vector_sum_expf_minus_c(size_t length, const float array[restrict static length], float c) {
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        sum += expf(array[i] - c);
    }
    return sum;
}

struct softmax_output_context {
    size_t channels;
    const float* input;
    float* output;
};

static void compute_softmax_output(
    const struct softmax_output_context context[restrict static 1],
    size_t sample)
{
    const size_t channels = context->channels;

    const float (*input)[channels] =
        (const float(*)[channels]) context->input;
    float (*output)[channels] =
        (float(*)[channels]) context->output;

    const float max_element = vector_maxf(channels, input[sample]);
    const float sum_exp = vector_sum_expf_minus_c(channels, input[sample], max_element);
    const float norm_factor = 1.0f / sum_exp;
    for (size_t channel = 0; channel < channels; channel++) {
        output[sample][channel] = norm_factor * expf(input[sample][channel] - max_element);
    }
}

void nnp_softmax_output__reference(
    size_t batch_size,
    size_t channels,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
    struct softmax_output_context softmax_output_context = {
        .channels = channels,
        .input = input,
        .output = output,
    };
    pthreadpool_compute_1d(threadpool,
        (pthreadpool_function_1d_t) compute_softmax_output,
        &softmax_output_context,
        batch_size);
}
