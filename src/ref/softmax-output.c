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

static inline void vector_softmax(size_t length, const float input[restrict static length], float output[restrict static length]) {
    const float max_element = vector_maxf(length, input);
    const float sum_exp = vector_sum_expf_minus_c(length, input, max_element);
    const float norm_factor = 1.0f / sum_exp;
    for (size_t i = 0; i < length; i++) {
        output[i] = norm_factor * expf(input[i] - max_element);
    }
}

void nnp_softmax_output__reference(
    size_t batch_size,
    size_t channels,
    const float* input_pointer,
    float* output_pointer,
    pthreadpool_t threadpool)
{
    const float (*input)[channels] = (const float(*)[channels]) input_pointer;
    float (*output)[channels] = (float(*)[channels]) output_pointer;
    for (size_t sample = 0; sample < batch_size; sample++) {
        vector_softmax(channels, input[sample], output[sample]);
    }
}
