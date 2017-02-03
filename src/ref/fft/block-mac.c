#include <stddef.h>

#include <nnpack/cmplxf.h>


void nnp_macc8x8__ref(float acc[], const float x[], const float y[]) {
    acc[0] += x[0] * y[0];
    acc[1] += x[1] * y[1];
    acc[8] += x[8] * y[8];
    acc[9] += x[9] * y[9];
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 8; j++) {
            if ((i == 0) && (j < 2)) {
                continue;
            }
            const float _Complex new_acc = CMPLXF(acc[i * 16 + j], acc[i * 16 + 8 + j]) + CMPLXF(x[i * 16 + j], x[i * 16 + 8 + j]) * CMPLXF(y[i * 16 + j], -y[i * 16 + 8 + j]);
            acc[i * 16 + j] = crealf(new_acc);
            acc[i * 16 + 8 + j] = cimagf(new_acc);
        }
    }
}

void nnp_macc16x16__ref(float acc[], const float x[], const float y[]) {
    acc[0] += x[0] * y[0];
    acc[1] += x[1] * y[1];
    acc[16] += x[16] * y[16];
    acc[17] += x[17] * y[17];
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            if ((i == 0) && (j < 2)) {
                continue;
            }
            const float _Complex new_acc = CMPLXF(acc[i * 32 + j], acc[i * 32 + 16 + j]) + CMPLXF(x[i * 32 + j], x[i * 32 + 16 + j]) * CMPLXF(y[i * 32 + j], -y[i * 32 + 16 + j]);
            acc[i * 32 + j] = crealf(new_acc);
            acc[i * 32 + 16 + j] = cimagf(new_acc);
        }
    }
}
