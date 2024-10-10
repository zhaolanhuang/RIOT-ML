#include <string.h>
#include <stdint.h>

int32_t memcpy_(float* dst, float* src, int32_t size) {
    memcpy(dst, src, size);
    return 0;
}