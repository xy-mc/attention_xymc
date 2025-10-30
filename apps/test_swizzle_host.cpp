#include <cstdint>
#include <cstdio>

static inline uint32_t swizzle_Q(uint32_t y, uint32_t x) {
    return ((y & 7u) ^ (x >> 2)) << 2 | (x & 3);
}

static inline
uint32_t swizzle_V(uint32_t y, uint32_t x) {
    return (((y & 7) >> 1) ^ (x >> 3)) << 3 | (x & 7);
}

int main() {
    // 64 x 64 grid, x 每次跨越 4
    for (uint32_t y = 0; y < 64; ++y) {
        std::printf("y=%2u: ", y);
        for (uint32_t x = 0; x < 64; x ++) {
            uint32_t out = swizzle_Q(y, x);
            std::printf("%2u ", out);
        }
        std::printf("\n");
    }
    return 0;
}


