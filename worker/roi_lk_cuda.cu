// 稀疏 LK 单层计算。共享内存保存补丁，复现 x86 OpenCV 的浮点累加次序。
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int window = 21;
constexpr int area = window * window;
constexpr int threads = 128;
constexpr float scale = 1.0f / 1048576.0f;

struct Weights {
    int x, y, top_left, top_right, bottom_left, bottom_right;
};

__device__ int reflect(int index, int size) {
    if (static_cast<unsigned>(index) < static_cast<unsigned>(size)) return index;
    if (size == 1) return 0;
    if (index < 0) index = -index;
    const int period = 2 * size - 2;
    if (index >= period) index %= period;
    return min(index, period - index);
}

__device__ int sample(const uint8_t* image, int x, int y, int width, int height) {
    return image[reflect(y, height) * width + reflect(x, width)];
}

__device__ int2 gradient(const uint8_t* image, int x, int y, int w, int h) {
    if (static_cast<unsigned>(x) >= static_cast<unsigned>(w) ||
        static_cast<unsigned>(y) >= static_cast<unsigned>(h)) return make_int2(0, 0);
    const int x0 = reflect(x - 1, w), x1 = reflect(x + 1, w);
    const auto* upper = image + reflect(y - 1, h) * w;
    const auto* lower = image + reflect(y + 1, h) * w;
    const auto* middle = image + y * w;
    return make_int2(3 * (upper[x1] + lower[x1] - upper[x0] - lower[x0])
                         + 10 * (middle[x1] - middle[x0]),
                     3 * (lower[x0] + lower[x1] - upper[x0] - upper[x1])
                         + 10 * (lower[x] - upper[x]));
}

__device__ Weights weights(float x, float y) {
    const int ix = __float2int_rd(x), iy = __float2int_rd(y);
    const float a = x - ix, b = y - iy;
    const int w00 = __float2int_rn((1.f - a) * (1.f - b) * 16384.f);
    const int w01 = __float2int_rn(a * (1.f - b) * 16384.f);
    const int w10 = __float2int_rn((1.f - a) * b * 16384.f);
    return {ix, iy, w00, w01, w10, 16384 - w00 - w01 - w10};
}

__device__ int intensity(const uint8_t* image, Weights q, int dx, int dy, int w, int h) {
    const int x = q.x + dx, y = q.y + dy;
    return (sample(image, x, y, w, h) * q.top_left
          + sample(image, x + 1, y, w, h) * q.top_right
          + sample(image, x, y + 1, w, h) * q.bottom_left
          + sample(image, x + 1, y + 1, w, h) * q.bottom_right + 256) >> 9;
}

__device__ bool inside(int x, int y, int width, int height) {
    return x >= -window && y >= -window && x < width && y < height;
}

template<bool paired>
__device__ float ordered_lane(const int16_t* left, const int16_t* right, int lane) {
    float value = 0.f;
    for (int row = 0; row < window; ++row) {
        const int16_t* a = left + row * window;
        const int16_t* b = right + row * window;
        if (lane == 4) {
            #pragma unroll
            for (int x = 16; x < window; ++x) value += __int2float_rn(a[x] * b[x]);
        } else if (paired) {
            value += __int2float_rn(a[lane] * b[lane] + a[lane + 4] * b[lane + 4]);
            value += __int2float_rn(a[lane + 8] * b[lane + 8] + a[lane + 12] * b[lane + 12]);
        } else {
            #pragma unroll
            for (int x = 0; x < 16; x += 4) value += __int2float_rn(a[x + lane] * b[x + lane]);
        }
    }
    return value;
}

__device__ float merged_lanes(const float* lane) {
    return (lane[4] + ((lane[0] + lane[2]) + (lane[1] + lane[3]))) * scale;
}

struct State {
    Weights target;
    float x, y, next_x, next_y, last_dx, last_dy;
    float a11, a12, a22, inverse;
    bool valid, target_valid, done;
};

__global__ void track_level(const uint8_t* images, const float2* points, float2* result,
                            uint8_t* status, int w, int h, int count, int level,
                            bool first, bool reverse) {
    __shared__ int16_t source[area], dx[area], dy[area], difference[area];
    __shared__ float partial[15];
    __shared__ State state;

    const int thread = threadIdx.x;
    const int ref = blockIdx.y;
    const int index = ref * count + blockIdx.x;
    const auto* input_ref = images + (ref + 1) * w * h;
    const auto* reference = reverse ? images : input_ref;
    const auto* origin = reverse ? input_ref : images;
    const float2 point = points[reverse ? index : blockIdx.x];
    const float factor = 1.0f / (1 << level);
    const float px = point.x * factor, py = point.y * factor;
    const Weights original = weights(px - 10.f, py - 10.f);
    const bool source_valid = inside(original.x, original.y, w, h);

    if (thread == 0) {
        state.next_x = first ? px : result[index].x * 2.f;
        state.next_y = first ? py : result[index].y * 2.f;
        state.x = state.next_x - 10.f;
        state.y = state.next_y - 10.f;
        state.last_dx = 0.f;
        state.last_dy = 0.f;
        state.valid = source_valid;
        state.target_valid = true;
        state.done = false;
    }

    if (source_valid) {
        for (int i = thread; i < area; i += blockDim.x) {
            const int x = i % window, y = i / window;
            source[i] = intensity(origin, original, x, y, w, h);
            const int sx = original.x + x, sy = original.y + y;
            const int2 a = gradient(origin, sx, sy, w, h);
            const int2 b = gradient(origin, sx + 1, sy, w, h);
            const int2 c = gradient(origin, sx, sy + 1, w, h);
            const int2 d = gradient(origin, sx + 1, sy + 1, w, h);
            dx[i] = (a.x * original.top_left + b.x * original.top_right
                   + c.x * original.bottom_left + d.x * original.bottom_right + 8192) >> 14;
            dy[i] = (a.y * original.top_left + b.y * original.top_right
                   + c.y * original.bottom_left + d.y * original.bottom_right + 8192) >> 14;
        }
        __syncthreads();
        if (thread < 15) {
            partial[thread] = ordered_lane<false>(thread < 10 ? dx : dy,
                                                  thread < 5 ? dx : dy, thread % 5);
        }
        __syncthreads();
        if (thread == 0) {
            state.a11 = merged_lanes(partial);
            state.a12 = merged_lanes(partial + 5);
            state.a22 = merged_lanes(partial + 10);
            const float det = state.a11 * state.a22 - state.a12 * state.a12;
            const float trace = state.a11 - state.a22;
            const float eig = (state.a11 + state.a22
                              - sqrtf(trace * trace + 4.f * state.a12 * state.a12)) / 882.f;
            state.valid = eig >= 0.0001f && det >= 1.1920928955078125e-7f;
            if (state.valid) state.inverse = 1.f / det;
        }
        __syncthreads();
        for (int iteration = 0; iteration < 30; ++iteration) {
            if (!state.valid || state.done) break;
            if (thread == 0) {
                state.target = weights(state.x, state.y);
                state.target_valid = inside(state.target.x, state.target.y, w, h);
            }
            __syncthreads();
            if (!state.target_valid) break;
            const Weights target = state.target;
            for (int i = thread; i < area; i += blockDim.x) {
                difference[i] = intensity(reference, target, i % window, i / window, w, h) - source[i];
            }
            __syncthreads();
            if (thread < 10) {
                partial[thread] = ordered_lane<true>(difference, thread < 5 ? dx : dy, thread % 5);
            }
            __syncthreads();
            if (thread == 0) {
                const float b1 = merged_lanes(partial), b2 = merged_lanes(partial + 5);
                const float shift_x = (state.a12 * b2 - state.a22 * b1) * state.inverse;
                const float shift_y = (state.a12 * b1 - state.a11 * b2) * state.inverse;
                state.x += shift_x;
                state.y += shift_y;
                state.next_x = state.x + 10.f;
                state.next_y = state.y + 10.f;
                state.done = double(shift_x) * shift_x + double(shift_y) * shift_y <= 0.0001;
                if (!state.done && iteration > 0
                    && fabsf(shift_x + state.last_dx) < 0.01
                    && fabsf(shift_y + state.last_dy) < 0.01) {
                    state.next_x -= shift_x * 0.5f;
                    state.next_y -= shift_y * 0.5f;
                    state.done = true;
                }
                state.last_dx = shift_x;
                state.last_dy = shift_y;
            }
            __syncthreads();
        }
    }
    if (thread == 0) {
        result[index] = make_float2(state.next_x, state.next_y);
        if (level == 0) {
            status[index] = state.valid && state.target_valid && inside(__float2int_rd(state.next_x - 10.f),
                                                  __float2int_rd(state.next_y - 10.f), w, h);
        }
    }
}
}

extern "C" int roi_lk_track_level(const uint8_t* images, const float2* points, float2* result,
                                  uint8_t* status, int width, int height, int count,
                                  int refs, int level, int first, int reverse, cudaStream_t stream) {
    track_level<<<dim3(count, refs), threads, 0, stream>>>(
        images, points, result, status, width, height, count, level, first, reverse);
    return cudaPeekAtLastError();
}
