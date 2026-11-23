/* SPDX-License-Identifier: LicenseRef-Bartsch-Personal-Free-5Pct-Commercial
 * License: Free for Individuals & Non-Commercial Use; 5% Attributable Net Profit Share for Companies/Commercial Use (see LICENSE)
 * Copyright (c) 2026 Philip Bartsch (batteryphil)
 *
 * vault_tts_cpu.c — High-Throughput Cache-Resident CPU Engine for VAULT + PRIMAL TTS
 *   - Keeps V3_P4_0 (4.50 bpw, G=32) and V3_Q2_0 (2.25 bpw, G=64) weights permanently bit-packed in L2/L3/RAM.
 *   - Uses 256-bit AVX2 VPSHUFB + VPSIGNB + VPMADDUBSW (zero-spill 4x2 register-blocked INT8xP4 / INT8xQ2 microkernel)
 *     at full Skylake-SP / Zen License-0 Turbo clock (no AVX-512 frequency throttling!), with portable scalar fallback.
 *   - Includes fused in-place AVX2 Rotary Position Embedding (RoPE) and packed 1D Convolution (Conv1d).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

static float g_f16_tab[65536];
static int g_init_done = 0;

static float f16_bits_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s << 31;
        } else {
            int shift = -1;
            do { shift++; m <<= 1; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            u = (s << 31) | ((uint32_t)(112 - shift) << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = s << 31; /* clamp Inf/NaN safely to 0 */
    } else {
        u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    }
    union { uint32_t u; float f; } cv;
    cv.u = u;
    return cv.f;
}

__attribute__((constructor))
void vault_tts_cpu_init(void) {
    if (g_init_done) return;
    for (uint32_t i = 0; i < 65536u; i++) {
        g_f16_tab[i] = f16_bits_to_f32((uint16_t)i);
    }
    g_init_done = 1;
}

static inline float fast_gelu_scalar(float x) {
    /* Exact tanh-approx / erf GELU accurate to < 2e-4 */
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    if (inner > 9.0f) return x;
    if (inner < -9.0f) return 0.0f;
    float e2 = expf(-2.0f * inner);
    float th = (1.0f - e2) / (1.0f + e2);
    return 0.5f * x * (1.0f + th);
}

/* ============================================================================
 * 1. One-Time Load-Time Planar SIMD Repacking (< 5 ms per model)
 * ========================================================================== */
void vault_repack_p4_g32(const uint8_t *q_in, uint8_t *q_out, int64_t rows, int64_t pcols) {
    int64_t n_blocks = pcols / 32;
    #pragma omp parallel for schedule(static) if (rows >= 64)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *src_r = q_in + r * (pcols / 2);
        uint8_t *dst_r = q_out + r * (pcols / 2);
        for (int64_t b = 0; b < n_blocks; b++) {
            const uint8_t *s = src_r + b * 16;
            uint8_t *d = dst_r + b * 16;
            /* Unpack interleaved (2*j, 2*j+1) into planar (k, k+16) for k=0..15 */
            uint8_t tmp[32];
            for (int j = 0; j < 16; j++) {
                tmp[2 * j + 0] = s[j] & 0x0Fu;
                tmp[2 * j + 1] = (s[j] >> 4) & 0x0Fu;
            }
            for (int k = 0; k < 16; k++) {
                d[k] = (uint8_t)(tmp[k] | (tmp[k + 16] << 4));
            }
        }
    }
}

void vault_repack_q2_g64(const uint8_t *q_in, uint8_t *q_out, int64_t rows, int64_t pcols) {
    int64_t n_blocks = pcols / 64;
    #pragma omp parallel for schedule(static) if (rows >= 64)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *src_r = q_in + r * (pcols / 4);
        uint8_t *dst_r = q_out + r * (pcols / 4);
        for (int64_t b = 0; b < n_blocks; b++) {
            const uint8_t *s = src_r + b * 16;
            uint8_t *d = dst_r + b * 16;
            uint8_t tmp[64];
            for (int j = 0; j < 16; j++) {
                uint8_t v = s[j];
                tmp[4 * j + 0] = v & 0x03u;
                tmp[4 * j + 1] = (v >> 2) & 0x03u;
                tmp[4 * j + 2] = (v >> 4) & 0x03u;
                tmp[4 * j + 3] = (v >> 6) & 0x03u;
            }
            for (int k = 0; k < 16; k++) {
                d[k] = (uint8_t)(tmp[k] | (tmp[k + 16] << 2) | (tmp[k + 32] << 4) | (tmp[k + 48] << 6));
            }
        }
    }
}

/* ============================================================================
 * 2. Dynamic Block-32 INT8 Activation Quantizer (~0.1 ms for [2400, 1024])
 * ========================================================================== */
static void quantize_act_q8_g32(
    const float *x,
    int8_t *qx,
    float *dx,
    int64_t M,
    int64_t K,
    int64_t pcols,
    int n_threads
) {
    int64_t n_blocks = pcols / 32;
    #pragma omp parallel for num_threads(n_threads) schedule(static) if (M * pcols >= 16384 && n_threads > 1)
    for (int64_t m = 0; m < M; m++) {
        const float *xm = x + m * K;
        int8_t *qxm = qx + m * pcols;
        float *dxm = dx + m * n_blocks;

        for (int64_t b = 0; b < n_blocks; b++) {
            int64_t col0 = b * 32;
            float buf[32];
            const float *xb;
            if (col0 + 32 <= K) {
                xb = xm + col0;
            } else {
                for (int j = 0; j < 32; j++) {
                    buf[j] = (col0 + j < K) ? xm[col0 + j] : 0.0f;
                }
                xb = buf;
            }

#ifdef __AVX2__
            const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 v0 = _mm256_loadu_ps(xb + 0);
            __m256 v1 = _mm256_loadu_ps(xb + 8);
            __m256 v2 = _mm256_loadu_ps(xb + 16);
            __m256 v3 = _mm256_loadu_ps(xb + 24);
            __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
            __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
            __m256 m_all = _mm256_max_ps(m01, m23);
            __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m_all), _mm256_extractf128_ps(m_all, 1));
            m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
            float amax = _mm_cvtss_f32(m128);

            if (amax < 1e-30f) {
                dxm[b] = 0.0f;
                _mm256_storeu_si256((__m256i *)(qxm + col0), _mm256_setzero_si256());
            } else {
                float d = amax * (1.0f / 127.0f);
                __m256 vinv = _mm256_set1_ps(127.0f / amax);
                dxm[b] = d;
                __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                __m128i s0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
                __m128i s1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
                __m128i s2 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
                __m128i s3 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));

                _mm_storeu_si128((__m128i *)(qxm + col0 + 0), _mm_packs_epi16(s0, s1));
                _mm_storeu_si128((__m128i *)(qxm + col0 + 16), _mm_packs_epi16(s2, s3));
            }
#else
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = fabsf(xb[j]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-30f) {
                dxm[b] = 0.0f;
                memset(qxm + col0, 0, 32);
            } else {
                float inv = 127.0f / amax;
                dxm[b] = amax * (1.0f / 127.0f);
                for (int j = 0; j < 32; j++) {
                    int v = (int)lrintf(xb[j] * inv);
                    if (v > 127) v = 127;
                    if (v < -127) v = -127;
                    qxm[col0 + j] = (int8_t)v;
                }
            }
#endif
        }
    }
}

#ifdef __AVX2__
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

static inline int choose_omp_threads(int64_t M, int64_t N, int64_t K, int max_threads) {
    int64_t ops = M * N * K;
    if (ops < 262144) return 1;        /* < 256K MACs: single thread beats barrier */
    if (ops < 1500000) return (max_threads > 4) ? 4 : max_threads;
    return max_threads;
}

/* ============================================================================
 * 3. Zero-Spill 4x2 Register-Blocked AVX2 VPMADDUBSW P4_0 GEMM
 * ========================================================================== */
void vault_gemm_p4_g32(
    const float *x,
    const uint8_t *q_simd,
    const uint16_t *d_f16,
    const float *bias,
    float *out,
    int64_t M,
    int64_t K,
    int64_t pcols,
    int64_t N,
    int fuse_gelu,
    int max_threads
) {
    if (!g_init_done) vault_tts_cpu_init();
    int64_t n_blocks = pcols / 32;
    int nt = choose_omp_threads(M, N, pcols, max_threads > 0 ? max_threads : 1);

    int8_t *qx = (int8_t *)aligned_alloc(64, ((size_t)(M * pcols) + 63u) & ~63u);
    float *dx = (float *)aligned_alloc(64, ((size_t)(M * n_blocks * (int64_t)sizeof(float)) + 63u) & ~63u);
    quantize_act_q8_g32(x, qx, dx, M, K, pcols, nt);

    static const int8_t p4_lut_arr[16] = {
        -127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101
    };

    int64_t n_pairs = (N + 1) / 2;

    #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int64_t np = 0; np < n_pairs; np++) {
        int64_t n0 = np * 2;
        int64_t n1 = (n0 + 1 < N) ? (n0 + 1) : n0;
        const uint8_t *w0_row = q_simd + n0 * (pcols / 2);
        const uint8_t *w1_row = q_simd + n1 * (pcols / 2);
        const uint16_t *d0_row = d_f16 + n0 * n_blocks;
        const uint16_t *d1_row = d_f16 + n1 * n_blocks;
        float b0 = bias ? bias[n0] : 0.0f;
        float b1 = bias ? bias[n1] : 0.0f;

#ifdef __AVX2__
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i lut128 = _mm_loadu_si128((const __m128i *)p4_lut_arr);
        const __m256i ones16 = _mm256_set1_epi16(1);

        int64_t m = 0;
        for (; m + 4 <= M; m += 4) {
            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

            const int8_t *qx0 = qx + (m + 0) * pcols;
            const int8_t *qx1 = qx + (m + 1) * pcols;
            const int8_t *qx2 = qx + (m + 2) * pcols;
            const int8_t *qx3 = qx + (m + 3) * pcols;
            const float *dx0 = dx + (m + 0) * n_blocks;
            const float *dx1 = dx + (m + 1) * n_blocks;
            const float *dx2 = dx + (m + 2) * n_blocks;
            const float *dx3 = dx + (m + 3) * n_blocks;

            for (int64_t b = 0; b < n_blocks; b++) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0_row + b * 16));
                __m128i w0_lo = _mm_shuffle_epi8(lut128, _mm_and_si128(r0, m0f));
                __m128i w0_hi = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(r0, 4), m0f));
                __m256i w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(w0_lo), w0_hi, 1);
                __m256i aw0 = _mm256_abs_epi8(w0);

                __m128i r1 = _mm_loadu_si128((const __m128i *)(w1_row + b * 16));
                __m128i w1_lo = _mm_shuffle_epi8(lut128, _mm_and_si128(r1, m0f));
                __m128i w1_hi = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(r1, 4), m0f));
                __m256i w1 = _mm256_inserti128_si256(_mm256_castsi128_si256(w1_lo), w1_hi, 1);
                __m256i aw1 = _mm256_abs_epi8(w1);

                float dw0 = g_f16_tab[d0_row[b]];
                float dw1 = g_f16_tab[d1_row[b]];
                int64_t off = b * 32;

                __m256i x0 = _mm256_loadu_si256((const __m256i *)(qx0 + off));
                acc00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(x0, w0)), ones16)), _mm256_set1_ps(dw0 * dx0[b]), acc00);
                acc01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(x0, w1)), ones16)), _mm256_set1_ps(dw1 * dx0[b]), acc01);

                __m256i x1 = _mm256_loadu_si256((const __m256i *)(qx1 + off));
                acc10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(x1, w0)), ones16)), _mm256_set1_ps(dw0 * dx1[b]), acc10);
                acc11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(x1, w1)), ones16)), _mm256_set1_ps(dw1 * dx1[b]), acc11);

                __m256i x2 = _mm256_loadu_si256((const __m256i *)(qx2 + off));
                acc20 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(x2, w0)), ones16)), _mm256_set1_ps(dw0 * dx2[b]), acc20);
                acc21 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(x2, w1)), ones16)), _mm256_set1_ps(dw1 * dx2[b]), acc21);

                __m256i x3 = _mm256_loadu_si256((const __m256i *)(qx3 + off));
                acc30 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(x3, w0)), ones16)), _mm256_set1_ps(dw0 * dx3[b]), acc30);
                acc31 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(x3, w1)), ones16)), _mm256_set1_ps(dw1 * dx3[b]), acc31);
            }

            float v00 = hsum256_ps(acc00) + b0, v10 = hsum256_ps(acc10) + b0, v20 = hsum256_ps(acc20) + b0, v30 = hsum256_ps(acc30) + b0;
            if (fuse_gelu) { v00 = fast_gelu_scalar(v00); v10 = fast_gelu_scalar(v10); v20 = fast_gelu_scalar(v20); v30 = fast_gelu_scalar(v30); }
            out[(m + 0) * N + n0] = v00;
            out[(m + 1) * N + n0] = v10;
            out[(m + 2) * N + n0] = v20;
            out[(m + 3) * N + n0] = v30;

            if (n1 != n0) {
                float v01 = hsum256_ps(acc01) + b1, v11 = hsum256_ps(acc11) + b1, v21 = hsum256_ps(acc21) + b1, v31 = hsum256_ps(acc31) + b1;
                if (fuse_gelu) { v01 = fast_gelu_scalar(v01); v11 = fast_gelu_scalar(v11); v21 = fast_gelu_scalar(v21); v31 = fast_gelu_scalar(v31); }
                out[(m + 0) * N + n1] = v01;
                out[(m + 1) * N + n1] = v11;
                out[(m + 2) * N + n1] = v21;
                out[(m + 3) * N + n1] = v31;
            }
        }

        for (; m < M; m++) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            const int8_t *qxm = qx + m * pcols;
            const float *dxm = dx + m * n_blocks;
            for (int64_t b = 0; b < n_blocks; b++) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0_row + b * 16));
                __m256i w0 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_shuffle_epi8(lut128, _mm_and_si128(r0, m0f))),
                    _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(r0, 4), m0f)), 1);
                __m256i xm = _mm256_loadu_si256((const __m256i *)(qxm + b * 32));
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(xm, w0)), ones16)), _mm256_set1_ps(g_f16_tab[d0_row[b]] * dxm[b]), a0);
                if (n1 != n0) {
                    __m128i r1 = _mm_loadu_si128((const __m128i *)(w1_row + b * 16));
                    __m256i w1 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_shuffle_epi8(lut128, _mm_and_si128(r1, m0f))),
                        _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(r1, 4), m0f)), 1);
                    a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(xm, w1)), ones16)), _mm256_set1_ps(g_f16_tab[d1_row[b]] * dxm[b]), a1);
                }
            }
            float v0 = hsum256_ps(a0) + b0;
            out[m * N + n0] = fuse_gelu ? fast_gelu_scalar(v0) : v0;
            if (n1 != n0) {
                float v1 = hsum256_ps(a1) + b1;
                out[m * N + n1] = fuse_gelu ? fast_gelu_scalar(v1) : v1;
            }
        }
#else
        for (int64_t m = 0; m < M; m++) {
            const int8_t *qxm = qx + m * pcols;
            const float *dxm = dx + m * n_blocks;
            float s0 = b0, s1 = b1;
            for (int64_t b = 0; b < n_blocks; b++) {
                int32_t isum0 = 0, isum1 = 0;
                const uint8_t *p0 = w0_row + b * 16;
                const uint8_t *p1 = w1_row + b * 16;
                const int8_t *xb = qxm + b * 32;
                for (int k = 0; k < 16; k++) {
                    isum0 += (int32_t)p4_lut_arr[p0[k] & 0x0Fu] * xb[k] + (int32_t)p4_lut_arr[p0[k] >> 4] * xb[k + 16];
                    isum1 += (int32_t)p4_lut_arr[p1[k] & 0x0Fu] * xb[k] + (int32_t)p4_lut_arr[p1[k] >> 4] * xb[k + 16];
                }
                s0 += (float)isum0 * (g_f16_tab[d0_row[b]] * dxm[b]);
                s1 += (float)isum1 * (g_f16_tab[d1_row[b]] * dxm[b]);
            }
            out[m * N + n0] = fuse_gelu ? fast_gelu_scalar(s0) : s0;
            if (n1 != n0) out[m * N + n1] = fuse_gelu ? fast_gelu_scalar(s1) : s1;
        }
#endif
    }

    free(qx);
    free(dx);
}

/* ============================================================================
 * 4. Zero-Spill 4x2 Register-Blocked AVX2 VPMADDUBSW Q2_0 (G=64) GEMM
 * ========================================================================== */
void vault_gemm_q2_g64(
    const float *x,
    const uint8_t *q_simd,
    const uint16_t *d_f16,
    const float *bias,
    float *out,
    int64_t M,
    int64_t K,
    int64_t pcols,
    int64_t N,
    int fuse_gelu,
    int max_threads
) {
    if (!g_init_done) vault_tts_cpu_init();
    int64_t n_blocks64 = pcols / 64;
    int64_t n_blocks32 = pcols / 32;
    int nt = choose_omp_threads(M, N, pcols, max_threads > 0 ? max_threads : 1);

    int8_t *qx = (int8_t *)aligned_alloc(64, ((size_t)(M * pcols) + 63u) & ~63u);
    float *dx = (float *)aligned_alloc(64, ((size_t)(M * n_blocks32 * (int64_t)sizeof(float)) + 63u) & ~63u);
    quantize_act_q8_g32(x, qx, dx, M, K, pcols, nt);

    int64_t n_pairs = (N + 1) / 2;

    #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int64_t np = 0; np < n_pairs; np++) {
        int64_t n0 = np * 2;
        int64_t n1 = (n0 + 1 < N) ? (n0 + 1) : n0;
        const uint8_t *w0_row = q_simd + n0 * (pcols / 4);
        const uint8_t *w1_row = q_simd + n1 * (pcols / 4);
        const uint16_t *d0_row = d_f16 + n0 * n_blocks64;
        const uint16_t *d1_row = d_f16 + n1 * n_blocks64;
        float b0 = bias ? bias[n0] : 0.0f;
        float b1 = bias ? bias[n1] : 0.0f;

#ifdef __AVX2__
        const __m256i m03 = _mm256_set1_epi8(0x03);
        const __m256i m2 = _mm256_set1_epi8(2);
        const __m256i ones16 = _mm256_set1_epi16(1);

        int64_t m = 0;
        for (; m + 4 <= M; m += 4) {
            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

            const int8_t *qx0 = qx + (m + 0) * pcols;
            const int8_t *qx1 = qx + (m + 1) * pcols;
            const int8_t *qx2 = qx + (m + 2) * pcols;
            const int8_t *qx3 = qx + (m + 3) * pcols;
            const float *dx0 = dx + (m + 0) * n_blocks32;
            const float *dx1 = dx + (m + 1) * n_blocks32;
            const float *dx2 = dx + (m + 2) * n_blocks32;
            const float *dx3 = dx + (m + 3) * n_blocks32;

            for (int64_t b = 0; b < n_blocks64; b++) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0_row + b * 16));
                __m256i w0_a = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 2), 1), m03), m2);
                __m256i w0_b = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_srli_epi16(r0, 4)), _mm_srli_epi16(r0, 6), 1), m03), m2);
                __m256i aw0_a = _mm256_abs_epi8(w0_a), aw0_b = _mm256_abs_epi8(w0_b);

                __m128i r1 = _mm_loadu_si128((const __m128i *)(w1_row + b * 16));
                __m256i w1_a = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 2), 1), m03), m2);
                __m256i w1_b = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_srli_epi16(r1, 4)), _mm_srli_epi16(r1, 6), 1), m03), m2);
                __m256i aw1_a = _mm256_abs_epi8(w1_a), aw1_b = _mm256_abs_epi8(w1_b);

                float dw0 = g_f16_tab[d0_row[b]];
                float dw1 = g_f16_tab[d1_row[b]];
                int64_t b0_32 = 2 * b, b1_32 = 2 * b + 1;
                int64_t off_a = b * 64, off_b = b * 64 + 32;

                /* Half A (cols 0..31) */
                __m256i xa0 = _mm256_loadu_si256((const __m256i *)(qx0 + off_a));
                acc00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_a, _mm256_sign_epi8(xa0, w0_a)), ones16)), _mm256_set1_ps(dw0 * dx0[b0_32]), acc00);
                acc01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_a, _mm256_sign_epi8(xa0, w1_a)), ones16)), _mm256_set1_ps(dw1 * dx0[b0_32]), acc01);

                __m256i xa1 = _mm256_loadu_si256((const __m256i *)(qx1 + off_a));
                acc10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_a, _mm256_sign_epi8(xa1, w0_a)), ones16)), _mm256_set1_ps(dw0 * dx1[b0_32]), acc10);
                acc11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_a, _mm256_sign_epi8(xa1, w1_a)), ones16)), _mm256_set1_ps(dw1 * dx1[b0_32]), acc11);

                __m256i xa2 = _mm256_loadu_si256((const __m256i *)(qx2 + off_a));
                acc20 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_a, _mm256_sign_epi8(xa2, w0_a)), ones16)), _mm256_set1_ps(dw0 * dx2[b0_32]), acc20);
                acc21 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_a, _mm256_sign_epi8(xa2, w1_a)), ones16)), _mm256_set1_ps(dw1 * dx2[b0_32]), acc21);

                __m256i xa3 = _mm256_loadu_si256((const __m256i *)(qx3 + off_a));
                acc30 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_a, _mm256_sign_epi8(xa3, w0_a)), ones16)), _mm256_set1_ps(dw0 * dx3[b0_32]), acc30);
                acc31 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_a, _mm256_sign_epi8(xa3, w1_a)), ones16)), _mm256_set1_ps(dw1 * dx3[b0_32]), acc31);

                /* Half B (cols 32..63) */
                __m256i xb0 = _mm256_loadu_si256((const __m256i *)(qx0 + off_b));
                acc00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_b, _mm256_sign_epi8(xb0, w0_b)), ones16)), _mm256_set1_ps(dw0 * dx0[b1_32]), acc00);
                acc01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_b, _mm256_sign_epi8(xb0, w1_b)), ones16)), _mm256_set1_ps(dw1 * dx0[b1_32]), acc01);

                __m256i xb1 = _mm256_loadu_si256((const __m256i *)(qx1 + off_b));
                acc10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_b, _mm256_sign_epi8(xb1, w0_b)), ones16)), _mm256_set1_ps(dw0 * dx1[b1_32]), acc10);
                acc11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_b, _mm256_sign_epi8(xb1, w1_b)), ones16)), _mm256_set1_ps(dw1 * dx1[b1_32]), acc11);

                __m256i xb2 = _mm256_loadu_si256((const __m256i *)(qx2 + off_b));
                acc20 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_b, _mm256_sign_epi8(xb2, w0_b)), ones16)), _mm256_set1_ps(dw0 * dx2[b1_32]), acc20);
                acc21 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_b, _mm256_sign_epi8(xb2, w1_b)), ones16)), _mm256_set1_ps(dw1 * dx2[b1_32]), acc21);

                __m256i xb3 = _mm256_loadu_si256((const __m256i *)(qx3 + off_b));
                acc30 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_b, _mm256_sign_epi8(xb3, w0_b)), ones16)), _mm256_set1_ps(dw0 * dx3[b1_32]), acc30);
                acc31 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_b, _mm256_sign_epi8(xb3, w1_b)), ones16)), _mm256_set1_ps(dw1 * dx3[b1_32]), acc31);
            }

            float v00 = hsum256_ps(acc00) + b0, v10 = hsum256_ps(acc10) + b0, v20 = hsum256_ps(acc20) + b0, v30 = hsum256_ps(acc30) + b0;
            if (fuse_gelu) { v00 = fast_gelu_scalar(v00); v10 = fast_gelu_scalar(v10); v20 = fast_gelu_scalar(v20); v30 = fast_gelu_scalar(v30); }
            out[(m + 0) * N + n0] = v00;
            out[(m + 1) * N + n0] = v10;
            out[(m + 2) * N + n0] = v20;
            out[(m + 3) * N + n0] = v30;

            if (n1 != n0) {
                float v01 = hsum256_ps(acc01) + b1, v11 = hsum256_ps(acc11) + b1, v21 = hsum256_ps(acc21) + b1, v31 = hsum256_ps(acc31) + b1;
                if (fuse_gelu) { v01 = fast_gelu_scalar(v01); v11 = fast_gelu_scalar(v11); v21 = fast_gelu_scalar(v21); v31 = fast_gelu_scalar(v31); }
                out[(m + 0) * N + n1] = v01;
                out[(m + 1) * N + n1] = v11;
                out[(m + 2) * N + n1] = v21;
                out[(m + 3) * N + n1] = v31;
            }
        }

        for (; m < M; m++) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            const int8_t *qxm = qx + m * pcols;
            const float *dxm = dx + m * n_blocks32;
            for (int64_t b = 0; b < n_blocks64; b++) {
                __m128i r0 = _mm_loadu_si128((const __m128i *)(w0_row + b * 16));
                __m256i w0_a = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r0), _mm_srli_epi16(r0, 2), 1), m03), m2);
                __m256i w0_b = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_srli_epi16(r0, 4)), _mm_srli_epi16(r0, 6), 1), m03), m2);
                __m256i xa = _mm256_loadu_si256((const __m256i *)(qxm + b * 64));
                __m256i xb = _mm256_loadu_si256((const __m256i *)(qxm + b * 64 + 32));
                float dw0 = g_f16_tab[d0_row[b]];
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0_a), _mm256_sign_epi8(xa, w0_a)), ones16)), _mm256_set1_ps(dw0 * dxm[2 * b]), a0);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0_b), _mm256_sign_epi8(xb, w0_b)), ones16)), _mm256_set1_ps(dw0 * dxm[2 * b + 1]), a0);
                if (n1 != n0) {
                    __m128i r1 = _mm_loadu_si128((const __m128i *)(w1_row + b * 16));
                    __m256i w1_a = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(r1), _mm_srli_epi16(r1, 2), 1), m03), m2);
                    __m256i w1_b = _mm256_sub_epi8(_mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_srli_epi16(r1, 4)), _mm_srli_epi16(r1, 6), 1), m03), m2);
                    float dw1 = g_f16_tab[d1_row[b]];
                    a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1_a), _mm256_sign_epi8(xa, w1_a)), ones16)), _mm256_set1_ps(dw1 * dxm[2 * b]), a1);
                    a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1_b), _mm256_sign_epi8(xb, w1_b)), ones16)), _mm256_set1_ps(dw1 * dxm[2 * b + 1]), a1);
                }
            }
            float v0 = hsum256_ps(a0) + b0;
            out[m * N + n0] = fuse_gelu ? fast_gelu_scalar(v0) : v0;
            if (n1 != n0) {
                float v1 = hsum256_ps(a1) + b1;
                out[m * N + n1] = fuse_gelu ? fast_gelu_scalar(v1) : v1;
            }
        }
#else
        for (int64_t m = 0; m < M; m++) {
            const int8_t *qxm = qx + m * pcols;
            const float *dxm = dx + m * n_blocks32;
            float s0 = b0, s1 = b1;
            for (int64_t b = 0; b < n_blocks64; b++) {
                int32_t ia0 = 0, ib0 = 0, ia1 = 0, ib1 = 0;
                const uint8_t *p0 = w0_row + b * 16;
                const uint8_t *p1 = w1_row + b * 16;
                const int8_t *xa = qxm + b * 64;
                for (int k = 0; k < 16; k++) {
                    ia0 += ((int)(p0[k] & 3u) - 2) * xa[k] + ((int)((p0[k] >> 2) & 3u) - 2) * xa[k + 16];
                    ib0 += ((int)((p0[k] >> 4) & 3u) - 2) * xa[k + 32] + ((int)((p0[k] >> 6) & 3u) - 2) * xa[k + 48];
                    ia1 += ((int)(p1[k] & 3u) - 2) * xa[k] + ((int)((p1[k] >> 2) & 3u) - 2) * xa[k + 16];
                    ib1 += ((int)((p1[k] >> 4) & 3u) - 2) * xa[k + 32] + ((int)((p1[k] >> 6) & 3u) - 2) * xa[k + 48];
                }
                s0 += g_f16_tab[d0_row[b]] * ((float)ia0 * dxm[2 * b] + (float)ib0 * dxm[2 * b + 1]);
                s1 += g_f16_tab[d1_row[b]] * ((float)ia1 * dxm[2 * b] + (float)ib1 * dxm[2 * b + 1]);
            }
            out[m * N + n0] = fuse_gelu ? fast_gelu_scalar(s0) : s0;
            if (n1 != n0) out[m * N + n1] = fuse_gelu ? fast_gelu_scalar(s1) : s1;
        }
#endif
    }

    free(qx);
    free(dx);
}

/* ============================================================================
 * 5. Cache-Resident Packed 1D Convolution (Conv1d, groups == 1) for VITS & TTS
 * ========================================================================== */
void vault_conv1d_packed(
    const float *x,          /* [B, C_in, T_in] */
    const uint8_t *q_simd,   /* [C_out, pcols/(2 or 4)] */
    const uint16_t *d_f16,
    const float *bias,       /* [C_out] or NULL */
    float *out,              /* [B, C_out, T_out] */
    int64_t B,
    int64_t C_in,
    int64_t T_in,
    int64_t C_out,
    int64_t K_w,
    int64_t stride,
    int64_t pad,
    int64_t dilation,
    int64_t pcols,
    int is_q2,
    int max_threads
) {
    int64_t T_out = (T_in + 2 * pad - dilation * (K_w - 1) - 1) / stride + 1;
    int64_t M = B * T_out;
    int64_t K_col = C_in * K_w;
    int nt = choose_omp_threads(M, C_out, pcols, max_threads > 0 ? max_threads : 1);

    float *col_buf = (float *)aligned_alloc(64, ((size_t)(M * K_col * (int64_t)sizeof(float)) + 63u) & ~63u);
    float *gemm_out = (float *)aligned_alloc(64, ((size_t)(M * C_out * (int64_t)sizeof(float)) + 63u) & ~63u);

    /* Parallel im2col -> [B * T_out, C_in * K_w] */
    #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1 && M >= 32)
    for (int64_t idx = 0; idx < M; idx++) {
        int64_t b = idx / T_out;
        int64_t to = idx % T_out;
        int64_t t_base = to * stride - pad;
        const float *xb = x + b * C_in * T_in;
        float *dst = col_buf + idx * K_col;
        if (K_w == 1) {
            for (int64_t c = 0; c < C_in; c++) {
                dst[c] = xb[c * T_in + t_base];
            }
        } else {
            for (int64_t c = 0; c < C_in; c++) {
                const float *xbc = xb + c * T_in;
                float *dc = dst + c * K_w;
                for (int64_t kw = 0; kw < K_w; kw++) {
                    int64_t ti = t_base + kw * dilation;
                    dc[kw] = (ti >= 0 && ti < T_in) ? xbc[ti] : 0.0f;
                }
            }
        }
    }

    if (is_q2) {
        vault_gemm_q2_g64(col_buf, q_simd, d_f16, bias, gemm_out, M, K_col, pcols, C_out, 0, nt);
    } else {
        vault_gemm_p4_g32(col_buf, q_simd, d_f16, bias, gemm_out, M, K_col, pcols, C_out, 0, nt);
    }

    /* Transpose [B, T_out, C_out] -> [B, C_out, T_out] */
    #pragma omp parallel for num_threads(nt) collapse(2) schedule(static) if (nt > 1 && B * C_out >= 32)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t co = 0; co < C_out; co++) {
            float *dst_co = out + (b * C_out + co) * T_out;
            const float *src_b = gemm_out + b * T_out * C_out + co;
            for (int64_t to = 0; to < T_out; to++) {
                dst_co[to] = src_b[to * C_out];
            }
        }
    }

    free(col_buf);
    free(gemm_out);
}

/* ============================================================================
 * 6. Fused In-Place AVX2 Rotary Position Embedding (VADDSUBPS: 4 instructions / 8 floats!)
 * ========================================================================== */
void vault_rope_inplace_f32(
    float *q,           /* [B, H, N, D] contiguous */
    float *k,           /* [B, H, N, D] contiguous */
    const float *cos_t, /* [N, D] contiguous */
    const float *sin_t, /* [N, D] contiguous */
    int64_t B,
    int64_t H_rot,      /* heads to rotate (usually all H) */
    int64_t H_total,
    int64_t N,
    int64_t D,
    int max_threads
) {
    int nt = (B * H_rot * N >= 512 && max_threads > 1) ? max_threads : 1;
    #pragma omp parallel for num_threads(nt) collapse(2) schedule(static) if (nt > 1)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H_rot; h++) {
            float *q_bh = q + (b * H_total + h) * N * D;
            float *k_bh = k + (b * H_total + h) * N * D;
            for (int64_t n = 0; n < N; n++) {
                float *qn = q_bh + n * D;
                float *kn = k_bh + n * D;
                const float *cn = cos_t + n * D;
                const float *sn = sin_t + n * D;
                int64_t d = 0;
#ifdef __AVX2__
                for (; d + 8 <= D; d += 8) {
                    __m256 vc = _mm256_loadu_ps(cn + d);
                    __m256 vs = _mm256_loadu_ps(sn + d);
                    __m256 vq = _mm256_loadu_ps(qn + d);
                    __m256 vqs = _mm256_permute_ps(vq, 0xB1);
                    _mm256_storeu_ps(qn + d, _mm256_addsub_ps(_mm256_mul_ps(vq, vc), _mm256_mul_ps(vqs, vs)));

                    __m256 vk = _mm256_loadu_ps(kn + d);
                    __m256 vks = _mm256_permute_ps(vk, 0xB1);
                    _mm256_storeu_ps(kn + d, _mm256_addsub_ps(_mm256_mul_ps(vk, vc), _mm256_mul_ps(vks, vs)));
                }
#endif
                for (; d + 2 <= D; d += 2) {
                    float c0 = cn[d], c1 = cn[d + 1], s0 = sn[d], s1 = sn[d + 1];
                    float q0 = qn[d], q1 = qn[d + 1];
                    qn[d] = q0 * c0 - q1 * s0;
                    qn[d + 1] = q1 * c1 + q0 * s1;
                    float k0 = kn[d], k1 = kn[d + 1];
                    kn[d] = k0 * c0 - k1 * s0;
                    kn[d + 1] = k1 * c1 + k0 * s1;
                }
            }
        }
    }
}

/* ============================================================================
 * 7. Sub-Millisecond AVX2 L3-Cache Scratch Unpackers (for Compute-Bound MKL GEMM/Conv1d)
 * ========================================================================== */
void vault_unpack_p4_f32(
    const uint8_t *q_simd,
    const uint16_t *d_f16,
    float *out,
    int64_t rows,
    int64_t cols,
    int64_t pcols,
    int max_threads
) {
    if (!g_init_done) vault_tts_cpu_init();
    int64_t nb = pcols / 32;
    int nt = (rows * pcols >= 524288 && max_threads > 1) ? max_threads : 1;
    static const int8_t p4_lut_arr[16] = {
        -127, -101, -80, -62, -46, -33, -21, -10, 0, 10, 21, 33, 46, 62, 80, 101
    };

    #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *qr = q_simd + r * (pcols / 2);
        const uint16_t *dr = d_f16 + r * nb;
        float * dst = out + r * cols;
#ifdef __AVX2__
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i vlut = _mm_loadu_si128((const __m128i *)p4_lut_arr);
#endif
        for (int64_t b = 0; b < nb; b++) {
            int64_t c0 = b * 32;
            float sc = g_f16_tab[dr[b]];
            if (c0 + 32 <= cols) {
#ifdef __AVX2__
                __m256 vsc = _mm256_set1_ps(sc);
                __m128i raw = _mm_loadu_si128((const __m128i *)(qr + b * 16));
                __m128i lo = _mm_shuffle_epi8(vlut, _mm_and_si128(raw, m0f));
                __m128i hi = _mm_shuffle_epi8(vlut, _mm_and_si128(_mm_srli_epi16(raw, 4), m0f));
                _mm256_storeu_ps(dst + c0 +  0, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), vsc));
                _mm256_storeu_ps(dst + c0 +  8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), vsc));
                _mm256_storeu_ps(dst + c0 + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), vsc));
                _mm256_storeu_ps(dst + c0 + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), vsc));
#else
                const uint8_t *p = qr + b * 16;
                for (int k = 0; k < 16; k++) {
                    dst[c0 + k] = (float)p4_lut_arr[p[k] & 0x0Fu] * sc;
                    dst[c0 + k + 16] = (float)p4_lut_arr[p[k] >> 4] * sc;
                }
#endif
            } else {
                const uint8_t *p = qr + b * 16;
                for (int k = 0; k < 16; k++) {
                    if (c0 + k < cols) dst[c0 + k] = (float)p4_lut_arr[p[k] & 0x0Fu] * sc;
                    if (c0 + k + 16 < cols) dst[c0 + k + 16] = (float)p4_lut_arr[p[k] >> 4] * sc;
                }
            }
        }
    }
}

void vault_unpack_q2_f32(
    const uint8_t *q_simd,
    const uint16_t *d_f16,
    float *out,
    int64_t rows,
    int64_t cols,
    int64_t pcols,
    int max_threads
) {
    if (!g_init_done) vault_tts_cpu_init();
    int64_t nb = pcols / 64;
    int nt = (rows * pcols >= 524288 && max_threads > 1) ? max_threads : 1;

    #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *qr = q_simd + r * (pcols / 4);
        const uint16_t *dr = d_f16 + r * nb;
        float *dst = out + r * cols;
#ifdef __AVX2__
        const __m128i m03 = _mm_set1_epi8(0x03);
        const __m128i m2 = _mm_set1_epi8(2);
#endif
        for (int64_t b = 0; b < nb; b++) {
            int64_t c0 = b * 64;
            float sc = g_f16_tab[dr[b]];
            if (c0 + 64 <= cols) {
#ifdef __AVX2__
                __m256 vsc = _mm256_set1_ps(sc);
                __m128i raw = _mm_loadu_si128((const __m128i *)(qr + b * 16));
                __m128i p0 = _mm_sub_epi8(_mm_and_si128(raw, m03), m2);
                __m128i p1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 2), m03), m2);
                __m128i p2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), m03), m2);
                __m128i p3 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 6), m03), m2);
                _mm256_storeu_ps(dst + c0 +  0, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p0)), vsc));
                _mm256_storeu_ps(dst + c0 +  8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(p0, 8))), vsc));
                _mm256_storeu_ps(dst + c0 + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p1)), vsc));
                _mm256_storeu_ps(dst + c0 + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(p1, 8))), vsc));
                _mm256_storeu_ps(dst + c0 + 32, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p2)), vsc));
                _mm256_storeu_ps(dst + c0 + 40, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(p2, 8))), vsc));
                _mm256_storeu_ps(dst + c0 + 48, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p3)), vsc));
                _mm256_storeu_ps(dst + c0 + 56, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(p3, 8))), vsc));
#else
                const uint8_t *p = qr + b * 16;
                for (int k = 0; k < 16; k++) {
                    dst[c0 + k]      = (float)((int)(p[k] & 3u) - 2) * sc;
                    dst[c0 + k + 16] = (float)((int)((p[k] >> 2) & 3u) - 2) * sc;
                    dst[c0 + k + 32] = (float)((int)((p[k] >> 4) & 3u) - 2) * sc;
                    dst[c0 + k + 48] = (float)((int)((p[k] >> 6) & 3u) - 2) * sc;
                }
#endif
            } else {
                const uint8_t *p = qr + b * 16;
                for (int k = 0; k < 16; k++) {
                    if (c0 + k < cols)      dst[c0 + k]      = (float)((int)(p[k] & 3u) - 2) * sc;
                    if (c0 + k + 16 < cols) dst[c0 + k + 16] = (float)((int)((p[k] >> 2) & 3u) - 2) * sc;
                    if (c0 + k + 32 < cols) dst[c0 + k + 32] = (float)((int)((p[k] >> 4) & 3u) - 2) * sc;
                    if (c0 + k + 48 < cols) dst[c0 + k + 48] = (float)((int)((p[k] >> 6) & 3u) - 2) * sc;
                }
            }
        }
    }
}

/* ============================================================================
 * 8. Fused Single-Pass Band-Limited Harmonic Pulse Train (for Tiny-Voca)
 * ========================================================================== */
void vault_pulse_train_f32(
    const float *f0,         /* [B, frames] */
    const float *phase_in,   /* [B] or NULL */
    float *out,              /* [B, frames * hop] */
    int64_t B,
    int64_t frames,
    int64_t hop,
    double sample_rate,
    double limit
) {
    const double two_pi = 6.283185307179586476925286766559;
    const double inv_sr_2pi = two_pi / sample_rate;
    const double inv_hop = 1.0 / (double)hop;
    const double max_nyq = limit * sample_rate;

    for (int64_t b = 0; b < B; b++) {
        const float *fb = f0 + b * frames;
        float *dst = out + b * frames * hop;
        double theta = (phase_in != NULL) ? (double)phase_in[b] : 0.0;
        double k0 = log((double)(fb[0] > 1.0f ? fb[0] : 1.0f));
        for (int64_t f = 0; f < frames; f++) {
            float f_next = (f + 1 < frames) ? fb[f + 1] : fb[f];
            double k1 = log((double)(f_next > 1.0f ? f_next : 1.0f));
            double dk = (k1 - k0) * inv_hop;
            double freq = exp(k0);
            double freq_mul = exp(dk);
            int64_t base = f * hop;
            for (int64_t j = 0; j < hop; j++) {
                theta += freq * inv_sr_2pi;
                if (theta >= two_pi || theta < 0.0) {
                    theta = fmod(theta, two_pi);
                    if (theta < 0.0) theta += two_pi;
                }
                double harm = max_nyq / freq;
                if (harm < 1.0) harm = 1.0;
                double whole = floor(harm);
                float th_f = (float)theta;
                float denom = 2.0f * sinf(0.5f * th_f);
                float core = (fabsf(denom) < 1e-7f)
                    ? (float)whole
                    : (sinf(((float)whole + 0.5f) * th_f) / denom - 0.5f);
                float wave = core + (float)(harm - whole) * cosf(((float)whole + 1.0f) * th_f);
                dst[base + j] = wave * sqrtf(2.0f / (float)harm);
                freq *= freq_mul;
            }
            k0 = k1;
        }
    }
}


