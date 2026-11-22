/*
 * Zero-libc Freestanding AArch64 NEON vs. x86_64 AVX2 End-to-End Execution Parity Harness.
 * Compiles cleanly for BOTH:
 *   1) ARM64 NEON freestanding Linux binary (zero glibc required, runs under qemu-aarch64-static):
 *      clang --target=aarch64-linux-gnu -static -O3 -ffreestanding -nostdlib -fuse-ld=lld -DV3_BAREMETAL_LINUX -Wall -Wextra -Werror \
 *            runtime/vault3.c benchmarks/test_arm64_neon_parity.c -o benchmarks/test_arm64_neon_aarch64
 *   2) x86_64 AVX2+FMA native binary:
 *      cc -O3 -mavx2 -mfma -mf16c runtime/vault3.c benchmarks/test_arm64_neon_parity.c -lm -o benchmarks/test_arm64_neon_x86
 *
 * Loads a real 3B `.vlt3` file (e.g. `vault_v3_3b_p4_0_primal.vlt3` exercising V3_P4_0 `vqtbl1q_s8`,
 * V3_Q8_0, V3_F32, RoPE, RMSNorm, SwiGLU, GQA), runs ChatML BPE tokenization + 36-layer prefill + 8
 * greedy decode steps, and prints exact fixed-point (1e-6) logits & token IDs for 1:1 parity diffing.
 */

#include "../runtime/vault3.h"

#if defined(__aarch64__) && defined(V3_BAREMETAL_LINUX)

/* Linux AArch64 syscall numbers */
#define SYS_openat     56
#define SYS_close      57
#define SYS_write      64
#define SYS_lseek      62
#define SYS_exit_group 94
#define SYS_mmap       222

#define AT_FDCWD       (-100)
#define O_RDONLY       0
#define PROT_READ      1
#define PROT_WRITE     2
#define MAP_PRIVATE    2
#define MAP_ANONYMOUS  0x20

static inline long sc6(long n, long a0, long a1, long a2, long a3, long a4, long a5) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a0;
    register long x1 __asm__("x1") = a1;
    register long x2 __asm__("x2") = a2;
    register long x3 __asm__("x3") = a3;
    register long x4 __asm__("x4") = a4;
    register long x5 __asm__("x5") = a5;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5) : "memory", "cc");
    return x0;
}

static void os_write(int fd, const void *buf, size_t len) {
    sc6(SYS_write, fd, (long)buf, (long)len, 0, 0, 0);
}

static void os_exit(int code) {
    sc6(SYS_exit_group, code, 0, 0, 0, 0, 0);
    __builtin_unreachable();
}

static void *os_mmap_anon(size_t sz) {
    long r = sc6(SYS_mmap, 0, (long)sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r < 0 && r > -4096) return 0;
    return (void *)r;
}

static const uint8_t *os_mmap_file(const char *path, uint64_t *out_sz) {
    long fd = sc6(SYS_openat, AT_FDCWD, (long)path, O_RDONLY, 0, 0, 0);
    if (fd < 0) return 0;
    long sz = sc6(SYS_lseek, fd, 0, 2 /* SEEK_END */, 0, 0, 0);
    if (sz <= 0) return 0;
    long addr = sc6(SYS_mmap, 0, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    sc6(SYS_close, fd, 0, 0, 0, 0, 0);
    if (addr < 0 && addr > -4096) return 0;
    *out_sz = (uint64_t)sz;
    return (const uint8_t *)addr;
}

/* Clang may emit calls to memcpy/memset for struct initialization even in -ffreestanding */
void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)c;
    return dst;
}

#else

#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void os_write(int fd, const void *buf, size_t len) {
    ssize_t r = write(fd, buf, len);
    (void)r;
}
static void *os_mmap_anon(size_t sz) {
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? 0 : p;
}
static const uint8_t *os_mmap_file(const char *path, uint64_t *out_sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return 0;
    *out_sz = (uint64_t)st.st_size;
    return (const uint8_t *)p;
}

#endif

static size_t c_strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

static void put_str(const char *s) { os_write(1, s, c_strlen(s)); }

static void put_i64(int64_t v) {
    char buf[32];
    int pos = 31;
    uint64_t u;
    if (v < 0) {
        put_str("-");
        u = (uint64_t)(-(v + 1)) + 1u;
    } else {
        u = (uint64_t)v;
    }
    if (u == 0) { put_str("0"); return; }
    while (u > 0) {
        buf[pos--] = (char)('0' + (u % 10u));
        u /= 10u;
    }
    os_write(1, &buf[pos + 1], (size_t)(31 - pos));
}

static void put_f6(float f) {
    if (f < 0.0f) { put_str("-"); f = -f; }
    int64_t scaled = (int64_t)(f * 1000000.0f + 0.5f);
    put_i64(scaled / 1000000);
    put_str(".");
    int64_t rem = scaled % 1000000;
    char frac[6];
    for (int i = 5; i >= 0; i--) {
        frac[i] = (char)('0' + (rem % 10));
        rem /= 10;
    }
    os_write(1, frac, 6);
}

static int run_parity(int argc, char **argv) {
    if (argc < 2) {
        put_str("ERR usage\n");
        return 2;
    }
    uint64_t sz = 0;
    const uint8_t *buf = os_mmap_file(argv[1], &sz);
    if (!buf) { put_str("ERR mmap\n"); return 3; }

    static v3_model m;
    static v3_state s;
    static v3_tokenizer tok;
    int rc = v3_load(&m, buf, sz, 0);
    if (rc != V3_OK) { put_str("ERR v3_load="); put_i64(rc); put_str("\n"); return 4; }

    size_t sa_sz = v3_state_arena_size(&m) + 4096u;
    uint8_t *sa_raw = (uint8_t *)os_mmap_anon(sa_sz);
    void *sa = (void *)(((uintptr_t)sa_raw + 63u) & ~(uintptr_t)63u);
    rc = v3_state_init(&s, &m, sa, sa_sz - 64u);
    if (rc != V3_OK) { put_str("ERR state_init\n"); return 5; }

    size_t ta_sz = v3_tok_arena_size(&m) + 4096u;
    uint8_t *ta_raw = (uint8_t *)os_mmap_anon(ta_sz);
    void *ta = (void *)(((uintptr_t)ta_raw + 63u) & ~(uintptr_t)63u);
    rc = v3_tok_init(&tok, &m, ta, ta_sz - 64u);
    if (rc != V3_OK) { put_str("ERR tok_init\n"); return 6; }

    const char *prompt = "<|im_start|>user\nWhat is 12 * 12? Reply concisely.<|im_end|>\n<|im_start|>assistant\n";
    static uint32_t ids[256];
    size_t n_ids = 0;
    rc = v3_encode(&tok, (const uint8_t *)prompt, c_strlen(prompt), ids, 256, &n_ids);
    if (rc != V3_OK || n_ids == 0) { put_str("ERR encode\n"); return 7; }

    const float *lg = 0;
    for (size_t i = 0; i < n_ids; i++) {
        lg = v3_step(&s, ids[i]);
    }

#if defined(__aarch64__)
    put_str("ARCH=aarch64_neon");
#else
    put_str("ARCH=x86_64_avx2");
#endif
    put_str(" N_PROMPT="); put_i64((int64_t)n_ids);
    /* Print 8 deterministic probe logits after full 36-layer prefill */
    uint32_t probes[8] = {0, 16, 17, 19, 220, 151643, 151645, v3_argmax(lg, m.vocab_size)};
    put_str(" LOGITS=");
    for (int i = 0; i < 8; i++) {
        if (i) put_str(",");
        put_i64(probes[i]); put_str(":"); put_f6(lg[probes[i]]);
    }
    /* Greedy decode 10 tokens */
    put_str(" GEN_IDS=");
    uint32_t gen[10];
    for (int step = 0; step < 10; step++) {
        uint32_t nx = v3_argmax(lg, m.vocab_size);
        gen[step] = nx;
        if (step) put_str(",");
        put_i64(nx);
        if (step + 1 < 10) lg = v3_step(&s, nx);
    }
    put_str(" TEXT=\"");
    for (int step = 0; step < 10; step++) {
        const uint8_t *tb = 0;
        uint32_t tl = 0;
        if (v3_token_bytes(&m, gen[step], &tb, &tl) == V3_OK && tl > 0) {
            os_write(1, tb, tl);
        }
    }
    put_str("\"\n");
    return 0;
}

#if defined(__aarch64__) && defined(V3_BAREMETAL_LINUX)
__attribute__((naked)) void _start(void) {
    __asm__ volatile(
        "mov x0, sp\n"
        "bl _c_entry\n"
    );
}
void _c_entry(long *sp) {
    int argc = (int)sp[0];
    char **argv = (char **)&sp[1];
    int rc = run_parity(argc, argv);
    os_exit(rc);
}
#else
int main(int argc, char **argv) {
    return run_parity(argc, argv);
}
#endif
