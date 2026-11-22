/* SPDX-License-Identifier: Apache-2.0
 * Freestanding C99 High-Eviction-Pressure 8,465-Token 26-Needle Verification Harness.
 * Links directly against runtime/vault3.c (0 libc/libm calls inside vault3.c).
 *
 * Prefills the 8,440-token 26-needle prefix document ONCE (pushing ~7,416 tokens into
 * Segment Memory with seg_budget=1024 -> 86.2% eviction pressure!), snapshots the fixed
 * 398.55 MB v3_state arena at its identical virtual address, and evaluates all 26
 * individual needle queries sequentially in ~15 seconds.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#include "../runtime/vault3.h"

#define MAX_WORKERS 64

typedef struct {
    int n_threads;
    uint32_t spin_limit;
    pthread_t th[MAX_WORKERS];
    v3_task_fn fn;
    void *arg;
    uint32_t n, grain;
    atomic_uint gen, next, done, sleeping, main_waiting;
    atomic_int stop;
} thread_pool;

typedef struct { thread_pool *p; int id; } worker_arg;

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static inline void futex_wait_u32(atomic_uint *addr, uint32_t expected) {
#ifdef __linux__
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
#else
    (void)addr; (void)expected;
    struct timespec ts = {0, 10000};
    nanosleep(&ts, NULL);
#endif
}

static inline void futex_wake_u32(atomic_uint *addr, int count) {
#ifdef __linux__
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAKE_PRIVATE, count, NULL, NULL, 0);
#else
    (void)addr; (void)count;
#endif
}

static void run_work(thread_pool *p) {
    uint32_t g = p->grain, n = p->n;
    for (;;) {
        uint32_t i0 = atomic_fetch_add_explicit(&p->next, g, memory_order_relaxed);
        if (i0 >= n) break;
        uint32_t i1 = i0 + g < n ? i0 + g : n;
        p->fn(p->arg, i0, i1);
    }
}

static void *worker_main(void *raw) {
    worker_arg wa = *(worker_arg *)raw;
    free(raw);
    thread_pool *p = wa.p;
    uint32_t seen = 0, want = (uint32_t)(p->n_threads - 1), slim = p->spin_limit;
    for (;;) {
        uint32_t spins = 0, g;
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
            if (spins < slim) { spins++; cpu_relax(); }
            else {
                atomic_fetch_add_explicit(&p->sleeping, 1u, memory_order_seq_cst);
                if (atomic_load_explicit(&p->gen, memory_order_seq_cst) == seen &&
                    !atomic_load_explicit(&p->stop, memory_order_seq_cst))
                    futex_wait_u32(&p->gen, seen);
                atomic_fetch_sub_explicit(&p->sleeping, 1u, memory_order_relaxed);
            }
        }
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
        seen = g;
        run_work(p);
        uint32_t prev = atomic_fetch_add_explicit(&p->done, 1u, memory_order_seq_cst);
        if (prev + 1u == want && atomic_load_explicit(&p->main_waiting, memory_order_seq_cst))
            futex_wake_u32(&p->done, 1);
    }
}

static void pool_parallel_for(void *ctx, v3_task_fn fn, void *arg, uint32_t n) {
    thread_pool *p = (thread_pool *)ctx;
    if (p->n_threads <= 1 || n == 0) { fn(arg, 0, n); return; }
    uint32_t g = n / ((uint32_t)p->n_threads * 4u);
    uint32_t min_g = (n >= 256u) ? 16u : 1u;
    if (g < min_g) g = min_g;
    p->fn = fn; p->arg = arg; p->n = n; p->grain = g;
    atomic_store_explicit(&p->next, 0u, memory_order_relaxed);
    atomic_store_explicit(&p->done, 0u, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1u, memory_order_seq_cst);
    if (atomic_load_explicit(&p->sleeping, memory_order_seq_cst) > 0u)
        futex_wake_u32(&p->gen, INT_MAX);
    run_work(p);
    uint32_t want = (uint32_t)(p->n_threads - 1), wspins = 0, slim = p->spin_limit, dcur;
    while ((dcur = atomic_load_explicit(&p->done, memory_order_acquire)) < want) {
        if (wspins < slim) { wspins++; cpu_relax(); }
        else {
            atomic_store_explicit(&p->main_waiting, 1u, memory_order_seq_cst);
            dcur = atomic_load_explicit(&p->done, memory_order_seq_cst);
            if (dcur < want) futex_wait_u32(&p->done, dcur);
            atomic_store_explicit(&p->main_waiting, 0u, memory_order_relaxed);
        }
    }
}

static void pool_init(thread_pool *p, int n_threads) {
    memset(p, 0, sizeof(*p));
    if (n_threads > MAX_WORKERS) n_threads = MAX_WORKERS;
    if (n_threads < 1) n_threads = 1;
    long onln = sysconf(_SC_NPROCESSORS_ONLN);
    long phys = (onln >= 4) ? (onln / 2) : (onln > 0 ? onln : 1);
    p->n_threads = n_threads;
    p->spin_limit = (n_threads <= (int)phys) ? 4096u : 32u;
    for (int i = 1; i < n_threads; i++) {
        worker_arg *wa = malloc(sizeof(*wa));
        wa->p = p; wa->id = i;
        if (pthread_create(&p->th[i], NULL, worker_main, wa) != 0) exit(1);
    }
}

static void pool_destroy(thread_pool *p) {
    atomic_store_explicit(&p->stop, 1, memory_order_seq_cst);
    atomic_fetch_add_explicit(&p->gen, 1u, memory_order_seq_cst);
    futex_wake_u32(&p->gen, INT_MAX);
    for (int i = 1; i < p->n_threads; i++) pthread_join(p->th[i], NULL);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double peak_rss_mb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
    return ru.ru_maxrss / 1024.0;
}

static char *read_text(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1u);
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    *len = (size_t)n;
    return b;
}

typedef struct { uint32_t *data; size_t n, cap; } id_vec;

static void id_vec_push(id_vec *v, uint32_t id) {
    if (v->n >= v->cap) {
        size_t ncap = v->cap ? v->cap * 2u : 256u;
        v->data = realloc(v->data, ncap * sizeof(uint32_t));
        v->cap = ncap;
    }
    v->data[v->n++] = id;
}

static void id_vec_encode(id_vec *v, v3_tokenizer *tk, const char *s, size_t len, int allow_special) {
    size_t need = len + 32u;
    for (;;) {
        if (v->n + need > v->cap) {
            size_t ncap = v->cap ? v->cap : 256u;
            while (ncap < v->n + need) ncap *= 2u;
            v->data = realloc(v->data, ncap * sizeof(uint32_t));
            v->cap = ncap;
        }
        size_t got = 0;
        int rc = v3_encode_ex(tk, (const uint8_t *)s, len, v->data + v->n, v->cap - v->n, &got, allow_special);
        if (rc == V3_OK) { v->n += got; return; }
        need *= 2u;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.vlt3 doc_prefix.txt queries.tsv [--threads 24] [--noseg]\n", argv[0]);
        return 2;
    }
    const char *vlt_path = argv[1];
    const char *doc_path = argv[2];
    const char *tsv_path = argv[3];
    int n_threads = 24;
    int noseg = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noseg")) noseg = 1;
    }

    int fd = open(vlt_path, O_RDONLY);
    if (fd < 0) { perror("open vlt3"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *buf = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }

    v3_model m;
    if (v3_load(&m, buf, (uint64_t)st.st_size, 0) != V3_OK) {
        fprintf(stderr, "v3_load failed\n");
        return 1;
    }

    size_t st_sz = v3_state_arena_size(&m);
    size_t tok_sz = v3_tok_arena_size(&m);
    void *arena = aligned_alloc(64u, ((st_sz + 63u) / 64u) * 64u);
    void *arena_bak = aligned_alloc(64u, ((st_sz + 63u) / 64u) * 64u);
    void *tok_arena = aligned_alloc(64u, ((tok_sz + 63u) / 64u) * 64u);
    if (!arena || !arena_bak || !tok_arena) { fprintf(stderr, "arena oom\n"); return 1; }

    v3_state s;
    if (v3_state_init(&s, &m, arena, st_sz) != V3_OK) { fprintf(stderr, "state_init failed\n"); return 1; }
    if (noseg) v3_set_seg_budget(&s, 0u);

    v3_tokenizer tk;
    if (v3_tok_init(&tk, &m, tok_arena, tok_sz) != V3_OK) { fprintf(stderr, "tok_init failed\n"); return 1; }

    thread_pool pool;
    pool_init(&pool, n_threads);
    s.par_for = pool_parallel_for;
    s.par_ctx = &pool;

    size_t doc_len = 0;
    char *doc_txt = read_text(doc_path, &doc_len);
    if (!doc_txt) { fprintf(stderr, "cannot read doc_path\n"); return 1; }

    const char *sys_prompt = "You are a helpful assistant.";
    id_vec pref = {0, 0, 0};
    id_vec_push(&pref, m.im_start_id);
    id_vec_encode(&pref, &tk, "system\n", 7, 0);
    id_vec_encode(&pref, &tk, sys_prompt, strlen(sys_prompt), 0);
    id_vec_push(&pref, m.im_end_id);
    id_vec_encode(&pref, &tk, "\n", 1, 0);
    id_vec_push(&pref, m.im_start_id);
    id_vec_encode(&pref, &tk, "user\n", 5, 0);
    id_vec_encode(&pref, &tk, doc_txt, doc_len, 0);

    fprintf(stderr, "[C99-8K] Prefilling shared 26-needle document (%zu tokens, seg_budget=%u, threads=%d)...\n",
            pref.n, s.seg_budget, n_threads);
    double t0 = now_s();
    for (size_t i = 0; i < pref.n; i++) {
        v3_step(&s, pref.data[i]);
        if ((i + 1) % 2000 == 0 || i + 1 == pref.n) {
            double el = now_s() - t0;
            fprintf(stderr, "  [prefill] %zu/%zu tok (%.1f tok/s, mem_cnt=%u, uniq_segs=%u)\n",
                    i + 1, pref.n, (i + 1) / (el > 0 ? el : 1e-9), s.mem_cnt, s.n_uniq_segs);
        }
    }
    double prefill_s = now_s() - t0;

    /* Snapshot exact post-document C99 state at identical arena virtual address */
    v3_state s_bak = s;
    memcpy(arena_bak, arena, st_sz);

    FILE *qf = fopen(tsv_path, "r");
    if (!qf) { perror("open tsv"); return 1; }
    char line[2048];
    int total_q = 0, hits = 0;
    double q_time = 0.0;
    size_t last_total_toks = pref.n;
    uint32_t final_mem_cnt = s.mem_cnt, final_uniq_segs = s.n_uniq_segs;

    while (fgets(line, sizeof(line), qf)) {
        size_t L = strlen(line);
        while (L > 0 && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        const char *entity = line;
        const char *expected = tab + 1;

        s = s_bak;
        memcpy(arena, arena_bak, st_sz);

        char q_str[512];
        int qn = snprintf(q_str, sizeof(q_str), "\n\nWhat is the exact telemetry cipher stored by %s?", entity);
        id_vec qids = {0, 0, 0};
        id_vec_encode(&qids, &tk, q_str, (size_t)qn, 0);
        id_vec_push(&qids, m.im_end_id);
        id_vec_encode(&qids, &tk, "\n", 1, 0);
        id_vec_push(&qids, m.im_start_id);
        id_vec_encode(&qids, &tk, "assistant\n", 10, 0);

        double qt0 = now_s();
        const float *lg = NULL;
        for (size_t j = 0; j < qids.n; j++) lg = v3_step(&s, qids.data[j]);
        last_total_toks = pref.n + qids.n;
        final_mem_cnt = s.mem_cnt;
        final_uniq_segs = s.n_uniq_segs;

        char reply[1024];
        size_t rpos = 0;
        for (int g = 0; g < 32 && lg != NULL; g++) {
            uint32_t nx = v3_argmax(lg, m.vocab_size);
            if (nx == m.im_end_id || nx == m.eos_id) break;
            const uint8_t *tb; uint32_t tl;
            if (v3_token_bytes(&m, nx, &tb, &tl) == V3_OK && rpos + tl + 1 < sizeof(reply)) {
                memcpy(reply + rpos, tb, tl);
                rpos += tl;
            }
            lg = v3_step(&s, nx);
        }
        reply[rpos] = 0;
        q_time += (now_s() - qt0);

        int ok = (strstr(reply, expected) != NULL) ? 1 : 0;
        hits += ok;
        total_q++;
        fprintf(stderr, "  [Q%02d/26] %s -> expected=%s | %s | reply=\"%s\"\n",
                total_q, entity, expected, ok ? "HIT" : "MISS", reply);
        free(qids.data);
    }
    fclose(qf);

    double cand_toks = (last_total_toks > m.window + s.n_pin) ? (double)(last_total_toks - m.window - s.n_pin) : 1.0;
    double surv_pct = 100.0 * (double)final_mem_cnt / cand_toks;
    double evict_pct = 100.0 - surv_pct;

    printf("{\"noseg\":%d,\"prompt_tokens\":%zu,\"window\":%u,\"n_pin\":%u,\"seg_budget\":%u,"
           "\"mem_cnt\":%u,\"uniq_segs\":%u,\"candidate_evicted_region_tokens\":%.0f,"
           "\"survival_rate_pct\":%.2f,\"eviction_pressure_pct\":%.2f,"
           "\"hits\":%d,\"total\":%d,\"acc_pct\":%.2f,"
           "\"prefill_s\":%.2f,\"prefill_tok_s\":%.2f,\"queries_26_s\":%.2f,\"state_arena_mb\":%.2f,\"peak_rss_mb\":%.2f}\n",
           noseg, last_total_toks, m.window, s.n_pin, s.seg_budget,
           final_mem_cnt, final_uniq_segs, cand_toks,
           surv_pct, evict_pct,
           hits, total_q, total_q > 0 ? 100.0 * hits / total_q : 0.0,
           prefill_s, pref.n / (prefill_s > 0 ? prefill_s : 1e-9), q_time, st_sz / 1e6, peak_rss_mb());

    pool_destroy(&pool);
    free(pref.data);
    free(doc_txt);
    free(arena);
    free(arena_bak);
    free(tok_arena);
    munmap((void *)buf, (size_t)st.st_size);
    close(fd);
    return 0;
}
