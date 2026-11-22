# VAULT v3 runtime build
#   make            -> hosted harness (vault3_host) + scalar, AVX2, & AArch64 NEON freestanding checks
#   make portable   -> pure scalar C11 build (works on any CPU architecture)
#   make freestanding

CC ?= cc
UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
UNAME_M := $(shell uname -m 2>/dev/null || echo Unknown)
HAS_CLANG := $(shell command -v clang >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(UNAME_M),x86_64)
  ifeq ($(UNAME_S),Linux)
    HAS_AVX2 := $(shell grep -qE '\bavx2\b' /proc/cpuinfo 2>/dev/null && grep -qE '\bfma\b' /proc/cpuinfo 2>/dev/null && grep -qE '\bf16c\b' /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
  else ifeq ($(UNAME_S),Darwin)
    HAS_AVX2 := $(shell sysctl -n hw.optional.avx2_0 2>/dev/null || echo 0)
  else
    HAS_AVX2 := 0
  endif
else
  HAS_AVX2 := 0
endif

ifeq ($(HAS_AVX2),1)
  ARCH_FLAGS ?= -mavx2 -mfma -mf16c
else
  ARCH_FLAGS ?=
endif

CFLAGS_FS_SCALAR = -std=c99 -O2 -ffreestanding -nostdlib -Wall -Wextra -Wframe-larger-than=2048
CFLAGS_FS_AVX2   = -std=c99 -O2 -ffreestanding -nostdlib -mavx2 -mfma -mf16c -Wall -Wextra -Wframe-larger-than=2048
CFLAGS_FS_NEON   = --target=aarch64-linux-gnu -std=c99 -O2 -ffreestanding -nostdlib -Wall -Wextra -Werror -Wframe-larger-than=2048
CFLAGS_HOST      = -std=c11 -O3 $(ARCH_FLAGS) -fno-math-errno -pthread -Wall -Wextra
CFLAGS_PORTABLE  = -std=c11 -O3 -fno-math-errno -pthread -Wall -Wextra

all: freestanding vault3_host

freestanding: vault3.c vault3.h vault3_nfc_tables.h
	$(CC) $(CFLAGS_FS_SCALAR) -DV3_FORCE_SCALAR_MATH -c vault3.c -o vault3_freestanding_scalar.o
	@if nm vault3_freestanding_scalar.o | grep -q " U "; then echo "undefined symbols in scalar:"; nm vault3_freestanding_scalar.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding_scalar.o (pure scalar C99, no -fno-math-errno): 0 external undefined symbols"; fi
ifeq ($(HAS_AVX2),1)
	$(CC) $(CFLAGS_FS_AVX2) -c vault3.c -o vault3_freestanding.o
	@if nm vault3_freestanding.o | grep -q " U "; then echo "undefined symbols in avx2:"; nm vault3_freestanding.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding.o (AVX2+FMA, no -fno-math-errno): 0 external undefined symbols"; fi
endif
ifeq ($(HAS_CLANG),1)
	clang $(CFLAGS_FS_NEON) -c vault3.c -o vault3_freestanding_neon.o
	@if nm vault3_freestanding_neon.o | grep -q " U "; then echo "undefined symbols in neon:"; nm vault3_freestanding_neon.o | grep " U "; exit 1; else echo "[ok] vault3_freestanding_neon.o (AArch64 NEON, Clang -Werror): 0 external undefined symbols"; fi
endif

vault3_host: vault3.c vault3.h vault3_nfc_tables.h vault3_host.c
	$(CC) $(CFLAGS_HOST) vault3.c vault3_host.c -o vault3_host

portable: vault3.c vault3.h vault3_nfc_tables.h vault3_host.c
	$(CC) $(CFLAGS_PORTABLE) -DV3_FORCE_SCALAR_MATH vault3.c vault3_host.c -o vault3_host

clean:
	rm -f vault3_host vault3_freestanding.o vault3_freestanding_scalar.o vault3_freestanding_neon.o

