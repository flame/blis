/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "immintrin.h"
#include "blis.h"

#ifdef BLIS_ENABLE_OPENMP
#include <omp.h>
#endif



/*
 * CH_S_x: Mapping of BLIS type characters to their real-type equivalents.
 * (s->s, d->d, c->s, z->d). Used for generating kernels with real/SIMD logic.
 */
#define CH_S_s s
#define CH_S_d d
#define CH_S_c s
#define CH_S_z d


// ============================================================
// ISA Abstraction Layer
//
// Set ARCH_SIMD_BITS to 512 (AVX-512) or 256 (AVX2) before
// including this header.  Defaults to 512 when not defined.
//
// This block defines:
//   SIMD_VEC_s / SIMD_VEC_d      — vector type names (__m512 / __m256)
//   SIMD_LOADU_P_s/d             — _mm*_loadu_ps/pd
//   SIMD_STOREU_P_s/d            — _mm*_storeu_ps/pd
//   SIMD_SETZERO_P_s/d           — _mm*_setzero_ps/pd
//   SIMD_SET1_P_s/d              — _mm*_set1_ps/pd
//   SIMD_FMADD_P_s/d             — _mm*_fmadd_ps/pd
//   SIMD_MUL_P_s/d               — _mm*_mul_ps/pd
//   SIMD_FMADDSUB_P_s/d          — _mm*_fmaddsub_ps/pd
//   SIMD_PERMUTE_P_s/d           — _mm*_permute_ps/pd
//   SIMD_MASKZ_LOADU_P_s/d(k,p) — masked zero-load
//   SIMD_MASK_STOREU_P_s/d(p,k,v)— masked store
//   SIMD_REDUCE_ADD_P_s/d(v)    — horizontal reduction to scalar
//   SIMD_MASK_s / SIMD_MASK_d   — mask type
//   ELEM_PER_REG_s/d/c/z         — elements per SIMD register
// ============================================================
// Convenience aliases: PASTECH(SIMD_VEC_, ch_s) expands to SIMD_VEC_s or SIMD_VEC_d
// which resolves to the correct vector type (__m512/__m256 or __m512d/__m256d).
// Similarly PASTECH(SIMD_LOADU_P_, ch_s) picks the right load intrinsic.
// These aliases exist so all kernel macros can be written once and work for both ISAs.

// ARCH_SIMD_BITS selects the SIMD width (256 for AVX2/YMM, 512 for AVX-512/ZMM)
// and MUST be defined by the including shim BEFORE this header is included.
// We deliberately do NOT default it: a silent fallback to 512 would pull AVX-512
// intrinsics into an AVX2-only translation unit and fail (often cryptically) on
// compilers/targets without AVX-512 support.
#ifndef ARCH_SIMD_BITS
  #error "ARCH_SIMD_BITS must be defined (256 or 512) before including bli_pp_common.h"
#endif

#if ARCH_SIMD_BITS == 512

  // --- AVX-512 (ZMM, 512-bit) ---
  #define SIMD_VEC_s           __m512
  #define SIMD_VEC_d           __m512d

  #define SIMD_LOADU_P_s       _mm512_loadu_ps
  #define SIMD_LOADU_P_d       _mm512_loadu_pd
  #define SIMD_STOREU_P_s      _mm512_storeu_ps
  #define SIMD_STOREU_P_d      _mm512_storeu_pd
  #define SIMD_SETZERO_P_s     _mm512_setzero_ps
  #define SIMD_SETZERO_P_d     _mm512_setzero_pd
  #define SIMD_SET1_P_s        _mm512_set1_ps
  #define SIMD_SET1_P_d        _mm512_set1_pd
  #define SIMD_FMADD_P_s       _mm512_fmadd_ps
  #define SIMD_FMADD_P_d       _mm512_fmadd_pd
  #define SIMD_MUL_P_s         _mm512_mul_ps
  #define SIMD_MUL_P_d         _mm512_mul_pd
  #define SIMD_FMADDSUB_P_s    _mm512_fmaddsub_ps
  #define SIMD_FMADDSUB_P_d    _mm512_fmaddsub_pd
  #define SIMD_PERMUTE_P_s     _mm512_permute_ps
  #define SIMD_PERMUTE_P_d     _mm512_permute_pd

  #define SIMD_MASK_s          __mmask16
  #define SIMD_MASK_d          __mmask8

  #define SIMD_MASKZ_LOADU_P_s(k, ptr)     _mm512_maskz_loadu_ps(k, ptr)
  #define SIMD_MASKZ_LOADU_P_d(k, ptr)     _mm512_maskz_loadu_pd(k, ptr)
  #define SIMD_MASK_STOREU_P_s(ptr, k, v)  _mm512_mask_storeu_ps(ptr, k, v)
  #define SIMD_MASK_STOREU_P_d(ptr, k, v)  _mm512_mask_storeu_pd(ptr, k, v)

  #define SIMD_REDUCE_ADD_P_s(v)           _mm512_reduce_add_ps(v)
  #define SIMD_REDUCE_ADD_P_d(v)           _mm512_reduce_add_pd(v)

  // ELEM_PER_REG: elements of each type per ZMM register
  #define ELEM_PER_REG_s  16    // floats  per ZMM
  #define ELEM_PER_REG_d   8    // doubles per ZMM
  #define ELEM_PER_REG_c   8    // scomplex per ZMM (each = 2 floats, 8 complex elements)
  #define ELEM_PER_REG_z   4    // dcomplex per ZMM (each = 2 doubles, 4 complex elements)

#elif ARCH_SIMD_BITS == 256

  // --- AVX2+FMA (YMM, 256-bit) ---
  #define SIMD_VEC_s           __m256
  #define SIMD_VEC_d           __m256d

  #define SIMD_LOADU_P_s       _mm256_loadu_ps
  #define SIMD_LOADU_P_d       _mm256_loadu_pd
  #define SIMD_STOREU_P_s      _mm256_storeu_ps
  #define SIMD_STOREU_P_d      _mm256_storeu_pd
  #define SIMD_SETZERO_P_s     _mm256_setzero_ps
  #define SIMD_SETZERO_P_d     _mm256_setzero_pd
  #define SIMD_SET1_P_s        _mm256_set1_ps
  #define SIMD_SET1_P_d        _mm256_set1_pd
  #define SIMD_FMADD_P_s       _mm256_fmadd_ps
  #define SIMD_FMADD_P_d       _mm256_fmadd_pd
  #define SIMD_MUL_P_s         _mm256_mul_ps
  #define SIMD_MUL_P_d         _mm256_mul_pd
  #define SIMD_FMADDSUB_P_s    _mm256_fmaddsub_ps
  #define SIMD_FMADDSUB_P_d    _mm256_fmaddsub_pd
  #define SIMD_PERMUTE_P_s     _mm256_permute_ps
  #define SIMD_PERMUTE_P_d     _mm256_permute_pd

  // AVX2 uses integer-vector masks (sign bits), not k-registers.
  // pp_avx2_mask_epi32/64: convert a BITMASK (matching AVX-512 mask semantics)
  // to a __m256i lane mask where lane i = -1 if bit i of k is set, else 0.
  // This keeps SIMD_MASKZ_LOADU/STOREU call sites uniform across ISAs.
  static inline __m256i pp_avx2_mask_epi32(unsigned k) {
      __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
      __m256i kv   = _mm256_set1_epi32((int)k);
      return _mm256_cmpeq_epi32(_mm256_and_si256(kv, bits), bits);
  }
  static inline __m256i pp_avx2_mask_epi64(unsigned k) {
      __m256i bits = _mm256_setr_epi64x(1, 2, 4, 8);
      __m256i kv   = _mm256_set1_epi64x((long long)k);
      return _mm256_cmpeq_epi64(_mm256_and_si256(kv, bits), bits);
  }

  #define SIMD_MASK_s          __m256i   // sign-bit mask for _mm256_maskload/store_ps
  #define SIMD_MASK_d          __m256i   // sign-bit mask for _mm256_maskload/store_pd

  #define SIMD_MASKZ_LOADU_P_s(k, ptr)    \
      _mm256_maskload_ps((ptr), pp_avx2_mask_epi32(k))
  #define SIMD_MASKZ_LOADU_P_d(k, ptr)    \
      _mm256_maskload_pd((ptr), pp_avx2_mask_epi64(k))
  #define SIMD_MASK_STOREU_P_s(ptr, k, v) \
      _mm256_maskstore_ps((ptr), pp_avx2_mask_epi32(k), (v))
  #define SIMD_MASK_STOREU_P_d(ptr, k, v) \
      _mm256_maskstore_pd((ptr), pp_avx2_mask_epi64(k), (v))

  // Horizontal reduce-add for AVX2 (no _mm256_reduce_add_p* intrinsic).
  static inline float pp_avx2_reduce_add_ps(__m256 v) {
      __m128 lo = _mm256_castps256_ps128(v);
      __m128 hi = _mm256_extractf128_ps(v, 1);
      __m128 s  = _mm_add_ps(lo, hi);
      s = _mm_hadd_ps(s, s);
      s = _mm_hadd_ps(s, s);
      return _mm_cvtss_f32(s);
  }
  static inline double pp_avx2_reduce_add_pd(__m256d v) {
      __m128d lo = _mm256_castpd256_pd128(v);
      __m128d hi = _mm256_extractf128_pd(v, 1);
      __m128d s  = _mm_add_pd(lo, hi);
      s = _mm_hadd_pd(s, s);
      return _mm_cvtsd_f64(s);
  }
  #define SIMD_REDUCE_ADD_P_s(v)           pp_avx2_reduce_add_ps(v)
  #define SIMD_REDUCE_ADD_P_d(v)           pp_avx2_reduce_add_pd(v)

  // ELEM_PER_REG: elements of each type per YMM register
  #define ELEM_PER_REG_s   8    // floats  per YMM
  #define ELEM_PER_REG_d   4    // doubles per YMM
  #define ELEM_PER_REG_c   4    // scomplex per YMM (4 complex elements)
  #define ELEM_PER_REG_z   2    // dcomplex per YMM (2 complex elements)

#else
  #error "ARCH_SIMD_BITS must be 512 (AVX-512) or 256 (AVX2)"
#endif


/* PERM_MASK_z: swaps [re,im] pairs in a dcomplex SIMD register.
 * _mm512_permute_pd takes an 8-bit imm8 → 0x55 (01010101b).
 * _mm256_permute_pd takes only a 4-bit imm4 → 0x5  (0101b), same semantics.
 * PERM_MASK_c (0xB1): same for scomplex; _mm256/512_permute_ps both accept 8-bit. */
#if ARCH_SIMD_BITS == 512
  #define PERM_MASK_z 0x55
#else
  #define PERM_MASK_z 0x5
#endif
#define PERM_MASK_c 0xB1

// Values of one for s/d/c/z
#define ONE_s 1
#define ONE_d 1
#define ONE_c {1, 0}
#define ONE_z {1, 0}

#define ZERO_s 0
#define ZERO_d 0
#define ZERO_c {0, 0}
#define ZERO_z {0, 0}

/*
 * Instruction-name helpers: PASTECH(INSTR, ch) produces lowercase (e.g. VXORPs),
 * but the BLIS ASM macros use uppercase (VXORPS). Map ch -> full uppercase instruction.
 */
#define ASM_VXOR_s        VXORPS
#define ASM_VXOR_d        VXORPD
#define ASM_VMOVUP_s      VMOVUPS
#define ASM_VMOVUP_d      VMOVUPD
#define ASM_VFMADD231P_s  VFMADD231PS
#define ASM_VFMADD231P_d  VFMADD231PD

/*
 * ZMM_E(n): evaluate n (which may be a macro expression) before token-pasting
 * into the ZMM register name. Normal ZMM(x) does Zmm##x which suppresses
 * expansion of x; ZMM_E forces expansion first via a one-level indirection.
 * Alternative solution could be to use ZMM(x + 0).
 */
#define ZMM_E(n) ZMM(n)


/*
 * UNROLL_LOOP_FULL(N): Compiler hint to fully unroll the immediately following loop.
 * Enabled for Clang and GCC (release builds only — in debug mode, partial unroll of 4
 * is used. Falls back to a no-op for other compilers.
 */
#if defined __clang__
    #define UNROLL_LOOP_FULL(N) _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define STR_HELPER(x) #x
    #define STR(x) STR_HELPER(x)
    /* The loop unroll macro */
    #ifdef NDEBUG
        #define UNROLL_LOOP_FULL(N) _Pragma(STR(GCC unroll N))
    #else
        #define UNROLL_LOOP_FULL(N) _Pragma("GCC unroll 4")
    #endif
#else
    #define UNROLL_LOOP_FULL(N)
#endif


/*
 * HSUM_z / HSUM_c: Horizontal sum reduction for complex accumulators.
 *
 * For the transpose case, the result y[ii] = dot(A_row[ii], x).
 * The complex FMA accumulates real and imaginary contributions interleaved
 * across SIMD lanes. After the main loop:
 *
 *   yv[ii]      = [...] holds products of re(A)*re(x), with alternating sign
 *                        due to FMADDSUB — needs alternating sum to collapse.
 *   yv[ii+NR]   = [...] holds products of re(A)*im(x) — needs plain sum.
 *
 * The real part is: sum of (re(A[k])*re(x[k]) - im(A[k])*im(x[k]))
 *   -> alternating +/- sum across SIMD lanes of yv[ii]
 * The imag part is: sum of (re(A[k])*im(x[k]) + im(A[k])*re(x[k]))
 *   -> plain sum across SIMD lanes of yv[ii+NR]
 *
 * Element counts are ISA-dependent:
 *   HSUM_z: ELEM_PER_REG_z*2 doubles in register (dcomplex: 8 for 512-bit, 4 for 256-bit)
 *   HSUM_c: ELEM_PER_REG_c*2 floats  in register (scomplex: 16 for 512-bit, 8 for 256-bit)
 */
#if ARCH_SIMD_BITS == 512

#define HSUM_z(NR)                                                                                  \
  temp.real = yv[ii][0] - yv[ii][1] + yv[ii][2] - yv[ii][3] +                                       \
              yv[ii][4] - yv[ii][5] + yv[ii][6] - yv[ii][7];                                        \
  temp.imag = yv[ii + NR][0] + yv[ii + NR][1] + yv[ii + NR][2] +                                    \
              yv[ii + NR][3] + yv[ii + NR][4] + yv[ii + NR][5] +                                    \
              yv[ii + NR][6] + yv[ii + NR][7];

#define HSUM_z_CONJ(NR)                                                                             \
  temp.real = yv[ii][0] + yv[ii][1] + yv[ii][2] + yv[ii][3] +                                       \
              yv[ii][4] + yv[ii][5] + yv[ii][6] + yv[ii][7];                                        \
  temp.imag = yv[ii + NR][0] - yv[ii + NR][1] + yv[ii + NR][2] - yv[ii + NR][3] +                   \
              yv[ii + NR][4] - yv[ii + NR][5] + yv[ii + NR][6] - yv[ii + NR][7];

#define HSUM_c(NR)                                                                                  \
  temp.real = yv[ii][0] - yv[ii][1] + yv[ii][2] - yv[ii][3] +                                       \
              yv[ii][4] - yv[ii][5] + yv[ii][6] - yv[ii][7] +                                       \
              yv[ii][8] - yv[ii][9] + yv[ii][10] - yv[ii][11] +                                     \
              yv[ii][12] - yv[ii][13] + yv[ii][14] - yv[ii][15];                                    \
  temp.imag = yv[ii + NR][0] + yv[ii + NR][1] + yv[ii + NR][2] +                                    \
              yv[ii + NR][3] + yv[ii + NR][4] + yv[ii + NR][5] +                                    \
              yv[ii + NR][6] + yv[ii + NR][7] + yv[ii + NR][8] + yv[ii + NR][9] + yv[ii + NR][10] + \
              yv[ii + NR][11] + yv[ii + NR][12] + yv[ii + NR][13] +                                 \
              yv[ii + NR][14] + yv[ii + NR][15];

#define HSUM_c_CONJ(NR)                                                                             \
  temp.real = yv[ii][0] + yv[ii][1] + yv[ii][2] + yv[ii][3] +                                       \
              yv[ii][4] + yv[ii][5] + yv[ii][6] + yv[ii][7] +                                       \
              yv[ii][8] + yv[ii][9] + yv[ii][10] + yv[ii][11] +                                     \
              yv[ii][12] + yv[ii][13] + yv[ii][14] + yv[ii][15];                                    \
  temp.imag = yv[ii + NR][0] - yv[ii + NR][1] + yv[ii + NR][2] - yv[ii + NR][3] +                   \
              yv[ii + NR][4] - yv[ii + NR][5] + yv[ii + NR][6] - yv[ii + NR][7] +                   \
              yv[ii + NR][8] - yv[ii + NR][9] + yv[ii + NR][10] - yv[ii + NR][11] +                 \
              yv[ii + NR][12] - yv[ii + NR][13] + yv[ii + NR][14] - yv[ii + NR][15];

#else  // ARCH_SIMD_BITS == 256

/* HSUM_z: 4 doubles per YMM (2 dcomplex elements, indices [0]..[3]) */
#define HSUM_z(NR)                                                                                  \
  temp.real = yv[ii][0] - yv[ii][1] + yv[ii][2] - yv[ii][3];                                        \
  temp.imag = yv[ii + NR][0] + yv[ii + NR][1] + yv[ii + NR][2] + yv[ii + NR][3];

#define HSUM_z_CONJ(NR)                                                                             \
  temp.real = yv[ii][0] + yv[ii][1] + yv[ii][2] + yv[ii][3];                                        \
  temp.imag = yv[ii + NR][0] - yv[ii + NR][1] + yv[ii + NR][2] - yv[ii + NR][3];

/* HSUM_c: 8 floats per YMM (4 scomplex elements, indices [0]..[7]) */
#define HSUM_c(NR)                                                                                  \
  temp.real = yv[ii][0] - yv[ii][1] + yv[ii][2] - yv[ii][3] +                                       \
              yv[ii][4] - yv[ii][5] + yv[ii][6] - yv[ii][7];                                        \
  temp.imag = yv[ii + NR][0] + yv[ii + NR][1] + yv[ii + NR][2] + yv[ii + NR][3] +                   \
              yv[ii + NR][4] + yv[ii + NR][5] + yv[ii + NR][6] + yv[ii + NR][7];

#define HSUM_c_CONJ(NR)                                                                             \
  temp.real = yv[ii][0] + yv[ii][1] + yv[ii][2] + yv[ii][3] +                                       \
              yv[ii][4] + yv[ii][5] + yv[ii][6] + yv[ii][7];                                        \
  temp.imag = yv[ii + NR][0] - yv[ii + NR][1] + yv[ii + NR][2] - yv[ii + NR][3] +                   \
              yv[ii + NR][4] - yv[ii + NR][5] + yv[ii + NR][6] - yv[ii + NR][7];

#endif  // ARCH_SIMD_BITS

// #region Compile Time loops

/*
 * Macros FE_0 to FE_10 are helper macros used to implement a variadic "FOR_EACH"
 * functionality at compile-time. They apply an 'ACTION' to each element 'X'
 * in a list of arguments, passing down three additional data parameters.
 */
#define FE_0(ACTION, ...)
#define FE_1(ACTION, DATA1, DATA2, DATA3, X) ACTION(DATA1, DATA2, DATA3, X)
#define FE_2(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_1(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_3(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_2(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_4(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_3(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_5(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_4(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_6(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_5(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_7(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_6(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_8(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_7(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_9(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_8(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)
#define FE_10(ACTION, DATA1, DATA2, DATA3, X, ...) ACTION(DATA1, DATA2, DATA3, X) FE_9(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)


/*
 * GET_MACRO and FOR_EACH: Used to count the number of arguments and dispatch
 * to the appropriate FE_N macro based on the argument count. Supports up to 10 args.
 */
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,NAME,...) NAME
#define FOR_EACH(ACTION, DATA1, DATA2, DATA3, ...) \
    GET_MACRO(__VA_ARGS__, FE_10, FE_9, FE_8, FE_7, FE_6, FE_5, FE_4, FE_3, FE_2, FE_1)(ACTION, DATA1, DATA2, DATA3, __VA_ARGS__)

/*
 * RANGE_DOWN_N: Recursively generates a descending sequence of action calls.
 * RANGE_DOWN_N(ACT, A, B, M) expands to:
 *   ACT(A, B, M, N)  ACT(A, B, M, N-1)  ...  ACT(A, B, M, 1)
 *
 * Used by GEN_FUNC_PAIR to install both the full-tile kernel (MR=M, NR=1..N)
 * and the fringe kernel (MR=M-1, NR=1..N) across all supported column counts.
 * Dispatcher macro:
 *   RANGE_DOWN(ACT, A, B, M, N) -> RANGE_DOWN_N(ACT, A, B, M)
 */
#define RANGE_DOWN_1(ACT, A, B, M) ACT(A, B, M, 1)
#define RANGE_DOWN_2(ACT, A, B, M) ACT(A, B, M, 2) RANGE_DOWN_1(ACT, A, B, M)
#define RANGE_DOWN_3(ACT, A, B, M) ACT(A, B, M, 3) RANGE_DOWN_2(ACT, A, B, M)
#define RANGE_DOWN_4(ACT, A, B, M) ACT(A, B, M, 4) RANGE_DOWN_3(ACT, A, B, M)
#define RANGE_DOWN_5(ACT, A, B, M) ACT(A, B, M, 5) RANGE_DOWN_4(ACT, A, B, M)
#define RANGE_DOWN_6(ACT, A, B, M) ACT(A, B, M, 6) RANGE_DOWN_5(ACT, A, B, M)
#define RANGE_DOWN_7(ACT, A, B, M) ACT(A, B, M, 7) RANGE_DOWN_6(ACT, A, B, M)
#define RANGE_DOWN_8(ACT, A, B, M) ACT(A, B, M, 8) RANGE_DOWN_7(ACT, A, B, M)
#define RANGE_DOWN_9(ACT, A, B, M) ACT(A, B, M, 9) RANGE_DOWN_8(ACT, A, B, M)
#define RANGE_DOWN_10(ACT, A, B, M) ACT(A, B, M, 10) RANGE_DOWN_9(ACT, A, B, M)
#define RANGE_DOWN_11(ACT, A, B, M) ACT(A, B, M, 11) RANGE_DOWN_10(ACT, A, B, M)
#define RANGE_DOWN_12(ACT, A, B, M) ACT(A, B, M, 12) RANGE_DOWN_11(ACT, A, B, M)
#define RANGE_DOWN_13(ACT, A, B, M) ACT(A, B, M, 13) RANGE_DOWN_12(ACT, A, B, M)
#define RANGE_DOWN_14(ACT, A, B, M) ACT(A, B, M, 14) RANGE_DOWN_13(ACT, A, B, M)
#define RANGE_DOWN_15(ACT, A, B, M) ACT(A, B, M, 15) RANGE_DOWN_14(ACT, A, B, M)
#define RANGE_DOWN_16(ACT, A, B, M) ACT(A, B, M, 16) RANGE_DOWN_15(ACT, A, B, M)
#define RANGE_DOWN_17(ACT, A, B, M) ACT(A, B, M, 17) RANGE_DOWN_16(ACT, A, B, M)
#define RANGE_DOWN_18(ACT, A, B, M) ACT(A, B, M, 18) RANGE_DOWN_17(ACT, A, B, M)
#define RANGE_DOWN_19(ACT, A, B, M) ACT(A, B, M, 19) RANGE_DOWN_18(ACT, A, B, M)
#define RANGE_DOWN_20(ACT, A, B, M) ACT(A, B, M, 20) RANGE_DOWN_19(ACT, A, B, M)
#define RANGE_DOWN_21(ACT, A, B, M) ACT(A, B, M, 21) RANGE_DOWN_20(ACT, A, B, M)
#define RANGE_DOWN_22(ACT, A, B, M) ACT(A, B, M, 22) RANGE_DOWN_21(ACT, A, B, M)
#define RANGE_DOWN_23(ACT, A, B, M) ACT(A, B, M, 23) RANGE_DOWN_22(ACT, A, B, M)
#define RANGE_DOWN_24(ACT, A, B, M) ACT(A, B, M, 24) RANGE_DOWN_23(ACT, A, B, M)
#define RANGE_DOWN_25(ACT, A, B, M) ACT(A, B, M, 25) RANGE_DOWN_24(ACT, A, B, M)
#define RANGE_DOWN_26(ACT, A, B, M) ACT(A, B, M, 26) RANGE_DOWN_25(ACT, A, B, M)
#define RANGE_DOWN_27(ACT, A, B, M) ACT(A, B, M, 27) RANGE_DOWN_26(ACT, A, B, M)
#define RANGE_DOWN_28(ACT, A, B, M) ACT(A, B, M, 28) RANGE_DOWN_27(ACT, A, B, M)
#define RANGE_DOWN_29(ACT, A, B, M) ACT(A, B, M, 29) RANGE_DOWN_28(ACT, A, B, M)
#define RANGE_DOWN_30(ACT, A, B, M) ACT(A, B, M, 30) RANGE_DOWN_29(ACT, A, B, M)
#define RANGE_DOWN_31(ACT, A, B, M) ACT(A, B, M, 31) RANGE_DOWN_30(ACT, A, B, M)
#define RANGE_DOWN_32(ACT, A, B, M) ACT(A, B, M, 32) RANGE_DOWN_31(ACT, A, B, M)
#define RANGE_DOWN_33(ACT, A, B, M) ACT(A, B, M, 33) RANGE_DOWN_32(ACT, A, B, M)
#define RANGE_DOWN_34(ACT, A, B, M) ACT(A, B, M, 34) RANGE_DOWN_33(ACT, A, B, M)
#define RANGE_DOWN_35(ACT, A, B, M) ACT(A, B, M, 35) RANGE_DOWN_34(ACT, A, B, M)
#define RANGE_DOWN_36(ACT, A, B, M) ACT(A, B, M, 36) RANGE_DOWN_35(ACT, A, B, M)
#define RANGE_DOWN_37(ACT, A, B, M) ACT(A, B, M, 37) RANGE_DOWN_36(ACT, A, B, M)
#define RANGE_DOWN_38(ACT, A, B, M) ACT(A, B, M, 38) RANGE_DOWN_37(ACT, A, B, M)
#define RANGE_DOWN_39(ACT, A, B, M) ACT(A, B, M, 39) RANGE_DOWN_38(ACT, A, B, M)
#define RANGE_DOWN_40(ACT, A, B, M) ACT(A, B, M, 40) RANGE_DOWN_39(ACT, A, B, M)
/* Dispatcher: RANGE_DOWN(ACT, A, B, M, N) -> RANGE_DOWN_N(ACT, A, B, M) */
#define RANGE_DOWN(ACT, A, B, M, N) RANGE_DOWN_##N(ACT, A, B, M)

/*
 * DEC_XX: Constants used to decrement specific block size values by 1.
 * These are tailored for common GEMV kernel block sizes.
 */
#define DEC_96 95
#define DEC_80 79
#define DEC_64 63
#define DEC_56 55
#define DEC_48 47
#define DEC_44 43
#define DEC_40 39
#define DEC_36 35
#define DEC_32 31
#define DEC_28 27
#define DEC_24 23
#define DEC_20 19
#define DEC_16 15
#define DEC_12 11
#define DEC_10  9
#define DEC_8 7
#define DEC_6 5
#define DEC_4 3
#define DEC_2 1

// #region PP_LOOP — count-up loop (complement of RANGE_DOWN)
/*
 * PP_LOOP(N, ACT, A, B, M):
 *   Expands to:  ACT(A, B, M, 1)  ACT(A, B, M, 2)  ...  ACT(A, B, M, N)
 *
 * Analogous to BOOST_PP_REPEAT / BOOST_PP_FOR counting upward.
 * Useful when you want column indices in ascending order rather than
 * the descending order that RANGE_DOWN produces.
 *
 * Supports N = 1 .. 40 (same upper bound as RANGE_DOWN).
 *
 * Example:
 *   PP_LOOP(4, MY_MACRO, ctype, ch, MR)
 *   expands to:
 *   MY_MACRO(ctype, ch, MR, 1)
 *   MY_MACRO(ctype, ch, MR, 2)
 *   MY_MACRO(ctype, ch, MR, 3)
 *   MY_MACRO(ctype, ch, MR, 4)
 */
#define PP_LOOP_1(ACT, A, B, M)  ACT(A, B, M, 1)
#define PP_LOOP_2(ACT, A, B, M)  PP_LOOP_1(ACT, A, B, M)  ACT(A, B, M, 2)
#define PP_LOOP_3(ACT, A, B, M)  PP_LOOP_2(ACT, A, B, M)  ACT(A, B, M, 3)
#define PP_LOOP_4(ACT, A, B, M)  PP_LOOP_3(ACT, A, B, M)  ACT(A, B, M, 4)
#define PP_LOOP_5(ACT, A, B, M)  PP_LOOP_4(ACT, A, B, M)  ACT(A, B, M, 5)
#define PP_LOOP_6(ACT, A, B, M)  PP_LOOP_5(ACT, A, B, M)  ACT(A, B, M, 6)
#define PP_LOOP_7(ACT, A, B, M)  PP_LOOP_6(ACT, A, B, M)  ACT(A, B, M, 7)
#define PP_LOOP_8(ACT, A, B, M)  PP_LOOP_7(ACT, A, B, M)  ACT(A, B, M, 8)
#define PP_LOOP_9(ACT, A, B, M)  PP_LOOP_8(ACT, A, B, M)  ACT(A, B, M, 9)
#define PP_LOOP_10(ACT, A, B, M) PP_LOOP_9(ACT, A, B, M)  ACT(A, B, M, 10)
#define PP_LOOP_11(ACT, A, B, M) PP_LOOP_10(ACT, A, B, M) ACT(A, B, M, 11)
#define PP_LOOP_12(ACT, A, B, M) PP_LOOP_11(ACT, A, B, M) ACT(A, B, M, 12)
#define PP_LOOP_13(ACT, A, B, M) PP_LOOP_12(ACT, A, B, M) ACT(A, B, M, 13)
#define PP_LOOP_14(ACT, A, B, M) PP_LOOP_13(ACT, A, B, M) ACT(A, B, M, 14)
#define PP_LOOP_15(ACT, A, B, M) PP_LOOP_14(ACT, A, B, M) ACT(A, B, M, 15)
#define PP_LOOP_16(ACT, A, B, M) PP_LOOP_15(ACT, A, B, M) ACT(A, B, M, 16)
#define PP_LOOP_17(ACT, A, B, M) PP_LOOP_16(ACT, A, B, M) ACT(A, B, M, 17)
#define PP_LOOP_18(ACT, A, B, M) PP_LOOP_17(ACT, A, B, M) ACT(A, B, M, 18)
#define PP_LOOP_19(ACT, A, B, M) PP_LOOP_18(ACT, A, B, M) ACT(A, B, M, 19)
#define PP_LOOP_20(ACT, A, B, M) PP_LOOP_19(ACT, A, B, M) ACT(A, B, M, 20)
#define PP_LOOP_21(ACT, A, B, M) PP_LOOP_20(ACT, A, B, M) ACT(A, B, M, 21)
#define PP_LOOP_22(ACT, A, B, M) PP_LOOP_21(ACT, A, B, M) ACT(A, B, M, 22)
#define PP_LOOP_23(ACT, A, B, M) PP_LOOP_22(ACT, A, B, M) ACT(A, B, M, 23)
#define PP_LOOP_24(ACT, A, B, M) PP_LOOP_23(ACT, A, B, M) ACT(A, B, M, 24)
#define PP_LOOP_25(ACT, A, B, M) PP_LOOP_24(ACT, A, B, M) ACT(A, B, M, 25)
#define PP_LOOP_26(ACT, A, B, M) PP_LOOP_25(ACT, A, B, M) ACT(A, B, M, 26)
#define PP_LOOP_27(ACT, A, B, M) PP_LOOP_26(ACT, A, B, M) ACT(A, B, M, 27)
#define PP_LOOP_28(ACT, A, B, M) PP_LOOP_27(ACT, A, B, M) ACT(A, B, M, 28)
#define PP_LOOP_29(ACT, A, B, M) PP_LOOP_28(ACT, A, B, M) ACT(A, B, M, 29)
#define PP_LOOP_30(ACT, A, B, M) PP_LOOP_29(ACT, A, B, M) ACT(A, B, M, 30)
#define PP_LOOP_31(ACT, A, B, M) PP_LOOP_30(ACT, A, B, M) ACT(A, B, M, 31)
#define PP_LOOP_32(ACT, A, B, M) PP_LOOP_31(ACT, A, B, M) ACT(A, B, M, 32)
#define PP_LOOP_33(ACT, A, B, M) PP_LOOP_32(ACT, A, B, M) ACT(A, B, M, 33)
#define PP_LOOP_34(ACT, A, B, M) PP_LOOP_33(ACT, A, B, M) ACT(A, B, M, 34)
#define PP_LOOP_35(ACT, A, B, M) PP_LOOP_34(ACT, A, B, M) ACT(A, B, M, 35)
#define PP_LOOP_36(ACT, A, B, M) PP_LOOP_35(ACT, A, B, M) ACT(A, B, M, 36)
#define PP_LOOP_37(ACT, A, B, M) PP_LOOP_36(ACT, A, B, M) ACT(A, B, M, 37)
#define PP_LOOP_38(ACT, A, B, M) PP_LOOP_37(ACT, A, B, M) ACT(A, B, M, 38)
#define PP_LOOP_39(ACT, A, B, M) PP_LOOP_38(ACT, A, B, M) ACT(A, B, M, 39)
#define PP_LOOP_40(ACT, A, B, M) PP_LOOP_39(ACT, A, B, M) ACT(A, B, M, 40)

/* PP_LOOP_0: no-op base case — needed when MR < ELEM_PER_REG (pure-fringe tile). */
#define PP_LOOP_0(ACT, A, B, M)

/*
 * PP_LOOP dispatcher — two-level to force argument pre-expansion.
 *
 * ## suppresses macro expansion of adjacent arguments (C standard §6.10.3.3).
 * Without the indirection, PP_LOOP(PP_DIV(MR,EPR_OF(ch)), ...) would try to
 * token-paste "PP_LOOP_" with the literal text "PP_DIV(MR,EPR_OF(ch))" rather
 * than with the resolved integer (e.g. 5).
 *
 * PP_LOOP_APPLY contains the ##, so its argument N is already fully expanded
 * (to e.g. 5) before the paste occurs. PP_LOOP forces that expansion by calling
 * PP_LOOP_APPLY with N as an ordinary (non-## adjacent) argument.
 *
 * This is the same pattern used by PP_DIV (PP_DIV_IMPL) and PP_IF (PP_IF_IMPL_).
 */
#define PP_LOOP_APPLY(N, ACT, A, B, M) PP_LOOP_##N(ACT, A, B, M)
#define PP_LOOP(N, ACT, A, B, M)       PP_LOOP_APPLY(N, ACT, A, B, M)

// #endregion PP_LOOP


// #region PP_ILOOP — inner count-up loop (separate family to avoid blue-paint)
/*
 * PP_ILOOP(N, ACT, A, B, M):
 *   Identical contract to PP_LOOP, but uses entirely separate macro names.
 *
 * WHY A SEPARATE FAMILY IS NEEDED:
 *   The C preprocessor "paints blue" a macro while it is being expanded.
 *   Any re-encounter of that same name during the rescan pass is left
 *   unexpanded (treated as a bare C identifier).
 *
 *   The outer col loop calls PP_LOOP(NR, GEMV_M_COL_ACT, ...).  While
 *   PP_LOOP_N is being expanded, its action GEMV_M_COL_ACT calls PP_LOOP
 *   again for the inner row-register loop.  That inner call can resolve
 *   to the same PP_LOOP_N token (e.g. both NR=2 and num_loads=2), which
 *   is still blue — the preprocessor then leaves PP_LOOP_2 as a plain
 *   identifier, causing an "implicit declaration of function" error.
 *
 *   PP_ILOOP_0 … PP_ILOOP_8 / PP_ILOOP_APPLY are never blue during an
 *   outer PP_LOOP pass, so the nesting works correctly.
 *
 * Range: N = 0 … 8  (max num_loads_per_MR = 8 for MR=64, d type: 64/8=8)
 */
#define PP_ILOOP_0(ACT, A, B, M)
#define PP_ILOOP_1(ACT, A, B, M)  ACT(A, B, M, 1)
#define PP_ILOOP_2(ACT, A, B, M)  PP_ILOOP_1(ACT, A, B, M)  ACT(A, B, M, 2)
#define PP_ILOOP_3(ACT, A, B, M)  PP_ILOOP_2(ACT, A, B, M)  ACT(A, B, M, 3)
#define PP_ILOOP_4(ACT, A, B, M)  PP_ILOOP_3(ACT, A, B, M)  ACT(A, B, M, 4)
#define PP_ILOOP_5(ACT, A, B, M)  PP_ILOOP_4(ACT, A, B, M)  ACT(A, B, M, 5)
#define PP_ILOOP_6(ACT, A, B, M)  PP_ILOOP_5(ACT, A, B, M)  ACT(A, B, M, 6)
#define PP_ILOOP_7(ACT, A, B, M)  PP_ILOOP_6(ACT, A, B, M)  ACT(A, B, M, 7)
#define PP_ILOOP_8(ACT, A, B, M)  PP_ILOOP_7(ACT, A, B, M)  ACT(A, B, M, 8)
#define PP_ILOOP_9(ACT, A, B, M)  PP_ILOOP_8(ACT, A, B, M)  ACT(A, B, M, 9)
#define PP_ILOOP_10(ACT, A, B, M) PP_ILOOP_9(ACT, A, B, M)  ACT(A, B, M, 10)
#define PP_ILOOP_11(ACT, A, B, M) PP_ILOOP_10(ACT, A, B, M) ACT(A, B, M, 11)
#define PP_ILOOP_12(ACT, A, B, M) PP_ILOOP_11(ACT, A, B, M) ACT(A, B, M, 12)
#define PP_ILOOP_13(ACT, A, B, M) PP_ILOOP_12(ACT, A, B, M) ACT(A, B, M, 13)
#define PP_ILOOP_14(ACT, A, B, M) PP_ILOOP_13(ACT, A, B, M) ACT(A, B, M, 14)
#define PP_ILOOP_15(ACT, A, B, M) PP_ILOOP_14(ACT, A, B, M) ACT(A, B, M, 15)
#define PP_ILOOP_16(ACT, A, B, M) PP_ILOOP_15(ACT, A, B, M) ACT(A, B, M, 16)
#define PP_ILOOP_17(ACT, A, B, M) PP_ILOOP_16(ACT, A, B, M) ACT(A, B, M, 17)
#define PP_ILOOP_18(ACT, A, B, M) PP_ILOOP_17(ACT, A, B, M) ACT(A, B, M, 18)
#define PP_ILOOP_19(ACT, A, B, M) PP_ILOOP_18(ACT, A, B, M) ACT(A, B, M, 19)
#define PP_ILOOP_20(ACT, A, B, M) PP_ILOOP_19(ACT, A, B, M) ACT(A, B, M, 20)
#define PP_ILOOP_21(ACT, A, B, M) PP_ILOOP_20(ACT, A, B, M) ACT(A, B, M, 21)
#define PP_ILOOP_22(ACT, A, B, M) PP_ILOOP_21(ACT, A, B, M) ACT(A, B, M, 22)
#define PP_ILOOP_23(ACT, A, B, M) PP_ILOOP_22(ACT, A, B, M) ACT(A, B, M, 23)
#define PP_ILOOP_24(ACT, A, B, M) PP_ILOOP_23(ACT, A, B, M) ACT(A, B, M, 24)

#define PP_ILOOP_APPLY(N, ACT, A, B, M)  PP_ILOOP_##N(ACT, A, B, M)
#define PP_ILOOP(N, ACT, A, B, M)        PP_ILOOP_APPLY(N, ACT, A, B, M)

// #endregion PP_ILOOP



/*
 * PP_JLOOP: third compile-time loop family (distinct from PP_LOOP / PP_ILOOP).
 * Needed to avoid blue-paint recursion: S_GEMV_N_Y_LOOP_ASM is called from
 * PP_ILOOP, and it in turn needs a column loop — using PP_ILOOP again would
 * leave the inner call unexpanded. PP_JLOOP uses the same semantics but a
 * different name so the preprocessor allows re-expansion.
 *   PP_JLOOP(N, ACT, A, B, M) -> ACT(A,B,M,1) ACT(A,B,M,2) ... ACT(A,B,M,N)
 */
#define PP_JLOOP_0(ACT, A, B, M)
#define PP_JLOOP_1(ACT, A, B, M)  ACT(A, B, M, 1)
#define PP_JLOOP_2(ACT, A, B, M)  PP_JLOOP_1(ACT, A, B, M)  ACT(A, B, M, 2)
#define PP_JLOOP_3(ACT, A, B, M)  PP_JLOOP_2(ACT, A, B, M)  ACT(A, B, M, 3)
#define PP_JLOOP_4(ACT, A, B, M)  PP_JLOOP_3(ACT, A, B, M)  ACT(A, B, M, 4)
#define PP_JLOOP_5(ACT, A, B, M)  PP_JLOOP_4(ACT, A, B, M)  ACT(A, B, M, 5)
#define PP_JLOOP_6(ACT, A, B, M)  PP_JLOOP_5(ACT, A, B, M)  ACT(A, B, M, 6)
#define PP_JLOOP_7(ACT, A, B, M)  PP_JLOOP_6(ACT, A, B, M)  ACT(A, B, M, 7)
#define PP_JLOOP_8(ACT, A, B, M)  PP_JLOOP_7(ACT, A, B, M)  ACT(A, B, M, 8)
#define PP_JLOOP_9(ACT, A, B, M)  PP_JLOOP_8(ACT, A, B, M)  ACT(A, B, M, 9)
#define PP_JLOOP_10(ACT, A, B, M) PP_JLOOP_9(ACT, A, B, M)  ACT(A, B, M, 10)
#define PP_JLOOP_APPLY(N, ACT, A, B, M)  PP_JLOOP_##N(ACT, A, B, M)
#define PP_JLOOP(N, ACT, A, B, M)        PP_JLOOP_APPLY(N, ACT, A, B, M)

// =============================================================================
// #region EPR_OF / IS_FRINGE — GEMV compile-time dispatch helpers
// =============================================================================

/*
 * EPR_OF(ch): expands to ELEM_PER_REG_<ch> (defined in the ISA block above).
 * IS_FRINGE_OF(MR, ch): expands to IS_FRINGE_<MR>_<ch>.
 *
 * Indirection macros so ## can live in a macro body rather than an argument
 * position (stray-## error), letting callers write EPR_OF(ch) inside other
 * macro bodies.
 */
#define EPR_OF(ch)            ELEM_PER_REG_##ch
#define IS_FRINGE_OF(MR, ch)  IS_FRINGE_##MR##_##ch

/*
 * IS_FRINGE_<MR>_<ch>: 1 if MR is NOT a multiple of ELEM_PER_REG_ch (masked load needed).
 * Coverage: all (MR, ch) pairs produced by GENERATE_* macros.
 */

/* d (AVX-512 EPR=8): full tiles are multiples of 8 */
#define IS_FRINGE_8_d   0
#define IS_FRINGE_16_d  0
#define IS_FRINGE_24_d  0
#define IS_FRINGE_32_d  0
#define IS_FRINGE_40_d  0
#define IS_FRINGE_48_d  0
#define IS_FRINGE_56_d  0
#define IS_FRINGE_64_d  0
/* fringe tiles */
#define IS_FRINGE_0_d   1
#define IS_FRINGE_1_d   1
#define IS_FRINGE_2_d   1
#define IS_FRINGE_3_d   1
#define IS_FRINGE_4_d   1
#define IS_FRINGE_5_d   1
#define IS_FRINGE_6_d   1
#define IS_FRINGE_7_d   1
#define IS_FRINGE_9_d   1
#define IS_FRINGE_10_d  1
#define IS_FRINGE_11_d  1
#define IS_FRINGE_12_d  1
#define IS_FRINGE_13_d  1
#define IS_FRINGE_14_d  1
#define IS_FRINGE_15_d  1
#define IS_FRINGE_23_d  1
#define IS_FRINGE_31_d  1
#define IS_FRINGE_39_d  1
#define IS_FRINGE_47_d  1
#define IS_FRINGE_55_d  1
#define IS_FRINGE_63_d  1

/* s (AVX-512 EPR=16): full tiles are multiples of 16 */
#define IS_FRINGE_16_s  0
#define IS_FRINGE_32_s  0
#define IS_FRINGE_48_s  0
#define IS_FRINGE_64_s  0
#define IS_FRINGE_80_s  0
#define IS_FRINGE_96_s  0
/* fringe tiles */
#define IS_FRINGE_0_s  1
#define IS_FRINGE_1_s  1
#define IS_FRINGE_2_s  1
#define IS_FRINGE_3_s  1
#define IS_FRINGE_4_s  1
#define IS_FRINGE_5_s  1
#define IS_FRINGE_6_s  1
#define IS_FRINGE_7_s  1
#define IS_FRINGE_8_s  1
#define IS_FRINGE_9_s  1
#define IS_FRINGE_10_s  1
#define IS_FRINGE_11_s  1
#define IS_FRINGE_12_s  1
#define IS_FRINGE_13_s  1
#define IS_FRINGE_14_s  1
#define IS_FRINGE_15_s  1
#define IS_FRINGE_31_s  1
#define IS_FRINGE_47_s  1
#define IS_FRINGE_63_s  1
#define IS_FRINGE_79_s  1
#define IS_FRINGE_95_s  1

/* z (AVX-512 EPR=4): full tiles are multiples of 4 */
#define IS_FRINGE_4_z   0
#define IS_FRINGE_8_z   0
#define IS_FRINGE_12_z  0
#define IS_FRINGE_16_z  0
#define IS_FRINGE_20_z  0
#define IS_FRINGE_24_z  0
#define IS_FRINGE_28_z  0
#define IS_FRINGE_32_z  0
/* fringe tiles */
#define IS_FRINGE_3_z   1
#define IS_FRINGE_7_z   1
#define IS_FRINGE_11_z  1
#define IS_FRINGE_15_z  1
#define IS_FRINGE_19_z  1
#define IS_FRINGE_23_z  1
#define IS_FRINGE_27_z  1
#define IS_FRINGE_31_z  1

/* c (AVX-512 EPR=8): same pattern as d */
#define IS_FRINGE_8_c   0
#define IS_FRINGE_16_c  0
#define IS_FRINGE_24_c  0
#define IS_FRINGE_32_c  0
#define IS_FRINGE_40_c  0
#define IS_FRINGE_48_c  0
/* fringe tiles */
#define IS_FRINGE_7_c   1
#define IS_FRINGE_15_c  1
#define IS_FRINGE_23_c  1
#define IS_FRINGE_31_c  1
#define IS_FRINGE_39_c  1
#define IS_FRINGE_47_c  1

/* AVX2 overrides (EPR_d=4, EPR_s=8, EPR_c=4, EPR_z=2) */
#if ARCH_SIMD_BITS == 256
/* d (AVX2 EPR=4) */
#undef IS_FRINGE_4_d
#undef IS_FRINGE_8_d
#undef IS_FRINGE_12_d
#undef IS_FRINGE_16_d
#undef IS_FRINGE_24_d
#undef IS_FRINGE_32_d
#define IS_FRINGE_4_d   0
#define IS_FRINGE_8_d   0
#define IS_FRINGE_12_d  0  /* AVX2 d N-kernel fringe MR=12 */
#define IS_FRINGE_16_d  0
#define IS_FRINGE_20_d  0  /* AVX2 d N-kernel MR=20 */
#define IS_FRINGE_24_d  0
#define IS_FRINGE_32_d  0
/* fringe tiles (MR not a multiple of EPR=4) */
#define IS_FRINGE_3_d   1
#define IS_FRINGE_7_d   1
#define IS_FRINGE_11_d  1  /* fringe tile of d MR=20 (m_idx=7) */
#define IS_FRINGE_15_d  1
#define IS_FRINGE_19_d  1  /* fringe of MR=20 */
#define IS_FRINGE_23_d  1
#define IS_FRINGE_31_d  1

/* s (AVX2 EPR=8) */
#undef IS_FRINGE_8_s
#undef IS_FRINGE_16_s
#undef IS_FRINGE_32_s
#undef IS_FRINGE_48_s
#define IS_FRINGE_8_s   0
#define IS_FRINGE_16_s  0
#define IS_FRINGE_24_s  0  /* AVX2 s T-kernel MR=24 */
#define IS_FRINGE_32_s  0
#define IS_FRINGE_40_s  0  /* AVX2 s N-kernel MR=40 */
#define IS_FRINGE_48_s  0
/* fringe tiles */
#define IS_FRINGE_7_s   1
#define IS_FRINGE_15_s  1
#define IS_FRINGE_23_s  1  /* fringe of MR=24 */
#define IS_FRINGE_31_s  1
#define IS_FRINGE_39_s  1  /* fringe of MR=40 */
#define IS_FRINGE_47_s  1

/* z (AVX2 EPR=2) */
#undef IS_FRINGE_4_z
#undef IS_FRINGE_8_z
#undef IS_FRINGE_12_z
#undef IS_FRINGE_16_z
#define IS_FRINGE_2_z   0  /* AVX2 z N-kernel MR=2 */
#define IS_FRINGE_4_z   0
#define IS_FRINGE_6_z   0  /* AVX2 z N-kernel MR=10, partial */
#define IS_FRINGE_8_z   0
#define IS_FRINGE_10_z  0  /* AVX2 z N-kernel MR=10 / T-kernel MR=10 */
#define IS_FRINGE_12_z  0
#define IS_FRINGE_16_z  0
/* fringe tiles */
#define IS_FRINGE_1_z   1
#define IS_FRINGE_3_z   1
#define IS_FRINGE_5_z   1
#define IS_FRINGE_7_z   1
#define IS_FRINGE_9_z   1  /* fringe of MR=10 */
#define IS_FRINGE_11_z  1
#define IS_FRINGE_15_z  1

/* c (AVX2 EPR=4) */
#undef IS_FRINGE_4_c
#undef IS_FRINGE_8_c
#undef IS_FRINGE_12_c
#undef IS_FRINGE_16_c
#undef IS_FRINGE_24_c
#define IS_FRINGE_4_c   0
#define IS_FRINGE_8_c   0
#define IS_FRINGE_12_c  0
#define IS_FRINGE_16_c  0
#define IS_FRINGE_20_c  0  /* AVX2 c N-kernel MR=20 */
#define IS_FRINGE_24_c  0
/* fringe tiles */
#define IS_FRINGE_3_c   1
#define IS_FRINGE_7_c   1
#define IS_FRINGE_11_c  1
#define IS_FRINGE_15_c  1
#define IS_FRINGE_19_c  1  /* fringe of MR=20 */
#define IS_FRINGE_23_c  1
#endif /* ARCH_SIMD_BITS == 256 */

// #endregion EPR_OF / IS_FRINGE

// #region PP_ADD — compile-time integer addition (lookup table)
/*
 * PP_ADD(A, B): Expands to the integer value A+B at preprocessing time.
 *
 * Analogous to BOOST_PP_ADD. Because the C preprocessor cannot perform
 * arithmetic, this is implemented as a two-level lookup table:
 *
 *   PP_ADD(A, B) -> PP_ADD_##A##_##B
 *
 * The table covers:
 *   - A ∈ {0..8}  (NR column counts used in these kernels)
 *   - B ∈ {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96}
 *     (MR block-size multiples used in these kernels)
 *
 * This intentionally covers only the values that arise in GEMV kernel
 * generation — not general-purpose addition — keeping the table small.
 *
 * Example:
 *   PP_ADD(4, 8)   -> 12
 *   PP_ADD(1, 40)  -> 41
 */
#define PP_ADD_IMPL(A, B) PP_ADD_##A##_##B
#define PP_ADD(A, B)      PP_ADD_IMPL(A, B)

/* --- A = 0 --- */
#define PP_ADD_0_0   0
#define PP_ADD_0_1   1
#define PP_ADD_0_2   2
#define PP_ADD_0_3   3
#define PP_ADD_0_4   4
#define PP_ADD_0_5   5
#define PP_ADD_0_6   6
#define PP_ADD_0_7   7
#define PP_ADD_0_8   8
#define PP_ADD_0_12  12
#define PP_ADD_0_16  16
#define PP_ADD_0_20  20
#define PP_ADD_0_24  24
#define PP_ADD_0_28  28
#define PP_ADD_0_32  32
#define PP_ADD_0_36  36
#define PP_ADD_0_40  40
#define PP_ADD_0_48  48
#define PP_ADD_0_56  56
#define PP_ADD_0_64  64
#define PP_ADD_0_80  80
#define PP_ADD_0_96  96

/* --- A = 1 --- */
#define PP_ADD_1_0   1
#define PP_ADD_1_1   2
#define PP_ADD_1_2   3
#define PP_ADD_1_3   4
#define PP_ADD_1_4   5
#define PP_ADD_1_5   6
#define PP_ADD_1_6   7
#define PP_ADD_1_7   8
#define PP_ADD_1_8   9
#define PP_ADD_1_12  13
#define PP_ADD_1_16  17
#define PP_ADD_1_20  21
#define PP_ADD_1_24  25
#define PP_ADD_1_28  29
#define PP_ADD_1_32  33
#define PP_ADD_1_36  37
#define PP_ADD_1_40  41
#define PP_ADD_1_48  49
#define PP_ADD_1_56  57
#define PP_ADD_1_64  65
#define PP_ADD_1_80  81
#define PP_ADD_1_96  97

/* --- A = 2 --- */
#define PP_ADD_2_0   2
#define PP_ADD_2_1   3
#define PP_ADD_2_2   4
#define PP_ADD_2_3   5
#define PP_ADD_2_4   6
#define PP_ADD_2_5   7
#define PP_ADD_2_6   8
#define PP_ADD_2_7   9
#define PP_ADD_2_8   10
#define PP_ADD_2_12  14
#define PP_ADD_2_16  18
#define PP_ADD_2_20  22
#define PP_ADD_2_24  26
#define PP_ADD_2_28  30
#define PP_ADD_2_32  34
#define PP_ADD_2_36  38
#define PP_ADD_2_40  42
#define PP_ADD_2_48  50
#define PP_ADD_2_56  58
#define PP_ADD_2_64  66
#define PP_ADD_2_80  82
#define PP_ADD_2_96  98

/* --- A = 3 --- */
#define PP_ADD_3_0   3
#define PP_ADD_3_1   4
#define PP_ADD_3_2   5
#define PP_ADD_3_3   6
#define PP_ADD_3_4   7
#define PP_ADD_3_5   8
#define PP_ADD_3_6   9
#define PP_ADD_3_7   10
#define PP_ADD_3_8   11
#define PP_ADD_3_12  15
#define PP_ADD_3_16  19
#define PP_ADD_3_20  23
#define PP_ADD_3_24  27
#define PP_ADD_3_28  31
#define PP_ADD_3_32  35
#define PP_ADD_3_36  39
#define PP_ADD_3_40  43
#define PP_ADD_3_48  51
#define PP_ADD_3_56  59
#define PP_ADD_3_64  67
#define PP_ADD_3_80  83
#define PP_ADD_3_96  99

/* --- A = 4 --- */
#define PP_ADD_4_0   4
#define PP_ADD_4_1   5
#define PP_ADD_4_2   6
#define PP_ADD_4_3   7
#define PP_ADD_4_4   8
#define PP_ADD_4_5   9
#define PP_ADD_4_6   10
#define PP_ADD_4_7   11
#define PP_ADD_4_8   12
#define PP_ADD_4_12  16
#define PP_ADD_4_16  20
#define PP_ADD_4_20  24
#define PP_ADD_4_24  28
#define PP_ADD_4_28  32
#define PP_ADD_4_32  36
#define PP_ADD_4_36  40
#define PP_ADD_4_40  44
#define PP_ADD_4_48  52
#define PP_ADD_4_56  60
#define PP_ADD_4_64  68
#define PP_ADD_4_80  84
#define PP_ADD_4_96  100

/* --- A = 5 --- */
#define PP_ADD_5_0   5
#define PP_ADD_5_1   6
#define PP_ADD_5_2   7
#define PP_ADD_5_3   8
#define PP_ADD_5_4   9
#define PP_ADD_5_5   10
#define PP_ADD_5_6   11
#define PP_ADD_5_7   12
#define PP_ADD_5_8   13
#define PP_ADD_5_12  17
#define PP_ADD_5_16  21
#define PP_ADD_5_20  25
#define PP_ADD_5_24  29
#define PP_ADD_5_28  33
#define PP_ADD_5_32  37
#define PP_ADD_5_36  41
#define PP_ADD_5_40  45
#define PP_ADD_5_48  53
#define PP_ADD_5_56  61
#define PP_ADD_5_64  69
#define PP_ADD_5_80  85
#define PP_ADD_5_96  101

/* --- A = 6 --- */
#define PP_ADD_6_0   6
#define PP_ADD_6_1   7
#define PP_ADD_6_2   8
#define PP_ADD_6_3   9
#define PP_ADD_6_4   10
#define PP_ADD_6_5   11
#define PP_ADD_6_6   12
#define PP_ADD_6_7   13
#define PP_ADD_6_8   14
#define PP_ADD_6_12  18
#define PP_ADD_6_16  22
#define PP_ADD_6_20  26
#define PP_ADD_6_24  30
#define PP_ADD_6_28  34
#define PP_ADD_6_32  38
#define PP_ADD_6_36  42
#define PP_ADD_6_40  46
#define PP_ADD_6_48  54
#define PP_ADD_6_56  62
#define PP_ADD_6_64  70
#define PP_ADD_6_80  86
#define PP_ADD_6_96  102

/* --- A = 7 --- */
#define PP_ADD_7_0   7
#define PP_ADD_7_1   8
#define PP_ADD_7_2   9
#define PP_ADD_7_3   10
#define PP_ADD_7_4   11
#define PP_ADD_7_5   12
#define PP_ADD_7_6   13
#define PP_ADD_7_7   14
#define PP_ADD_7_8   15
#define PP_ADD_7_12  19
#define PP_ADD_7_16  23
#define PP_ADD_7_20  27
#define PP_ADD_7_24  31
#define PP_ADD_7_28  35
#define PP_ADD_7_32  39
#define PP_ADD_7_36  43
#define PP_ADD_7_40  47
#define PP_ADD_7_48  55
#define PP_ADD_7_56  63
#define PP_ADD_7_64  71
#define PP_ADD_7_80  87
#define PP_ADD_7_96  103

/* --- A = 8 --- */
#define PP_ADD_8_0   8
#define PP_ADD_8_1   9
#define PP_ADD_8_2   10
#define PP_ADD_8_3   11
#define PP_ADD_8_4   12
#define PP_ADD_8_5   13
#define PP_ADD_8_6   14
#define PP_ADD_8_7   15
#define PP_ADD_8_8   16
#define PP_ADD_8_12  20
#define PP_ADD_8_16  24
#define PP_ADD_8_20  28
#define PP_ADD_8_24  32
#define PP_ADD_8_28  36
#define PP_ADD_8_32  40
#define PP_ADD_8_36  44
#define PP_ADD_8_40  48
#define PP_ADD_8_48  56
#define PP_ADD_8_56  64
#define PP_ADD_8_64  72
#define PP_ADD_8_80  88
#define PP_ADD_8_96  104

// #endregion PP_ADD


// #region PP_IF — compile-time conditional (Boost.PP_IF equivalent)
/*
 * PP_IF(COND, THEN_MACRO, ...):
 *   If COND == 1: expands THEN_MACRO##1(__VA_ARGS__)
 *   If COND == 0: expands THEN_MACRO##0(__VA_ARGS__)
 *
 * Analogous to BOOST_PP_IF. The trick is that COND must be a preprocessor
 * integer literal (0 or 1), not a C 'const' variable. Provide a companion
 * THEN_MACRO_0 (no-op) and THEN_MACRO_1 (active body) and PP_IF will select
 * the correct one without any runtime branch.
 *
 * Canonical usage pattern:
 *
 *   #define MY_FRINGE_0(...)   // nothing — no fringe
 *   #define MY_FRINGE_1(...)   <fringe load/store code>
 *
 *   PP_IF(IS_FRINGE, MY_FRINGE_, arg1, arg2)
 *   // expands to MY_FRINGE_0(...) or MY_FRINGE_1(...) depending on IS_FRINGE
 *
 * The two-level indirection (PP_IF_IMPL_) ensures that COND is fully
 * expanded before concatenation, matching Boost.PP semantics.
 */
#define PP_IF_IMPL_(COND, MACRO, ...) MACRO##COND(__VA_ARGS__)
#define PP_IF(COND, MACRO, ...)       PP_IF_IMPL_(COND, MACRO, __VA_ARGS__)

// #endregion PP_IF


// #region PP_DIV — compile-time integer division (lookup table)
/*
 * PP_DIV(A, B): Expands to the integer value A/B (integer division) at
 * preprocessing time.
 *
 * Analogous to BOOST_PP_DIV. Implemented as a lookup table scoped to the
 * (MR, EPR) pairs that actually arise in GEMV kernel generation:
 *
 *   A ∈ {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96}  — MR values
 *   B ∈ {8, 16}  — EPR (elements per register) for d and s precision
 *
 * Result = num_loads_per_MR = MR / ELEM_PER_REG, needed so that
 * PP_LOOP(PP_DIV(MR, EPR), ...) can be token-pasted to the correct chain.
 *
 * Example:
 *   PP_DIV(40, 8)  -> 5     (40 doubles fit in 5 ZMM registers)
 *   PP_DIV(32, 16) -> 2     (32 floats fit in 2 ZMM registers)
 */
#define PP_DIV_IMPL(A, B) PP_DIV_##A##_##B
#define PP_DIV(A, B)      PP_DIV_IMPL(A, B)

/* B = 8  (doubles: 8 per ZMM) */
#define PP_DIV_0_8    0
#define PP_DIV_1_8    0
#define PP_DIV_2_8    0
#define PP_DIV_3_8    0
#define PP_DIV_4_8    0
#define PP_DIV_5_8    0
#define PP_DIV_6_8    0
#define PP_DIV_7_8    0
#define PP_DIV_8_8    1
#define PP_DIV_9_8    1
#define PP_DIV_10_8   1
#define PP_DIV_11_8   1
#define PP_DIV_12_8   1
#define PP_DIV_13_8   1
#define PP_DIV_14_8   1
#define PP_DIV_15_8   1
#define PP_DIV_16_8   2
#define PP_DIV_17_8   2
#define PP_DIV_18_8   2
#define PP_DIV_19_8   2
#define PP_DIV_20_8   2
#define PP_DIV_21_8   2
#define PP_DIV_22_8   2
#define PP_DIV_23_8   2
#define PP_DIV_24_8   3
#define PP_DIV_25_8   3
#define PP_DIV_26_8   3
#define PP_DIV_27_8   3
#define PP_DIV_28_8   3
#define PP_DIV_29_8   3
#define PP_DIV_30_8   3
#define PP_DIV_31_8   3
#define PP_DIV_32_8   4
#define PP_DIV_33_8   4
#define PP_DIV_34_8   4
#define PP_DIV_35_8   4
#define PP_DIV_36_8   4
#define PP_DIV_37_8   4
#define PP_DIV_38_8   4
#define PP_DIV_39_8   4
#define PP_DIV_40_8   5
#define PP_DIV_41_8   5
#define PP_DIV_42_8   5
#define PP_DIV_43_8   5
#define PP_DIV_44_8   5
#define PP_DIV_45_8   5
#define PP_DIV_46_8   5
#define PP_DIV_47_8   5
#define PP_DIV_48_8   6

#define PP_DIV_55_8   6
#define PP_DIV_56_8   7
#define PP_DIV_63_8   7
#define PP_DIV_64_8   8

/* B = 16  (floats: 16 per ZMM) */
#define PP_DIV_0_16   0
#define PP_DIV_1_16   0
#define PP_DIV_2_16   0
#define PP_DIV_3_16   0
#define PP_DIV_4_16   0
#define PP_DIV_5_16   0
#define PP_DIV_6_16   0
#define PP_DIV_7_16   0
#define PP_DIV_8_16   0
#define PP_DIV_9_16   0
#define PP_DIV_10_16  0
#define PP_DIV_11_16  0
#define PP_DIV_12_16  0
#define PP_DIV_13_16  0
#define PP_DIV_14_16  0
#define PP_DIV_15_16  0
#define PP_DIV_16_16  1
#define PP_DIV_17_16  1
#define PP_DIV_18_16  1
#define PP_DIV_19_16  1
#define PP_DIV_20_16  1
#define PP_DIV_21_16  1
#define PP_DIV_22_16  1
#define PP_DIV_23_16  1
#define PP_DIV_24_16  1
#define PP_DIV_25_16  1
#define PP_DIV_26_16  1
#define PP_DIV_27_16  1
#define PP_DIV_28_16  1
#define PP_DIV_29_16  1
#define PP_DIV_30_16  1
#define PP_DIV_31_16  1
#define PP_DIV_32_16  2
#define PP_DIV_33_16  2
#define PP_DIV_34_16  2
#define PP_DIV_35_16  2
#define PP_DIV_36_16  2
#define PP_DIV_37_16  2
#define PP_DIV_38_16  2
#define PP_DIV_39_16  2
#define PP_DIV_40_16  2
#define PP_DIV_41_16  2
#define PP_DIV_42_16  2
#define PP_DIV_43_16  2
#define PP_DIV_44_16  2
#define PP_DIV_45_16  2
#define PP_DIV_46_16  2
#define PP_DIV_47_16  2
#define PP_DIV_48_16  3

#define PP_DIV_63_16  3
#define PP_DIV_64_16  4
#define PP_DIV_79_16  4
#define PP_DIV_80_16  5
#define PP_DIV_95_16  5
#define PP_DIV_96_16  6

/* B = 4  (dcomplex: 4 per ZMM) */
#define PP_DIV_4_4    1
#define PP_DIV_8_4    2
#define PP_DIV_12_4   3
#define PP_DIV_16_4   4
#define PP_DIV_20_4   5
#define PP_DIV_24_4   6
#define PP_DIV_28_4   7
#define PP_DIV_32_4   8

#define PP_DIV_3_4    0
#define PP_DIV_7_4    1
#define PP_DIV_11_4   2
#define PP_DIV_15_4   3
#define PP_DIV_19_4   4
#define PP_DIV_23_4   5
#define PP_DIV_27_4   6
#define PP_DIV_31_4   7

/* B = 2  (scomplex: 2 per ZMM — treating element as complex pair) */
/* (Included for completeness; kernels may not use this directly) */
#define PP_DIV_2_2    1
#define PP_DIV_4_2    2
#define PP_DIV_6_2    3
#define PP_DIV_8_2    4

/* B = 4  (AVX2 doubles/scomplex: 4 per YMM) — extends the ZMM-dcomplex B=4 table above */
/* Full tiles (multiples of 4) used by AVX2 d/c kernels with MR=40 */
#define PP_DIV_36_4   9
#define PP_DIV_40_4   10
#define PP_DIV_44_4   11
#define PP_DIV_48_4   12
#define PP_DIV_80_4   20
/* Fringe tiles (MR = N*4-1) used by GENERATE_d/c_KERNELS_40_N fringe functions */
#define PP_DIV_35_4   8   /* floor(35/4) */
#define PP_DIV_39_4   9   /* floor(39/4) */
#define PP_DIV_43_4   10  /* floor(43/4) */
#define PP_DIV_47_4   11  /* floor(47/4) */

/* B = 8 (AVX2 float EPR=8) — adds MR=80 and fringe MR=79 beyond existing B=8 table */
#define PP_DIV_79_8   9   /* floor(79/8) — fringe of MR=80 */
#define PP_DIV_80_8   10

/* B = 2 (AVX2 dcomplex EPR_z=2) — extends the B=2 table for larger MR values */
#define PP_DIV_1_2    0   /* floor(1/2)  — fringe of MR=2 */
#define PP_DIV_3_2    1   /* floor(3/2)  — fringe of MR=4 */
#define PP_DIV_5_2    2   /* floor(5/2) */
#define PP_DIV_7_2    3   /* floor(7/2) */
#define PP_DIV_9_2    4   /* floor(9/2) — fringe of MR=10 */
#define PP_DIV_10_2   5
#define PP_DIV_11_2   5   /* floor(11/2) */
#define PP_DIV_12_2   6
#define PP_DIV_13_2   6   /* floor(13/2) */
#define PP_DIV_14_2   7
#define PP_DIV_15_2   7   /* floor(15/2) */
#define PP_DIV_16_2   8
#define PP_DIV_17_2   8   /* floor(17/2) */
#define PP_DIV_18_2   9
#define PP_DIV_19_2   9   /* floor(19/2) */
#define PP_DIV_20_2   10

// #endregion PP_DIV


// #region PP_MUL — compile-time integer multiplication (lookup table)
/*
 * PP_MUL(A, B): Expands to the integer value A*B at preprocessing time.
 *
 * Analogous to BOOST_PP_MUL. Implemented as a two-level lookup table
 * scoped to the (NR, MR-step) pairs that appear in GEMV kernel generation:
 *
 *   A ∈ {1..8}           — NR column tile sizes
 *   B ∈ {4, 8, 16, 32}  — typical MR step sizes / sub-register strides
 *
 * The general formula PP_MUL(A, B) = PP_MUL_##A##_##B.
 *
 * Example:
 *   PP_MUL(4, 8)  -> 32
 *   PP_MUL(3, 16) -> 48
 */
#define PP_MUL_IMPL(A, B) PP_MUL_##A##_##B
#define PP_MUL(A, B)      PP_MUL_IMPL(A, B)

/* B = 1 */
#define PP_MUL_1_1   1
#define PP_MUL_2_1   2
#define PP_MUL_3_1   3
#define PP_MUL_4_1   4
#define PP_MUL_5_1   5
#define PP_MUL_6_1   6
#define PP_MUL_7_1   7
#define PP_MUL_8_1   8

/* B = 2 */
#define PP_MUL_1_2   2
#define PP_MUL_2_2   4
#define PP_MUL_3_2   6
#define PP_MUL_4_2   8
#define PP_MUL_5_2   10
#define PP_MUL_6_2   12
#define PP_MUL_7_2   14
#define PP_MUL_8_2   16

/* B = 3 */
#define PP_MUL_1_3   3
#define PP_MUL_2_3   6
#define PP_MUL_3_3   9
#define PP_MUL_4_3   12
#define PP_MUL_5_3   15
#define PP_MUL_6_3   18
#define PP_MUL_7_3   21
#define PP_MUL_8_3   24

/* B = 4 */
#define PP_MUL_1_4   4
#define PP_MUL_2_4   8
#define PP_MUL_3_4   12
#define PP_MUL_4_4   16
#define PP_MUL_5_4   20
#define PP_MUL_6_4   24
#define PP_MUL_7_4   28
#define PP_MUL_8_4   32

/* B = 5 */
#define PP_MUL_1_5   5
#define PP_MUL_2_5   10
#define PP_MUL_3_5   15
#define PP_MUL_4_5   20
#define PP_MUL_5_5   25
#define PP_MUL_6_5   30
#define PP_MUL_7_5   35
#define PP_MUL_8_5   40

/* B = 6 */
#define PP_MUL_1_6   6
#define PP_MUL_2_6   12
#define PP_MUL_3_6   18
#define PP_MUL_4_6   24
#define PP_MUL_5_6   30
#define PP_MUL_6_6   36
#define PP_MUL_7_6   42
#define PP_MUL_8_6   48

/* B = 7 */
#define PP_MUL_1_7   7
#define PP_MUL_2_7   14
#define PP_MUL_3_7   21
#define PP_MUL_4_7   28
#define PP_MUL_5_7   35
#define PP_MUL_6_7   42
#define PP_MUL_7_7   49
#define PP_MUL_8_7   56

/* B = 8 */
#define PP_MUL_1_8   8
#define PP_MUL_2_8   16
#define PP_MUL_3_8   24
#define PP_MUL_4_8   32
#define PP_MUL_5_8   40
#define PP_MUL_6_8   48
#define PP_MUL_7_8   56
#define PP_MUL_8_8   64

/* B = 12 */
#define PP_MUL_1_12  12
#define PP_MUL_2_12  24
#define PP_MUL_3_12  36
#define PP_MUL_4_12  48
#define PP_MUL_5_12  60
#define PP_MUL_6_12  72
#define PP_MUL_7_12  84
#define PP_MUL_8_12  96

/* B = 16 */
#define PP_MUL_1_16  16
#define PP_MUL_2_16  32
#define PP_MUL_3_16  48
#define PP_MUL_4_16  64
#define PP_MUL_5_16  80
#define PP_MUL_6_16  96
#define PP_MUL_7_16  112
#define PP_MUL_8_16  128

/* B = 20 */
#define PP_MUL_1_20  20
#define PP_MUL_2_20  40
#define PP_MUL_3_20  60
#define PP_MUL_4_20  80
#define PP_MUL_5_20  100
#define PP_MUL_6_20  120
#define PP_MUL_7_20  140
#define PP_MUL_8_20  160

/* B = 24 */
#define PP_MUL_1_24  24
#define PP_MUL_2_24  48
#define PP_MUL_3_24  72
#define PP_MUL_4_24  96
#define PP_MUL_5_24  120
#define PP_MUL_6_24  144
#define PP_MUL_7_24  168
#define PP_MUL_8_24  192

/* B = 32 */
#define PP_MUL_1_32  32
#define PP_MUL_2_32  64
#define PP_MUL_3_32  96
#define PP_MUL_4_32  128
#define PP_MUL_5_32  160
#define PP_MUL_6_32  192
#define PP_MUL_7_32  224
#define PP_MUL_8_32  256

/* B = 40 */
#define PP_MUL_1_40  40
#define PP_MUL_2_40  80
#define PP_MUL_3_40  120
#define PP_MUL_4_40  160
#define PP_MUL_5_40  200
#define PP_MUL_6_40  240
#define PP_MUL_7_40  280
#define PP_MUL_8_40  320

/* B = 48 */
#define PP_MUL_1_48  48
#define PP_MUL_2_48  96
#define PP_MUL_3_48  144
#define PP_MUL_4_48  192
#define PP_MUL_5_48  240
#define PP_MUL_6_48  288
#define PP_MUL_7_48  336
#define PP_MUL_8_48  384

/* B = 64 */
#define PP_MUL_1_64  64
#define PP_MUL_2_64  128
#define PP_MUL_3_64  192
#define PP_MUL_4_64  256
#define PP_MUL_5_64  320
#define PP_MUL_6_64  384
#define PP_MUL_7_64  448
#define PP_MUL_8_64  512

/* B = 80 */
#define PP_MUL_1_80  80
#define PP_MUL_2_80  160
#define PP_MUL_3_80  240
#define PP_MUL_4_80  320
#define PP_MUL_5_80  400
#define PP_MUL_6_80  480
#define PP_MUL_7_80  560
#define PP_MUL_8_80  640

/* B = 96 */
#define PP_MUL_1_96  96
#define PP_MUL_2_96  192
#define PP_MUL_3_96  288
#define PP_MUL_4_96  384
#define PP_MUL_5_96  480
#define PP_MUL_6_96  576
#define PP_MUL_7_96  672
#define PP_MUL_8_96  768

// #endregion PP_MUL


/*
 * PP_DEC1(n): expand to n-1 as a literal integer token (1..16 -> 0..15).
 * Used to convert 1-based PP_JLOOP counters to 0-based ZMM indices before
 * passing them through ZMM_E so the token-paste inside ZMM sees a literal.
 */
#define PP_DEC1_IMPL(n) PP_DEC1_##n
#define PP_DEC1(n)      PP_DEC1_IMPL(n)
#define PP_DEC1_1    0
#define PP_DEC1_2    1
#define PP_DEC1_3    2
#define PP_DEC1_4    3
#define PP_DEC1_5    4
#define PP_DEC1_6    5
#define PP_DEC1_7    6
#define PP_DEC1_8    7
#define PP_DEC1_9    8
#define PP_DEC1_10   9
#define PP_DEC1_11   10
#define PP_DEC1_12   11
#define PP_DEC1_13   12
#define PP_DEC1_14   13
#define PP_DEC1_15   14
#define PP_DEC1_16   15

// #endregion compile time loops