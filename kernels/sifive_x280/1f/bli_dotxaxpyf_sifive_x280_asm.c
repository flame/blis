/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, SiFive, Inc.

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

#include "blis.h"
#include "../riscv_cmul_macros_asm.h"
#include <math.h>
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>

#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define FMUL "fmul.s "
#define VLE "vle32.v "
#define VLSE "vlse32.v "
#define VSE "vse32.v "
#define VSSE "vsse32.v "

void bli_sdotxaxpyf_sifive_x280_asm(
             conj_t           conjat,
             conj_t           conja,
             conj_t           conjw,
             conj_t           conjx,
             dim_t            m,
             dim_t            b,
       const void*   restrict alpha_,
       const void*   restrict a_, inc_t inca, inc_t lda,
       const void*   restrict w_, inc_t incw,
       const void*   restrict x_, inc_t incx,
       const void*   restrict beta_,
             void*   restrict y_, inc_t incy,
             void*   restrict z_, inc_t incz,
       const cntx_t* restrict cntx
                             ) {
  (void)conjat;
  (void)conja;
  (void)conjw;
  (void)conjx;
  (void)cntx;
  const float *restrict alpha = alpha_;
  const float *restrict beta = beta_;
  const float *restrict a = a_;
  const float *restrict w = w_;
  const float *restrict x = x_;
  float *restrict y = y_;
  float *restrict z = z_;

  if (b == 0)
    return;
  else if (m == 0 || *alpha == 0.f) {
    // scale y by beta
    if (*beta == 0.f)
        bli_ssetv_sifive_x280_asm(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
    else
        bli_sscalv_sifive_x280_intr(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
    return;
  }

  __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
  __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
  inca *= FLT_SIZE;
  lda *= FLT_SIZE;
  incw *= FLT_SIZE;
  incx *= FLT_SIZE;
  incy *= FLT_SIZE;
  incz *= FLT_SIZE;
  inc_t a_bump = 5 * lda;
  while (b >= 5) {
    // compute dot product of w with 5 rows of a
    const float* restrict w_tmp = w;
    const float* restrict z_tmp = z;
    const float* restrict a_col = a;
    size_t avl = m;
    bool first = true;
    while (avl) {
      const float* restrict a_row = a_col;
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
      if (incw == FLT_SIZE)
        __asm__(VLE "v28, (%0)" : : "r"(w_tmp));
      else
        __asm__(VLSE "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
      if (inca == FLT_SIZE) {
        // a unit stride
        if (first) {
          __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmul.vv v0, v24, v28");
          __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmul.vv v4, v24, v28");
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmul.vv v8, v24, v28");
          __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmul.vv v12, v24, v28");
          __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmul.vv v16, v24, v28");
          first = false;
        }
        else {
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmacc.vv v0, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmacc.vv v4, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmacc.vv v8, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmacc.vv v12, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmacc.vv v16, v24, v28");
        }
      } // end a unit stride
      else {
        // a non-unit stride
        if (first) {
          __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmul.vv v0, v24, v28");
          __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmul.vv v4, v24, v28");
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmul.vv v8, v24, v28");
          __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmul.vv v12, v24, v28");
          __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmul.vv v16, v24, v28");
          first = false;
        }
        else {
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmacc.vv v0, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmacc.vv v4, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmacc.vv v8, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmacc.vv v12, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmacc.vv v16, v24, v28");
        }
      } // end a non-unit stride

      if (incz == FLT_SIZE) {
        __asm__(VLE "v24, (%0)" : : "r"(z_tmp));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSE "v24, (%0)" : : "r"(z_tmp));
      } else {
        __asm__(VLSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
      }

      __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incx));
      __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
      __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
      avl -= vl;
    }

    __asm__("vmv.s.x v31, x0");

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v0, v0, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.f) {
      __asm__("vfmul.vf v0, v0, ft10");
      __asm__(VSE "v0, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v0");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v4, v4, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.f) {
      __asm__("vfmul.vf v4, v4, ft10");
      __asm__(VSE "v4, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v4");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v8, v8, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.f) {
      __asm__("vfmul.vf v8, v8, ft10");
      __asm__(VSE "v8, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v8");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v12, v12, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.f) {
      __asm__("vfmul.vf v12, v12, ft10");
      __asm__(VSE "v12, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v12");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v16, v16, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.f) {
      __asm__("vfmul.vf v16, v16, ft10");
      __asm__(VSE "v16, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v16");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
    b -= 5;
  }

  if (b > 0) {
    const float* restrict w_tmp = w;
    const float* restrict z_tmp = z;
    const float* restrict a_col;
    __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
    __asm__("add %0, %0, %1" : "+r"(x) : "r"((b - 1) * incx));
    size_t avl = m;
    bool first = true;
    while (avl) {
      const float* restrict a_row = a_col;
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
      if (incw == FLT_SIZE)
        __asm__(VLE "v28, (%0)" : : "r"(w_tmp));
      else
        __asm__(VLSE "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
      __asm__("vmv.v.i v20, 0");
      if (inca == FLT_SIZE) {
        // a unit stride
        if (first) {
          switch (b) {
          case 4:
            __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmul.vv v12, v24, v28");
          case 3:
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmul.vv v8, v24, v28");
          case 2:
            __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmul.vv v4, v24, v28");
          case 1:
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmul.vv v0, v24, v28");
          }
          first = false;
        }
        else {
          switch (b) {
          case 4:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmacc.vv v12, v24, v28");
          case 3:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmacc.vv v8, v24, v28");
          case 2:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmacc.vv v4, v24, v28");
          case 1:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmacc.vv v0, v24, v28");
          }
        }
      } // end a unit stride
      else {
        // a non-unit stride
        if (first) {
          switch (b) {
          case 4:
            __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmul.vv v12, v24, v28");
          case 3:
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmul.vv v8, v24, v28");
          case 2:
            __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmul.vv v4, v24, v28");
          case 1:
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmul.vv v0, v24, v28");
          }
          first = false;
        }
        else {
          switch (b) {
          case 4:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmacc.vv v12, v24, v28");
          case 3:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmacc.vv v8, v24, v28");
          case 2:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmacc.vv v4, v24, v28");
          case 1:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmacc.vv v0, v24, v28");
          }
        }
      } // end a non-unit stride

      if (incz == FLT_SIZE) {
        __asm__(VLE "v24, (%0)" : : "r"(z_tmp));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSE "v24, (%0)" : : "r"(z_tmp));
      } else {
        __asm__(VLSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
      }

      __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
      __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
      __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
      avl -= vl;
    }

    __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
    __asm__("vmv.s.x v31, x0");

    switch (b) {
    case 4:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v12, v12, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.f) {
        __asm__("vfmul.vf v12, v12, ft10");
        __asm__(VSE "v12, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v12");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 3:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v8, v8, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.f) {
        __asm__("vfmul.vf v8, v8, ft10");
        __asm__(VSE "v8, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v8");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 2:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v4, v4, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.f) {
        __asm__("vfmul.vf v4, v4, ft10");
        __asm__(VSE "v4, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v4");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 1:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v0, v0, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.f) {
        __asm__("vfmul.vf v0, v0, ft10");
        __asm__(VSE "v0, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v0");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
    }
  } // end cleanup
  return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef FMUL
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define FMUL "fmul.d "
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "

void bli_ddotxaxpyf_sifive_x280_asm(
             conj_t           conjat,
             conj_t           conja,
             conj_t           conjw,
             conj_t           conjx,
             dim_t            m,
             dim_t            b,
       const void*   restrict alpha_,
       const void*   restrict a_, inc_t inca, inc_t lda,
       const void*   restrict w_, inc_t incw,
       const void*   restrict x_, inc_t incx,
       const void*   restrict beta_,
             void*   restrict y_, inc_t incy,
             void*   restrict z_, inc_t incz,
       const cntx_t* restrict cntx
                             ) {
  (void)conjat;
  (void)conja;
  (void)conjw;
  (void)conjx;
  (void)cntx;
  const double *restrict alpha = alpha_;
  const double *restrict beta = beta_;
  const double *restrict a = a_;
  const double *restrict w = w_;
  const double *restrict x = x_;
  double *restrict y = y_;
  double *restrict z = z_;

  if (b == 0)
    return;
  else if (m == 0 || *alpha == 0.) {
    // scale y by beta
    if (*beta == 0.)
        bli_dsetv_sifive_x280_asm(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
    else
        bli_dscalv_sifive_x280_intr(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
    return;
  }

  __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
  __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
  inca *= FLT_SIZE;
  lda *= FLT_SIZE;
  incw *= FLT_SIZE;
  incx *= FLT_SIZE;
  incy *= FLT_SIZE;
  incz *= FLT_SIZE;
  inc_t a_bump = 5 * lda;
  while (b >= 5) {
    // compute dot product of w with 5 rows of a
    const double* restrict w_tmp = w;
    const double* restrict z_tmp = z;
    const double* restrict a_col = a;
    size_t avl = m;
    bool first = true;
    while (avl) {
      const double* restrict a_row = a_col;
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
      if (incw == FLT_SIZE)
        __asm__(VLE "v28, (%0)" : : "r"(w_tmp));
      else
        __asm__(VLSE "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
      if (inca == FLT_SIZE) {
        // a unit stride
        if (first) {
          __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmul.vv v0, v24, v28");
          __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmul.vv v4, v24, v28");
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmul.vv v8, v24, v28");
          __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmul.vv v12, v24, v28");
          __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmul.vv v16, v24, v28");
          first = false;
        }
        else {
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmacc.vv v0, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmacc.vv v4, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmacc.vv v8, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmacc.vv v12, v24, v28");
          __asm__(VLE "v24, (%0)" : : "r"(a_row));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmacc.vv v16, v24, v28");
        }
      } // end a unit stride
      else {
        // a non-unit stride
        if (first) {
          __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmul.vv v0, v24, v28");
          __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmul.vv v4, v24, v28");
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmul.vv v8, v24, v28");
          __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmul.vv v12, v24, v28");
          __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : : "r"(x), "r"(incx));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmul.vv v16, v24, v28");
          first = false;
        }
        else {
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmul.vf v20, v24, ft0");
          __asm__("vfmacc.vv v0, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft1, v24");
          __asm__("vfmacc.vv v4, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft2, v24");
          __asm__("vfmacc.vv v8, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
          __asm__("vfmacc.vf v20, ft3, v24");
          __asm__("vfmacc.vv v12, v24, v28");
          __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
          __asm__("vfmacc.vf v20, ft4, v24");
          __asm__("vfmacc.vv v16, v24, v28");
        }
      } // end a non-unit stride

      if (incz == FLT_SIZE) {
        __asm__(VLE "v24, (%0)" : : "r"(z_tmp));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSE "v24, (%0)" : : "r"(z_tmp));
      } else {
        __asm__(VLSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
      }

      __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incx));
      __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
      __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
      avl -= vl;
    }

    __asm__("vmv.s.x v31, x0");

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v0, v0, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.) {
      __asm__("vfmul.vf v0, v0, ft10");
      __asm__(VSE "v0, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v0");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v4, v4, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.) {
      __asm__("vfmul.vf v4, v4, ft10");
      __asm__(VSE "v4, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v4");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v8, v8, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.) {
      __asm__("vfmul.vf v8, v8, ft10");
      __asm__(VSE "v8, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v8");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v12, v12, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.) {
      __asm__("vfmul.vf v12, v12, ft10");
      __asm__(VSE "v12, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v12");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
    __asm__("vfredusum.vs v16, v16, v31");
    __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
    if (*beta == 0.) {
      __asm__("vfmul.vf v16, v16, ft10");
      __asm__(VSE "v16, (%0)" : : "r"(y));
    }
    else {
      __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
      __asm__(FMUL "ft0, ft11, ft0");
      __asm__("vfmv.s.f v30, ft0");
      __asm__("vfmacc.vf v30, ft10, v16");
      __asm__(VSE "v30, (%0)" : : "r"(y));
    }
    __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

    __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
    b -= 5;
  }

  if (b > 0) {
    const double* restrict w_tmp = w;
    const double* restrict z_tmp = z;
    const double* restrict a_col;
    __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
    __asm__("add %0, %0, %1" : "+r"(x) : "r"((b - 1) * incx));
    size_t avl = m;
    bool first = true;
    while (avl) {
      const double* restrict a_row = a_col;
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
      if (incw == FLT_SIZE)
        __asm__(VLE "v28, (%0)" : : "r"(w_tmp));
      else
        __asm__(VLSE "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
      __asm__("vmv.v.i v20, 0");
      if (inca == FLT_SIZE) {
        // a unit stride
        if (first) {
          switch (b) {
          case 4:
            __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmul.vv v12, v24, v28");
          case 3:
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmul.vv v8, v24, v28");
          case 2:
            __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmul.vv v4, v24, v28");
          case 1:
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmul.vv v0, v24, v28");
          }
          first = false;
        }
        else {
          switch (b) {
          case 4:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmacc.vv v12, v24, v28");
          case 3:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmacc.vv v8, v24, v28");
          case 2:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmacc.vv v4, v24, v28");
          case 1:
            __asm__(VLE "v24, (%0)" : : "r"(a_row));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmacc.vv v0, v24, v28");
          }
        }
      } // end a unit stride
      else {
        // a non-unit stride
        if (first) {
          switch (b) {
          case 4:
            __asm__(FLT_LOAD "ft3, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmul.vv v12, v24, v28");
          case 3:
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmul.vv v8, v24, v28");
          case 2:
            __asm__(FLT_LOAD "ft1, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : : "r"(x), "r"(incx));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmul.vv v4, v24, v28");
          case 1:
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmul.vv v0, v24, v28");
          }
          first = false;
        }
        else {
          switch (b) {
          case 4:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft3, v24");
            __asm__("vfmacc.vv v12, v24, v28");
          case 3:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft2, v24");
            __asm__("vfmacc.vv v8, v24, v28");
          case 2:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
            __asm__("vfmacc.vf v20, ft1, v24");
            __asm__("vfmacc.vv v4, v24, v28");
          case 1:
            __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
            __asm__("vfmacc.vf v20, ft0, v24");
            __asm__("vfmacc.vv v0, v24, v28");
          }
        }
      } // end a non-unit stride

      if (incz == FLT_SIZE) {
        __asm__(VLE "v24, (%0)" : : "r"(z_tmp));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSE "v24, (%0)" : : "r"(z_tmp));
      } else {
        __asm__(VLSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
        __asm__("vfmacc.vf v24, ft10, v20");
        __asm__(VSSE "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
      }

      __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
      __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
      __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
      avl -= vl;
    }

    __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
    __asm__("vmv.s.x v31, x0");

    switch (b) {
    case 4:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v12, v12, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.) {
        __asm__("vfmul.vf v12, v12, ft10");
        __asm__(VSE "v12, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v12");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 3:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v8, v8, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.) {
        __asm__("vfmul.vf v8, v8, ft10");
        __asm__(VSE "v8, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v8");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 2:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v4, v4, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.) {
        __asm__("vfmul.vf v4, v4, ft10");
        __asm__(VSE "v4, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v4");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
      __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
    case 1:
      __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
      __asm__("vfredusum.vs v0, v0, v31");
      __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
      if (*beta == 0.) {
        __asm__("vfmul.vf v0, v0, ft10");
        __asm__(VSE "v0, (%0)" : : "r"(y));
      }
      else {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
        __asm__(FMUL "ft0, ft11, ft0");
        __asm__("vfmv.s.f v30, ft0");
        __asm__("vfmacc.vf v30, ft10, v0");
        __asm__(VSE "v30, (%0)" : : "r"(y));
      }
    }
  } // end cleanup
  return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef FMUL
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define FMUL "fmul.s "
#define FMADD "fmadd.s "
#define FNMSUB "fnmsub.s "
#define FNEG "fneg.s "
#define VLSEG2 "vlseg2e32.v "
#define VLSSEG2 "vlsseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "
#define VSE "vse32.v "

void bli_cdotxaxpyf_sifive_x280_asm
     (
             conj_t           conjat,
             conj_t           conja,
             conj_t           conjw,
             conj_t           conjx,
             dim_t            m,
             dim_t            b,
       const void*   restrict alpha_,
       const void*   restrict a_, inc_t inca, inc_t lda,
       const void*   restrict w_, inc_t incw,
       const void*   restrict x_, inc_t incx,
       const void*   restrict beta_,
             void*   restrict y_, inc_t incy,
             void*   restrict z_, inc_t incz,
       const cntx_t* restrict cntx
     )
{
    (void)cntx;
    const scomplex *restrict alpha = alpha_;
    const scomplex *restrict beta = beta_;
    const scomplex *restrict a = a_;
    const scomplex *restrict w = w_;
    const scomplex *restrict x = x_;
    scomplex *restrict y = y_;
    scomplex *restrict z = z_;
    
    if (b == 0)
        return;
    else if (m == 0 || (alpha->real == 0.f && alpha->imag == 0.f)) {
        // scale y by beta
        if (beta->real == 0.f && beta->imag == 0.f)
            bli_csetv_sifive_x280_asm(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        else
            bli_cscalv_sifive_x280_intr(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        return;
    }

    // use ft0-ft9 to store 5 entries of x, ft10-ft11 to store alpha,
    // and fa6-fa7 to store beta
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    __asm__(FLT_LOAD "fa6, (%0)" : : "r"(beta));
    __asm__(FLT_LOAD "fa7, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
    // Reduce to case when A^T is not conjugated, then conjugate
    // computed product A^T * w if needed.
    conj_t conjatw = BLIS_NO_CONJUGATE;
    if (conjat == BLIS_CONJUGATE) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjw);
        bli_toggle_conj(&conjatw);
    }
    conj_t conjax = BLIS_NO_CONJUGATE;
    if (conja == BLIS_CONJUGATE) {
        bli_toggle_conj(&conja);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjax);
    }
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incw *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    incz *= 2 * FLT_SIZE;
    // these are used to bump a and y, resp.
    inc_t a_bump = 5 * lda;
    inc_t y_bump = incy - FLT_SIZE;
    while (b >= 5) {
        // compute dot product of w with 6 rows of a
        const scomplex* restrict w_tmp = w;
        const scomplex* restrict z_tmp = z;
        const scomplex* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const scomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incw == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(w_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
            if (inca == 2 * FLT_SIZE) {
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjw = no conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a unit stride, conjw = conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjw = no conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a non-unit stride, conjw = conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a non-unit stride

            if (incz == 2 * FLT_SIZE) {
                __asm__(VLSEG2 "v24, (%0)" : : "r"(z_tmp));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSEG2 "v24, (%0)" : : "r"(z_tmp));
            }
            else {
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
            }

            __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
            __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("vmv.s.x v31, x0");

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v0, v0, v31");
        __asm__("vfredusum.vs v2, v2, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v0, v2, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v0, v2, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v0, v2);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v0, v2);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v4, v4, v31");
        __asm__("vfredusum.vs v6, v6, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v4, v6, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v4, v6, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v4, v6);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v4, v6);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v8, v8, v31");
        __asm__("vfredusum.vs v10, v10, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v8, v10, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v8, v10, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v8, v10);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v8, v10);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v12, v12, v31");
        __asm__("vfredusum.vs v14, v14, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v12, v14, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v12, v14, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v12, v14);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v12, v14);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v16, v16, v31");
        __asm__("vfredusum.vs v18, v18, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v16, v18, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v16, v18, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v16, v18);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v16, v18);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        // a += 5 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 5;
    }

    if (b > 0) {
        // cleanup loop, 0 < b < 5
        const scomplex* restrict w_tmp = w;
        const scomplex* restrict z_tmp = z;
        const scomplex* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        __asm__("add %0, %0, %1" : "+r"(x) : "r"((b - 1) * incx));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const scomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incw == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(w_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
            __asm__("vmv.v.i v20, 0");
            __asm__("vmv.v.i v22, 0");
            if (inca == 2 * FLT_SIZE) {
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjw = no conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a unit stride, conjw = conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjw = no conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a non-unit stride, conjw = conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a non-unit stride

            if (incz == 2 * FLT_SIZE) {
                __asm__(VLSEG2 "v24, (%0)" : : "r"(z_tmp));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSEG2 "v24, (%0)" : : "r"(z_tmp));
            }
            else {
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
            }

            __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
            __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        y_bump = incy + FLT_SIZE;
        __asm__("vmv.s.x v31, x0");

        switch (b) {
        case 4:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v12, v12, v31");
            __asm__("vfredusum.vs v14, v14, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v12, v14, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v12, v14, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v12, v14);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v12, v14);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 3:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v8, v8, v31");
            __asm__("vfredusum.vs v10, v10, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v8, v10, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v8, v10, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v8, v10);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v8, v10);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 2:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v4, v4, v31");
            __asm__("vfredusum.vs v6, v6, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v4, v6, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v4, v6, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v4, v6);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v4, v6);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 1:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v0, v0, v31");
            __asm__("vfredusum.vs v2, v2, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v0, v2, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v0, v2, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v0, v2);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v0, v2);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
        }
    }
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef FMUL
#undef FMADD
#undef FNMSUB
#undef FNEG
#undef VLSEG2
#undef VLSSEG2
#undef VSSEG2
#undef VSSSEG2
#undef VSE

#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define FMUL "fmul.d "
#define FMADD "fmadd.d "
#define FNMSUB "fnmsub.d "
#define FNEG "fneg.d "
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "
#define VSE "vse64.v "

void bli_zdotxaxpyf_sifive_x280_asm
     (
             conj_t           conjat,
             conj_t           conja,
             conj_t           conjw,
             conj_t           conjx,
             dim_t            m,
             dim_t            b,
       const void*   restrict alpha_,
       const void*   restrict a_, inc_t inca, inc_t lda,
       const void*   restrict w_, inc_t incw,
       const void*   restrict x_, inc_t incx,
       const void*   restrict beta_,
             void*   restrict y_, inc_t incy,
             void*   restrict z_, inc_t incz,
       const cntx_t* restrict cntx
     )
{
    (void)cntx;
    const dcomplex *restrict alpha = alpha_;
    const dcomplex *restrict beta = beta_;
    const dcomplex *restrict a = a_;
    const dcomplex *restrict w = w_;
    const dcomplex *restrict x = x_;
    dcomplex *restrict y = y_;
    dcomplex *restrict z = z_;

    if (b == 0)
        return;
    else if (m == 0 || (alpha->real == 0. && alpha->imag == 0.)) {
        // scale y by beta
        if (beta->real == 0. && beta->imag == 0.)
            bli_zsetv_sifive_x280_asm(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        else
            bli_zscalv_sifive_x280_intr(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        return;
    }

    // use ft0-ft9 to store 5 entries of x, ft10-ft11 to store alpha,
    // and fa6-fa7 to store beta
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    __asm__(FLT_LOAD "fa6, (%0)" : : "r"(beta));
    __asm__(FLT_LOAD "fa7, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
    // Reduce to case when A^T is not conjugated, then conjugate
    // computed product A^T * w if needed.
    conj_t conjatw = BLIS_NO_CONJUGATE;
    if (conjat == BLIS_CONJUGATE) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjw);
        bli_toggle_conj(&conjatw);
    }
    conj_t conjax = BLIS_NO_CONJUGATE;
    if (conja == BLIS_CONJUGATE) {
        bli_toggle_conj(&conja);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjax);
    }
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incw *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    incz *= 2 * FLT_SIZE;
    // these are used to bump a and y, resp.
    inc_t a_bump = 5 * lda;
    inc_t y_bump = incy - FLT_SIZE;
    while (b >= 5) {
        // compute dot product of w with 6 rows of a
        const dcomplex* restrict w_tmp = w;
        const dcomplex* restrict z_tmp = z;
        const dcomplex* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const dcomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incw == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(w_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
            if (inca == 2 * FLT_SIZE) {
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjw = no conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a unit stride, conjw = conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjw = no conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a non-unit stride, conjw = conj
                    if (first) {
                        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                        __asm__(FLT_LOAD "ft8, (%0)" : : "r"(x));
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft9, ft9"); }
                        __asm__("add %0, %0, %1" : "+r"(x) : "r"(incx));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vf(v20, v22, v24, v26, ft0, ft1);
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);

                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vf(v20, v22, ft8, ft9, v24, v26);
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a non-unit stride

            if (incz == 2 * FLT_SIZE) {
                __asm__(VLSEG2 "v24, (%0)" : : "r"(z_tmp));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSEG2 "v24, (%0)" : : "r"(z_tmp));
            }
            else {
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
            }

            __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
            __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("vmv.s.x v31, x0");

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v0, v0, v31");
        __asm__("vfredusum.vs v2, v2, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v0, v2, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v0, v2, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v0, v2);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v0, v2);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v4, v4, v31");
        __asm__("vfredusum.vs v6, v6, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v4, v6, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v4, v6, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v4, v6);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v4, v6);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v8, v8, v31");
        __asm__("vfredusum.vs v10, v10, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v8, v10, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v8, v10, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v8, v10);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v8, v10);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v12, v12, v31");
        __asm__("vfredusum.vs v14, v14, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v12, v14, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v12, v14, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v12, v14);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v12, v14);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v16, v16, v31");
        __asm__("vfredusum.vs v18, v18, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmul_vf(v28, v29, v16, v18, ft10, ft11);
          }
          else {
            vcmul_vf_conj(v28, v29, v16, v18, ft10, ft11);
          }
        }
        else {
          __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
          __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
          cmul(ft0, ft1, fa6, fa7, ft2, ft3);
          __asm__("vfmv.s.f v28, ft0");
          __asm__("vfmv.s.f v29, ft1");
          if (conjatw == BLIS_NO_CONJUGATE) {
            vcmacc_vf(v28, v29, ft10, ft11, v16, v18);
          }
          else {
            vcmacc_vf_conj(v28, v29, ft10, ft11, v16, v18);
          }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        // a += 5 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 5;
    }

    if (b > 0) {
        // cleanup loop, 0 < b < 5
        const dcomplex* restrict w_tmp = w;
        const dcomplex* restrict z_tmp = z;
        const dcomplex* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        __asm__("add %0, %0, %1" : "+r"(x) : "r"((b - 1) * incx));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const dcomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incw == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(w_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(w_tmp), "r"(incw));
            __asm__("vmv.v.i v20, 0");
            __asm__("vmv.v.i v22, 0");
            if (inca == 2 * FLT_SIZE) {
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjw = no conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a unit stride, conjw = conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjw == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjw = no conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_NO_CONJUGATE
                else { // conjw == BLIS_CONJUGATE
                    // a non-unit stride, conjw = conj
                    if (first) {
                        switch (b) {
                        case 4:
                            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft6, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft7, ft7"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft4, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft5, ft5"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft3, ft3"); }
                            __asm__("sub %0, %0, %1" : "+r"(x) : "r"(incx));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x), "I"(FLT_SIZE));
                            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x));
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            if (conjx == BLIS_CONJUGATE) { __asm__(FNEG "ft1, ft1"); }
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 4:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft6, ft7, v24, v26);
                            vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft4, ft5, v24, v26);
                            vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                            vcmacc_vf(v20, v22, ft2, ft3, v24, v26);
                            vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                            vcmacc_vf(v20, v22, ft0, ft1, v24, v26);
                            vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjw == BLIS_CONJUGATE
            } // end a non-unit stride

            if (incz == 2 * FLT_SIZE) {
                __asm__(VLSEG2 "v24, (%0)" : : "r"(z_tmp));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSEG2 "v24, (%0)" : : "r"(z_tmp));
            }
            else {
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
                if (conjax == BLIS_NO_CONJUGATE) {
                    vcmacc_vf(v24, v26, ft10, ft11, v20, v22);
                }
                else {
                    vcmacc_vf_conj(v24, v26, ft10, ft11, v20, v22);
                }
                __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(z_tmp), "r"(incz));
            }

            __asm__("add %0, %0, %1" : "+r"(w_tmp) : "r"(vl * incw));
            __asm__("add %0, %0, %1" : "+r"(z_tmp) : "r"(vl * incz));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        y_bump = incy + FLT_SIZE;
        __asm__("vmv.s.x v31, x0");

        switch (b) {
        case 4:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v12, v12, v31");
            __asm__("vfredusum.vs v14, v14, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v12, v14, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v12, v14, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v12, v14);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v12, v14);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 3:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v8, v8, v31");
            __asm__("vfredusum.vs v10, v10, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v8, v10, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v8, v10, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v8, v10);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v8, v10);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 2:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v4, v4, v31");
            __asm__("vfredusum.vs v6, v6, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v4, v6, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v4, v6, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v4, v6);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v4, v6);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 1:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v0, v0, v31");
            __asm__("vfredusum.vs v2, v2, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v0, v2, ft10, ft11);
              }
              else {
                vcmul_vf_conj(v28, v29, v0, v2, ft10, ft11);
              }
            }
            else {
              __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
              __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
              cmul(ft0, ft1, fa6, fa7, ft2, ft3);
              __asm__("vfmv.s.f v28, ft0");
              __asm__("vfmv.s.f v29, ft1");
              if (conjatw == BLIS_NO_CONJUGATE) {
                vcmacc_vf(v28, v29, ft10, ft11, v0, v2);
              }
              else {
                vcmacc_vf_conj(v28, v29, ft10, ft11, v0, v2);
              }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
        }
    }
    return;
}
