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

void bli_sdotxf_sifive_x280_asm(
              conj_t           conjat,
              conj_t           conjx,
              dim_t            m,
              dim_t            b,
        const void*   restrict alpha_,
        const void*   restrict a_, inc_t inca, inc_t lda,
        const void*   restrict x_, inc_t incx,
        const void*   restrict beta_,
              void*   restrict y_, inc_t incy,
        const cntx_t* restrict cntx
        ) {
    // think of a as b x m row major matrix (i.e. rsa = lda, csa = inca)
    // we process 6 elements of y per iteration, using y_tmp to load/store from
    // y a points to the 6 x m block of a needed this iteration each 6 x m block
    // is broken into 6 x vl blocks a_col points to the current 6 x vl block, we
    // use x_tmp to load from x a_row is used to load each of the 6 rows of this
    // 6 x vl block
    (void)conjat;
    (void)conjx;
    (void)cntx;
    const float* restrict alpha = alpha_;
    const float* restrict a = a_;
    const float* restrict x = x_;
    const float* restrict beta = beta_;
    float* restrict y = y_;

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
    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    inc_t a_bump = 6 * lda; // to bump a down 6 rows

    while (b >= 6) {
        // compute dot product of x with 6 rows of a
        const float* restrict x_tmp = x;
        const float* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const float* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == FLT_SIZE)
                __asm__(VLE "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSE "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == FLT_SIZE) {
                // a unit stride
                if (first) {
                    __asm__(VLE "v0, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v0, v0, v28");
                    __asm__(VLE "v4, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v4, v4, v28");
                    __asm__(VLE "v8, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v8, v8, v28");
                    __asm__(VLE "v12, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v12, v12, v28");
                    __asm__(VLE "v16, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v16, v16, v28");
                    __asm__(VLE "v20, (%0)" : : "r"(a_row));
                    __asm__("vfmul.vv v20, v20, v28");
                    first = false;
                }
                else {
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v0, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v4, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v8, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v12, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v16, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("vfmacc.vv v20, v24, v28");
                }
            } // end a unit stride
            else {
                // a non-unit stride
                if (first) {
                    __asm__(VLSE "v0, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v0, v0, v28");
                    __asm__(VLSE "v4, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v4, v4, v28");
                    __asm__(VLSE "v8, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v8, v8, v28");
                    __asm__(VLSE "v12, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v12, v12, v28");
                    __asm__(VLSE "v16, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v16, v16, v28");
                    __asm__(VLSE "v20, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("vfmul.vv v20, v20, v28");
                    first = false;
                }
                else {
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v0, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v4, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v8, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v12, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v16, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("vfmacc.vv v20, v24, v28");
                }
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
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

        __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v20, v20, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (*beta == 0.f) {
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__(VSE "v20, (%0)" : : "r"(y));
        }
        else {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
            __asm__(FMUL "ft0, ft11, ft0");
            __asm__("vfmv.s.f v30, ft0");
            __asm__("vfmacc.vf v30, ft10, v20");
            __asm__(VSE "v30, (%0)" : : "r"(y));
        }
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

        // a += 6 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 6;
    }

    if (b > 0) {
        // compute dot product of x with remaining < 6 rows of a
        const float* restrict x_tmp = x;
        // a_col will move along the last row of a!
        const float* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const float* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == FLT_SIZE)
                __asm__(VLE "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSE "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == FLT_SIZE) {
                // a unit stride
                if (first) {
                    switch (b) {
                    case 5:
                        __asm__(VLE "v16, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v16, v16, v28");
                    case 4:
                        __asm__(VLE "v12, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v12, v12, v28");
                    case 3:
                        __asm__(VLE "v8, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v8, v8, v28");
                    case 2:
                        __asm__(VLE "v4, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v4, v4, v28");
                    case 1:
                        __asm__(VLE "v0, (%0)" : : "r"(a_row));
                        __asm__("vfmul.vv v0, v0, v28");
                    }
                    first = false;
                }
                else {
                    switch (b) {
                    case 5:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v16, v24, v28");
                    case 4:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v12, v24, v28");
                    case 3:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v8, v24, v28");
                    case 2:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v4, v24, v28");
                    case 1:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("vfmacc.vv v0, v24, v28");
                    }
                }
            } // end a unit stride
            else {
                // a non-unit stride
                if (first) {
                    switch (b) {
                    case 5:
                        __asm__(VLSE "v16, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v16, v16, v28");
                    case 4:
                        __asm__(VLSE "v12, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v12, v12, v28");
                    case 3:
                        __asm__(VLSE "v8, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v8, v8, v28");
                    case 2:
                        __asm__(VLSE "v4, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v4, v4, v28");
                    case 1:
                        __asm__(VLSE "v0, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("vfmul.vv v0, v0, v28");
                    }
                    first = false;
                }
                else {
                    switch (b) {
                    case 5:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v16, v24, v28");
                    case 4:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v12, v24, v28");
                    case 3:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v8, v24, v28");
                    case 2:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v4, v24, v28");
                    case 1:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("vfmacc.vv v0, v24, v28");
                    }
                }
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        __asm__("vmv.s.x v31, x0");
        switch (b) {
        case 5:
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
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
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

void bli_ddotxf_sifive_x280_asm(
              conj_t           conjat,
              conj_t           conjx,
              dim_t            m,
              dim_t            b,
        const void*   restrict alpha_,
        const void*   restrict a_, inc_t inca, inc_t lda,
        const void*   restrict x_, inc_t incx,
        const void*   restrict beta_,
              void*   restrict y_, inc_t incy,
        const cntx_t* restrict cntx
        ) {
    // think of a as b x m row major matrix (i.e. rsa = lda, csa = inca)
    // we process 6 elements of y per iteration, using y_tmp to load/store from
    // y a points to the 6 x m block of a needed this iteration each 6 x m block
    // is broken into 6 x vl blocks a_col points to the current 6 x vl block, we
    // use x_tmp to load from x a_row is used to load each of the 6 rows of this
    // 6 x vl block
    (void)conjat;
    (void)conjx;
    (void)cntx;
    const double* restrict alpha = alpha_;
    const double* restrict a = a_;
    const double* restrict x = x_;
    const double* restrict beta = beta_;
    double* restrict y = y_;

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
    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    inc_t a_bump = 6 * lda; // to bump a down 6 rows

    while (b >= 6) {
        // compute dot product of x with 6 rows of a
        const double* restrict x_tmp = x;
        const double* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const double* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == FLT_SIZE)
                __asm__(VLE "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSE "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == FLT_SIZE) {
                // a unit stride
                if (first) {
                    __asm__(VLE "v0, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v0, v0, v28");
                    __asm__(VLE "v4, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v4, v4, v28");
                    __asm__(VLE "v8, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v8, v8, v28");
                    __asm__(VLE "v12, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v12, v12, v28");
                    __asm__(VLE "v16, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v16, v16, v28");
                    __asm__(VLE "v20, (%0)" : : "r"(a_row));
                    __asm__("vfmul.vv v20, v20, v28");
                    first = false;
                }
                else {
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v0, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v4, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v8, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v12, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v16, v24, v28");
                    __asm__(VLE "v24, (%0)" : : "r"(a_row));
                    __asm__("vfmacc.vv v20, v24, v28");
                }
            } // end a unit stride
            else {
                // a non-unit stride
                if (first) {
                    __asm__(VLSE "v0, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v0, v0, v28");
                    __asm__(VLSE "v4, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v4, v4, v28");
                    __asm__(VLSE "v8, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v8, v8, v28");
                    __asm__(VLSE "v12, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v12, v12, v28");
                    __asm__(VLSE "v16, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmul.vv v16, v16, v28");
                    __asm__(VLSE "v20, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("vfmul.vv v20, v20, v28");
                    first = false;
                }
                else {
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v0, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v4, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v8, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v12, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                    __asm__("vfmacc.vv v16, v24, v28");
                    __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                    __asm__("vfmacc.vv v20, v24, v28");
                }
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
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

        __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v20, v20, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (*beta == 0.) {
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__(VSE "v20, (%0)" : : "r"(y));
        }
        else {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(y));
            __asm__(FMUL "ft0, ft11, ft0");
            __asm__("vfmv.s.f v30, ft0");
            __asm__("vfmacc.vf v30, ft10, v20");
            __asm__(VSE "v30, (%0)" : : "r"(y));
        }
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(incy));

        // a += 6 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 6;
    }

    if (b > 0) {
        // compute dot product of x with remaining < 6 rows of a
        const double* restrict x_tmp = x;
        // a_col will move along the last row of a!
        const double* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const double* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m4, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == FLT_SIZE)
                __asm__(VLE "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSE "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == FLT_SIZE) {
                // a unit stride
                if (first) {
                    switch (b) {
                    case 5:
                        __asm__(VLE "v16, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v16, v16, v28");
                    case 4:
                        __asm__(VLE "v12, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v12, v12, v28");
                    case 3:
                        __asm__(VLE "v8, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v8, v8, v28");
                    case 2:
                        __asm__(VLE "v4, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v4, v4, v28");
                    case 1:
                        __asm__(VLE "v0, (%0)" : : "r"(a_row));
                        __asm__("vfmul.vv v0, v0, v28");
                    }
                    first = false;
                }
                else {
                    switch (b) {
                    case 5:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v16, v24, v28");
                    case 4:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v12, v24, v28");
                    case 3:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v8, v24, v28");
                    case 2:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v4, v24, v28");
                    case 1:
                        __asm__(VLE "v24, (%0)" : : "r"(a_row));
                        __asm__("vfmacc.vv v0, v24, v28");
                    }
                }
            } // end a unit stride
            else {
                // a non-unit stride
                if (first) {
                    switch (b) {
                    case 5:
                        __asm__(VLSE "v16, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v16, v16, v28");
                    case 4:
                        __asm__(VLSE "v12, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v12, v12, v28");
                    case 3:
                        __asm__(VLSE "v8, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v8, v8, v28");
                    case 2:
                        __asm__(VLSE "v4, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmul.vv v4, v4, v28");
                    case 1:
                        __asm__(VLSE "v0, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("vfmul.vv v0, v0, v28");
                    }
                    first = false;
                }
                else {
                    switch (b) {
                    case 5:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v16, v24, v28");
                    case 4:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v12, v24, v28");
                    case 3:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v8, v24, v28");
                    case 2:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        __asm__("vfmacc.vv v4, v24, v28");
                    case 1:
                        __asm__(VLSE "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("vfmacc.vv v0, v24, v28");
                    }
                }
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        __asm__("vmv.s.x v31, x0");
        switch (b) {
        case 5:
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
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(incy));
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
#define VLSEG2 "vlseg2e32.v "
#define VLSSEG2 "vlsseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "
#define VSE "vse32.v "

void bli_cdotxf_sifive_x280_asm(
              conj_t           conjat,
              conj_t           conjx,
              dim_t            m,
              dim_t            b,
        const void*   restrict alpha_,
        const void*   restrict a_, inc_t inca, inc_t lda,
        const void*   restrict x_, inc_t incx,
        const void*   restrict beta_,
              void*   restrict y_, inc_t incy,
        const cntx_t* restrict cntx
        ) {
    (void)cntx;
    const scomplex* restrict alpha = alpha_;
    const scomplex* restrict a = a_;
    const scomplex* restrict x = x_;
    const scomplex* restrict beta = beta_;
    scomplex* restrict y = y_;

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

    __asm__(FLT_LOAD "ft8, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(beta));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
    // Reduce to case when A^T is not conjugated, then conjugate
    // computed product A^T * x if needed.
    conj_t conjatx = BLIS_NO_CONJUGATE;
    if (conjat == BLIS_CONJUGATE) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjatx);
    }
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    // these are used to bump a and y, resp.
    inc_t a_bump = 6 * lda;
    inc_t y_bump = incy - FLT_SIZE;
    while (b >= 6) {
        // compute dot product of x with 6 rows of a
        const scomplex* restrict x_tmp = x;
        const scomplex* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const scomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == 2 * FLT_SIZE) {
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjx = no conj
                    if (first) {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmul_vv(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vv(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a unit stride, conjx = conj
                    if (first) {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmul_vv_conj(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vv_conj(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjx = no conj
                    if (first) {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmul_vv(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vv(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx = BLIS_CONJUGATE
                    // a non-unit stride, conjx = conj
                    if (first) {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmul_vv_conj(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vv_conj(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("vmv.s.x v31, x0");

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v0, v0, v31");
        __asm__("vfredusum.vs v2, v2, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v0, v2, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v0, v2, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v0, v2);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v0, v2);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v4, v6, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v4, v6, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v4, v6);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v4, v6);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v8, v10, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v8, v10, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v8, v10);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v8, v10);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v12, v14, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v12, v14, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v12, v14);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v12, v14);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v16, v18, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v16, v18, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v16, v18);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v16, v18);
            }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v20, v20, v31");
        __asm__("vfredusum.vs v22, v22, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0.f && beta->imag == 0.f) {
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v20, v22, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v20, v22, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v20, v22);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v20, v22);
            }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        // a += 6 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 6;
    }

    if (b > 0) {
        // cleanup loop, 0 < b < 6
        const scomplex* restrict x_tmp = x;
        const scomplex* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const scomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == 2 * FLT_SIZE) {
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjx = no conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a unit stride, conjx = conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjx = no conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a non-unit stride, conjx = conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        y_bump = incy + FLT_SIZE;
        __asm__("vmv.s.x v31, x0");
        
        switch (b) {
        case 5:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v16, v16, v31");
            __asm__("vfredusum.vs v18, v18, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v16, v18, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v16, v18, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v16, v18);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v16, v18);
                }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 4:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v12, v12, v31");
            __asm__("vfredusum.vs v14, v14, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0.f && beta->imag == 0.f) {
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v12, v14, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v12, v14, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v12, v14);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v12, v14);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v8, v10, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v8, v10, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v8, v10);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v8, v10);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v4, v6, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v4, v6, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v4, v6);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v4, v6);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v0, v2, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v0, v2, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v0, v2);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v0, v2);
                }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
        }
    } // end cleanup
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef FMUL
#undef FMADD
#undef FNMSUB
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
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "
#define VSE "vse64.v "

void bli_zdotxf_sifive_x280_asm(
              conj_t           conjat,
              conj_t           conjx,
              dim_t            m,
              dim_t            b,
        const void*   restrict alpha_,
        const void*   restrict a_, inc_t inca, inc_t lda,
        const void*   restrict x_, inc_t incx,
        const void*   restrict beta_,
              void*   restrict y_, inc_t incy,
        const cntx_t* restrict cntx
        ) {
    (void)cntx;
    const dcomplex* restrict alpha = alpha_;
    const dcomplex* restrict a = a_;
    const dcomplex* restrict x = x_;
    const dcomplex* restrict beta = beta_;
    dcomplex* restrict y = y_;

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

    __asm__(FLT_LOAD "ft8, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(beta));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
    // Reduce to case when A^T is not conjugated, then conjugate
    // computed product A^T * x if needed.
    conj_t conjatx = BLIS_NO_CONJUGATE;
    if (conjat == BLIS_CONJUGATE) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjatx);
    }
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    // these are used to bump a and y, resp.
    inc_t a_bump = 6 * lda;
    inc_t y_bump = incy - FLT_SIZE;
    while (b >= 6) {
        // compute dot product of x with 6 rows of a
        const dcomplex* restrict x_tmp = x;
        const dcomplex* restrict a_col = a;
        size_t avl = m;
        bool first = true;
        while (avl) {
            const dcomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == 2 * FLT_SIZE) {
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjx = no conj
                    if (first) {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmul_vv(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vv(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a unit stride, conjx = conj
                    if (first) {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmul_vv_conj(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                        vcmacc_vv_conj(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjx = no conj
                    if (first) {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmul_vv(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vv(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx = BLIS_CONJUGATE
                    // a non-unit stride, conjx = conj
                    if (first) {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmul_vv_conj(v20, v22, v24, v26, v28, v30);
                        first = false;
                    }
                    else {
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        __asm__("add %0, %0, %1" : "+r"(a_row) : "r"(lda));
                        vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                        vcmacc_vv_conj(v20, v22, v24, v26, v28, v30);
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("vmv.s.x v31, x0");

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v0, v0, v31");
        __asm__("vfredusum.vs v2, v2, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v0, v2, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v0, v2, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v0, v2);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v0, v2);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v4, v6, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v4, v6, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v4, v6);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v4, v6);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v8, v10, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v8, v10, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v8, v10);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v8, v10);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v12, v14, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v12, v14, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v12, v14);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v12, v14);
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
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v16, v18, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v16, v18, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v16, v18);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v16, v18);
            }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
        __asm__("vfredusum.vs v20, v20, v31");
        __asm__("vfredusum.vs v22, v22, v31");
        __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
        if (beta->real == 0. && beta->imag == 0.) {
            if (conjatx == BLIS_NO_CONJUGATE) {
                vcmul_vf(v28, v29, v20, v22, ft8, ft9);
            }
            else {
                vcmul_vf_conj(v28, v29, v20, v22, ft8, ft9);
            }
        }
        else {
            __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
            cmul(ft0, ft1, ft10, ft11, ft2, ft3);
            __asm__("vfmv.s.f v28, ft0");
            __asm__("vfmv.s.f v29, ft1");
            if (conjatx == BLIS_NO_CONJUGATE) {
              vcmacc_vf(v28, v29, ft8, ft9, v20, v22);
            }
            else {
              vcmacc_vf_conj(v28, v29, ft8, ft9, v20, v22);
            }
        }
        __asm__(VSE "v28, (%0)" : : "r"(y));
        __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
        __asm__(VSE "v29, (%0)" : : "r"(y));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(y_bump));

        // a += 6 * lda;
        __asm__("add %0, %0, %1" : "+r"(a) : "r"(a_bump));
        b -= 6;
    }

    if (b > 0) {
        // cleanup loop, 0 < b < 6
        const dcomplex* restrict x_tmp = x;
        const dcomplex* restrict a_col;
        __asm__("add %0, %1, %2" : "=r"(a_col) : "r"(a), "r"((b - 1) * lda));
        size_t avl = m;
        bool first = true;
        while (avl) {
            const dcomplex* restrict a_row = a_col;
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m2, tu, ma" : "=r"(vl) : "r"(avl), "i"(8 * FLT_SIZE));
            if (incx == 2 * FLT_SIZE)
                __asm__(VLSEG2 "v28, (%0)" : : "r"(x_tmp));
            else
                __asm__(VLSSEG2 "v28, (%0), %1" : : "r"(x_tmp), "r"(incx));
            if (inca == 2 * FLT_SIZE) {
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a unit stride, conjx = no conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a unit stride, conjx = conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSEG2 "v24, (%0)" : : "r"(a_row));
                          vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a unit stride
            else { // a non-unit stride
                if (conjx == BLIS_NO_CONJUGATE) {
                    // a non-unit stride, conjx = no conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmul_vv(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmacc_vv(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_NO_CONJUGATE
                else { // conjx == BLIS_CONJUGATE
                    // a non-unit stride, conjx = conj
                    if (first) {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmul_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmul_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                        first = false;
                    }
                    else {
                        switch (b) {
                        case 5:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v16, v18, v24, v26, v28, v30);
                        case 4:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v12, v14, v24, v26, v28, v30);
                        case 3:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v8, v10, v24, v26, v28, v30);
                        case 2:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          __asm__("sub %0, %0, %1" : "+r"(a_row) : "r"(lda));
                          vcmacc_vv_conj(v4, v6, v24, v26, v28, v30);
                        case 1:
                          __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(a_row), "r"(inca));
                          vcmacc_vv_conj(v0, v2, v24, v26, v28, v30);
                        }
                    }
                } // end conjx == BLIS_CONJUGATE
            } // end a non-unit stride
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(vl * incx));
            __asm__("add %0, %0, %1" : "+r"(a_col) : "r"(vl * inca));
            avl -= vl;
        }

        __asm__("add %0, %0, %1" : "+r"(y) : "r"((b - 1) * incy));
        y_bump = incy + FLT_SIZE;
        __asm__("vmv.s.x v31, x0");
        
        switch (b) {
        case 5:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v16, v16, v31");
            __asm__("vfredusum.vs v18, v18, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v16, v18, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v16, v18, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v16, v18);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v16, v18);
                }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
            __asm__("sub %0, %0, %1" : "+r"(y) : "r"(y_bump));
        case 4:
            __asm__("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(m), "i"(8 * FLT_SIZE));
            __asm__("vfredusum.vs v12, v12, v31");
            __asm__("vfredusum.vs v14, v14, v31");
            __asm__("vsetivli zero, 1, e%0, m1, ta, ma" : : "i"(8 * FLT_SIZE));
            if (beta->real == 0. && beta->imag == 0.) {
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v12, v14, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v12, v14, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v12, v14);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v12, v14);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v8, v10, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v8, v10, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v8, v10);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v8, v10);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v4, v6, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v4, v6, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v4, v6);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v4, v6);
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
                if (conjatx == BLIS_NO_CONJUGATE) {
                    vcmul_vf(v28, v29, v0, v2, ft8, ft9);
                }
                else {
                    vcmul_vf_conj(v28, v29, v0, v2, ft8, ft9);
                }
            }
            else {
                __asm__(FLT_LOAD "ft2, (%0)" : : "r"(y));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(y), "I"(FLT_SIZE));
                cmul(ft0, ft1, ft10, ft11, ft2, ft3);
                __asm__("vfmv.s.f v28, ft0");
                __asm__("vfmv.s.f v29, ft1");
                if (conjatx == BLIS_NO_CONJUGATE) {
                  vcmacc_vf(v28, v29, ft8, ft9, v0, v2);
                }
                else {
                  vcmacc_vf_conj(v28, v29, ft8, ft9, v0, v2);
                }
            }
            __asm__(VSE "v28, (%0)" : : "r"(y));
            __asm__("addi %0, %0, %1" : "+r"(y) : "I"(FLT_SIZE));
            __asm__(VSE "v29, (%0)" : : "r"(y));
        }
    } // end cleanup
    return;
}
