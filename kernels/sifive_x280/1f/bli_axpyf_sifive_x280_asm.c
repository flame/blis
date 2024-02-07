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
#include <math.h>
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>

#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define VLE "vle32.v "
#define VLSE "vlse32.v "
#define VSE "vse32.v "
#define VSSE "vsse32.v "

void bli_saxpyf_sifive_x280_asm(conj_t conja, conj_t conjx, dim_t m, dim_t b,
                         const void *restrict alpha_, const void *restrict a_, inc_t inca,
                         inc_t lda, const void *restrict x_, inc_t incx,
                         void *restrict y_, inc_t incy, const cntx_t *restrict cntx) {
    (void)conja;
    (void)conjx;
    (void)cntx;
    const float *restrict alpha = alpha_;
    const float *restrict a = a_;
    const float *restrict x = x_;
    float *restrict y = y_;

    if (m == 0 || b == 0)
        return;
    __asm__(FLT_LOAD "ft11, (%0)" : : "r"(alpha));
    inca *= FLT_SIZE;
    lda *= FLT_SIZE;
    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    size_t avl = m;
    while (avl) {
        // process vl elements of y at a time
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        // x_tmp traverses x
        // a points to the vl x b block of a needed this iteration
        // a_tmp traverses the columns of this block
        const float* restrict x_tmp = x;
        const float* restrict a_tmp = a;
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
        if (inca == FLT_SIZE)
            __asm__(VLE "v0, (%0)" : : "r"(a_tmp));
        else
            __asm__(VLSE "v0, (%0), %1" : : "r"(a_tmp), "r"(inca));
        __asm__("vfmul.vf v0, v0, ft0");
        __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
        __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));

        for (dim_t i = 1; i < b; ++i) {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
            if (inca == FLT_SIZE)
                __asm__(VLE "v24, (%0)" : : "r"(a_tmp));
            else
                __asm__(VLSE "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
            __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
            __asm__("vfmacc.vf v0, ft0, v24");
        }

        if (incy == FLT_SIZE) {
            __asm__(VLE "v24, (%0)" : : "r"(y));
            __asm__("vfmacc.vf v24, ft11, v0");
            __asm__(VSE "v24, (%0)" : : "r"(y));
        } else {
            __asm__(VLSE "v24, (%0), %1" : : "r"(y), "r"(incy));
            __asm__("vfmacc.vf v24, ft11, v0");
            __asm__(VSSE "v24, (%0), %1" : : "r"(y), "r"(incy));
        }

        __asm__("add %0, %0, %1" : "+r"(a) : "r"(vl * inca));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "

void bli_daxpyf_sifive_x280_asm(conj_t conja, conj_t conjx, dim_t m, dim_t b,
                         const void *restrict alpha_, const void *restrict a_, inc_t inca,
                         inc_t lda, const void *restrict x_, inc_t incx,
                         void *restrict y_, inc_t incy, const cntx_t *restrict cntx) {
    (void)conja;
    (void)conjx;
    (void)cntx;
    const double *restrict alpha = alpha_;
    const double *restrict a = a_;
    const double *restrict x = x_;
    double *restrict y = y_;

    if (m == 0 || b == 0)
        return;
    __asm__(FLT_LOAD "ft11, (%0)" : : "r"(alpha));
    inca *= FLT_SIZE;
    lda *= FLT_SIZE;
    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    size_t avl = m;
    while (avl) {
        // process vl elements of y at a time
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        // x_tmp traverses x
        // a points to the vl x b block of a needed this iteration
        // a_tmp traverses the columns of this block
        const double* restrict x_tmp = x;
        const double* restrict a_tmp = a;
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
        if (inca == FLT_SIZE)
            __asm__(VLE "v0, (%0)" : : "r"(a_tmp));
        else
            __asm__(VLSE "v0, (%0), %1" : : "r"(a_tmp), "r"(inca));
        __asm__("vfmul.vf v0, v0, ft0");
        __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
        __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));

        for (dim_t i = 1; i < b; ++i) {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
            if (inca == FLT_SIZE)
                __asm__(VLE "v24, (%0)" : : "r"(a_tmp));
            else
                __asm__(VLSE "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
            __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
            __asm__("vfmacc.vf v0, ft0, v24");
        }

        if (incy == FLT_SIZE) {
            __asm__(VLE "v24, (%0)" : : "r"(y));
            __asm__("vfmacc.vf v24, ft11, v0");
            __asm__(VSE "v24, (%0)" : : "r"(y));
        } else {
            __asm__(VLSE "v24, (%0), %1" : : "r"(y), "r"(incy));
            __asm__("vfmacc.vf v24, ft11, v0");
            __asm__(VSSE "v24, (%0), %1" : : "r"(y), "r"(incy));
        }

        __asm__("add %0, %0, %1" : "+r"(a) : "r"(vl * inca));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define VLSEG "vlseg2e32.v "
#define VLSSEG "vlsseg2e32.v "
#define VSSEG "vsseg2e32.v "
#define VSSSEG "vssseg2e32.v "

void bli_caxpyf_sifive_x280_asm(conj_t conja, conj_t conjx, dim_t m, dim_t b,
                         const void *restrict alpha_, const void *restrict a_,
                         inc_t inca, inc_t lda, const void *restrict x_,
                         inc_t incx, void *restrict y_, inc_t incy,
                         const cntx_t *restrict cntx) {
    (void)cntx;
    const scomplex *restrict alpha = alpha_;
    const scomplex *restrict a = a_;
    const scomplex *restrict x = x_;
    scomplex *restrict y = y_;
    
    if (m == 0 || b == 0)
        return;
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    size_t avl = m;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        const scomplex* restrict x_tmp = x;
        const scomplex* restrict a_tmp = a;
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x_tmp), "I"(FLT_SIZE));
        if (inca == 2 * FLT_SIZE)
            __asm__(VLSEG "v24, (%0)" : : "r"(a_tmp));
        else
            __asm__(VLSSEG "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
        __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
        __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
        __asm__("vfmul.vf v0, v24, ft0");
        __asm__("vfmul.vf v4, v24, ft1");
        if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
            __asm__("vfnmsac.vf v0, ft1, v28");
            __asm__("vfmacc.vf v4, ft0, v28");
        } else if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_CONJUGATE) {
            __asm__("vfmacc.vf v0, ft1, v28");
            __asm__("vfmsac.vf v4, ft0, v28");
        } else if (conja == BLIS_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
            __asm__("vfmacc.vf v0, ft1, v28");
            __asm__("vfnmsac.vf v4, ft0, v28");
        } else {
            __asm__("vfnmsac.vf v0, ft1, v28");
            __asm__("vfnmacc.vf v4, ft0, v28");
        }

        for (dim_t i = 1; i < b; ++i) {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x_tmp), "I"(FLT_SIZE));
            if (inca == 2 * FLT_SIZE)
                __asm__(VLSEG "v24, (%0)" : : "r"(a_tmp));
            else
                __asm__(VLSSEG "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
            __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
            __asm__("vfmacc.vf v0, ft0, v24");
            if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
                __asm__("vfmacc.vf v4, ft1, v24");
                __asm__("vfnmsac.vf v0, ft1, v28");
                __asm__("vfmacc.vf v4, ft0, v28");
            } else if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_CONJUGATE) {
                __asm__("vfnmsac.vf v4, ft1, v24");
                __asm__("vfmacc.vf v0, ft1, v28");
                __asm__("vfmacc.vf v4, ft0, v28");
            } else if (conja == BLIS_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
                __asm__("vfmacc.vf v4, ft1, v24");
                __asm__("vfmacc.vf v0, ft1, v28");
                __asm__("vfnmsac.vf v4, ft0, v28");
            } else { // conja == BLIS_CONJUGATE && conjx == BLIS_CONJUGATE
                __asm__("vfnmsac.vf v4, ft1, v24");
                __asm__("vfnmsac.vf v0, ft1, v28");
                __asm__("vfnmsac.vf v4, ft0, v28");
            }
        }

        if (incy == 2 * FLT_SIZE) {
            __asm__(VLSEG "v24, (%0)" : : "r"(y));
            __asm__("vfmacc.vf v24, ft10, v0");
            __asm__("vfmacc.vf v28, ft10, v4");
            __asm__("vfnmsac.vf v24, ft11, v4");
            __asm__("vfmacc.vf v28, ft11, v0");
            __asm__(VSSEG "v24, (%0)" : : "r"(y));
        } else {
            __asm__(VLSSEG "v24, (%0), %1" : : "r"(y), "r"(incy));
            __asm__("vfmacc.vf v24, ft10, v0");
            __asm__("vfmacc.vf v28, ft10, v4");
            __asm__("vfnmsac.vf v24, ft11, v4");
            __asm__("vfmacc.vf v28, ft11, v0");
            __asm__(VSSSEG "v24, (%0), %1" : : "r"(y), "r"(incy));
        }

        __asm__("add %0, %0, %1" : "+r"(a) : "r"(vl * inca));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VLSEG
#undef VLSSEG
#undef VSSEG
#undef VSSSEG

#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define VLSEG "vlseg2e64.v "
#define VLSSEG "vlsseg2e64.v "
#define VSSEG "vsseg2e64.v "
#define VSSSEG "vssseg2e64.v "

void bli_zaxpyf_sifive_x280_asm(conj_t conja, conj_t conjx, dim_t m, dim_t b,
                         const void *restrict alpha_, const void *restrict a_,
                         inc_t inca, inc_t lda, const void *restrict x_,
                         inc_t incx, void *restrict y_, inc_t incy,
                         const cntx_t *restrict cntx) {
    (void)cntx;
    const dcomplex *restrict alpha = alpha_;
    const dcomplex *restrict a = a_;
    const dcomplex *restrict x = x_;
    dcomplex *restrict y = y_;

    if (m == 0 || b == 0)
        return;
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));
    inca *= 2 * FLT_SIZE;
    lda *= 2 * FLT_SIZE;
    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    size_t avl = m;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        const dcomplex* restrict x_tmp = x;
        const dcomplex* restrict a_tmp = a;
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x_tmp), "I"(FLT_SIZE));
        if (inca == 2 * FLT_SIZE)
            __asm__(VLSEG "v24, (%0)" : : "r"(a_tmp));
        else
            __asm__(VLSSEG "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
        __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
        __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
        __asm__("vfmul.vf v0, v24, ft0");
        __asm__("vfmul.vf v4, v24, ft1");
        if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
            __asm__("vfnmsac.vf v0, ft1, v28");
            __asm__("vfmacc.vf v4, ft0, v28");
        } else if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_CONJUGATE) {
            __asm__("vfmacc.vf v0, ft1, v28");
            __asm__("vfmsac.vf v4, ft0, v28");
        } else if (conja == BLIS_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
            __asm__("vfmacc.vf v0, ft1, v28");
            __asm__("vfnmsac.vf v4, ft0, v28");
        } else {
            __asm__("vfnmsac.vf v0, ft1, v28");
            __asm__("vfnmacc.vf v4, ft0, v28");
        }

        for (dim_t i = 1; i < b; ++i) {
            __asm__(FLT_LOAD "ft0, (%0)" : : "r"(x_tmp));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(x_tmp), "I"(FLT_SIZE));
            if (inca == 2 * FLT_SIZE)
                __asm__(VLSEG "v24, (%0)" : : "r"(a_tmp));
            else
                __asm__(VLSSEG "v24, (%0), %1" : : "r"(a_tmp), "r"(inca));
            __asm__("add %0, %0, %1" : "+r"(x_tmp) : "r"(incx));
            __asm__("add %0, %0, %1" : "+r"(a_tmp) : "r"(lda));
            __asm__("vfmacc.vf v0, ft0, v24");
            if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
                __asm__("vfmacc.vf v4, ft1, v24");
                __asm__("vfnmsac.vf v0, ft1, v28");
                __asm__("vfmacc.vf v4, ft0, v28");
            } else if (conja == BLIS_NO_CONJUGATE && conjx == BLIS_CONJUGATE) {
                __asm__("vfnmsac.vf v4, ft1, v24");
                __asm__("vfmacc.vf v0, ft1, v28");
                __asm__("vfmacc.vf v4, ft0, v28");
            } else if (conja == BLIS_CONJUGATE && conjx == BLIS_NO_CONJUGATE) {
                __asm__("vfmacc.vf v4, ft1, v24");
                __asm__("vfmacc.vf v0, ft1, v28");
                __asm__("vfnmsac.vf v4, ft0, v28");
            } else { // conja == BLIS_CONJUGATE && conjx == BLIS_CONJUGATE
                __asm__("vfnmsac.vf v4, ft1, v24");
                __asm__("vfnmsac.vf v0, ft1, v28");
                __asm__("vfnmsac.vf v4, ft0, v28");
            }
        }

        if (incy == 2 * FLT_SIZE) {
            __asm__(VLSEG "v24, (%0)" : : "r"(y));
            __asm__("vfmacc.vf v24, ft10, v0");
            __asm__("vfmacc.vf v28, ft10, v4");
            __asm__("vfnmsac.vf v24, ft11, v4");
            __asm__("vfmacc.vf v28, ft11, v0");
            __asm__(VSSEG "v24, (%0)" : : "r"(y));
        } else {
            __asm__(VLSSEG "v24, (%0), %1" : : "r"(y), "r"(incy));
            __asm__("vfmacc.vf v24, ft10, v0");
            __asm__("vfmacc.vf v28, ft10, v4");
            __asm__("vfnmsac.vf v24, ft11, v4");
            __asm__("vfmacc.vf v28, ft11, v0");
            __asm__(VSSSEG "v24, (%0), %1" : : "r"(y), "r"(incy));
        }

        __asm__("add %0, %0, %1" : "+r"(a) : "r"(vl * inca));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}
