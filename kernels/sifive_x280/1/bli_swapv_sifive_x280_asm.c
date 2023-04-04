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
#define VLE "vle32.v "
#define VLSE "vlse32.v "
#define VSE "vse32.v "
#define VSSE "vsse32.v "

void bli_sswapv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx, void * restrict y_,
                     inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    float* restrict x = x_;
    float* restrict y = y_;
    if (n <= 0)
        return;

    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == FLT_SIZE)
            __asm__(VLE "v0, (%0)" : : "r"(x));
        else
            __asm__(VLSE "v0, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == FLT_SIZE)
            __asm__(VLE "v8, (%0)" : : "r"(y));
        else
            __asm__(VLSE "v8, (%0), %1" : : "r"(y), "r"(incy));

        if (incx == FLT_SIZE)
            __asm__(VSE "v8, (%0)" : : "r"(x));
        else
            __asm__(VSSE "v8, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == FLT_SIZE)
            __asm__(VSE "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 8
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "

void bli_dswapv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    double* restrict x = x_;
    double* restrict y = y_;
    if (n <= 0)
        return;

    incx *= FLT_SIZE;
    incy *= FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == FLT_SIZE)
            __asm__(VLE "v0, (%0)" : : "r"(x));
        else
            __asm__(VLSE "v0, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == FLT_SIZE)
            __asm__(VLE "v8, (%0)" : : "r"(y));
        else
            __asm__(VLSE "v8, (%0), %1" : : "r"(y), "r"(incy));

        if (incx == FLT_SIZE)
            __asm__(VSE "v8, (%0)" : : "r"(x));
        else
            __asm__(VSSE "v8, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == FLT_SIZE)
            __asm__(VSE "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 4
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "

void bli_cswapv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    scomplex* restrict x = x_;
    scomplex* restrict y = y_;
    if (n <= 0)
        return;

    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * 2 * FLT_SIZE));
        if (incx == 2 * FLT_SIZE)
            __asm__(VLE "v0, (%0)" : : "r"(x));
        else
            __asm__(VLSE "v0, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == 2 * FLT_SIZE)
            __asm__(VLE "v8, (%0)" : : "r"(y));
        else
            __asm__(VLSE "v8, (%0), %1" : : "r"(y), "r"(incy));

        if (incx == 2 * FLT_SIZE)
            __asm__(VSE "v8, (%0)" : : "r"(x));
        else
            __asm__(VSSE "v8, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == 2 * FLT_SIZE)
            __asm__(VSE "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE

#define FLT_SIZE 8
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "

void bli_zswapv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    dcomplex* restrict x = x_;
    dcomplex* restrict y = y_;
    if (n <= 0)
        return;

    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == 2 * FLT_SIZE)
            __asm__(VLSEG2 "v0, (%0)" : : "r"(x));
        else
            __asm__(VLSSEG2 "v0, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == 2 * FLT_SIZE)
            __asm__(VLSEG2 "v8, (%0)" : : "r"(y));
        else
            __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(y), "r"(incy));

        if (incx == 2 * FLT_SIZE)
            __asm__(VSSEG2 "v8, (%0)" : : "r"(x));
        else
            __asm__(VSSSEG2 "v8, (%0), %1" : : "r"(x), "r"(incx));
        if (incy == 2 * FLT_SIZE)
            __asm__(VSSEG2 "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(y), "r"(incy));

        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(vl * incy));
        avl -= vl;
    }
    return;
}
