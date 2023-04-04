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

void bli_sinvertv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                           const cntx_t *cntx) {
    (void)cntx;
    float* restrict x = x_;
    if (n <= 0)
        return;

    float one = 1.f;
    __asm__(FLT_LOAD "f0, (%0)" : : "r"(&one));
    incx *= FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == FLT_SIZE) {
            __asm__(VLE "v0, (%0)" : : "r"(x));
            __asm__("vfrdiv.vf v0, v0, f0");
            __asm__(VSE "v0, (%0)" : : "r"(x));
        } else {
            __asm__(VLSE "v0, (%0), %1" : : "r"(x), "r"(incx));
            __asm__("vfrdiv.vf v0, v0, f0");
            __asm__(VSSE "v0, (%0), %1" : : "r"(x), "r"(incx));
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
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

void bli_dinvertv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                           const cntx_t *cntx) {
    (void)cntx;
    double* restrict x = x_;
    if (n <= 0)
        return;

    double one = 1.;
    __asm__(FLT_LOAD "f0, (%0)" : : "r"(&one));
    incx *= FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == FLT_SIZE) {
            __asm__(VLE "v0, (%0)" : : "r"(x));
            __asm__("vfrdiv.vf v0, v0, f0");
            __asm__(VSE "v0, (%0)" : : "r"(x));
        } else {
            __asm__(VLSE "v0, (%0), %1" : : "r"(x), "r"(incx));
            __asm__("vfrdiv.vf v0, v0, f0");
            __asm__(VSSE "v0, (%0), %1" : : "r"(x), "r"(incx));
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
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
#define VLSEG2 "vlseg2e32.v "
#define VLSSEG2 "vlsseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "

void bli_cinvertv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                           const cntx_t *cntx) {
    (void)cntx;
    scomplex* restrict x = x_;
    if (n <= 0)
        return;

    incx *= 2 * FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == 2 * FLT_SIZE) {
            __asm__(VLSEG2 "v0, (%0)" : : "r"(x));
            __asm__("vfneg.v v4, v4");
            __asm__("vfmul.vv v8, v0, v0");
            __asm__("vfmacc.vv v8, v4, v4");
            __asm__("vfdiv.vv v0, v0, v8");
            __asm__("vfdiv.vv v4, v4, v8");
            __asm__(VSSEG2 "v0, (%0)" : : "r"(x));
        } else {
            __asm__(VLSSEG2 "v0, (%0), %1" : : "r"(x), "r"(incx));
            __asm__("vfneg.v v4, v4");
            __asm__("vfmul.vv v8, v0, v0");
            __asm__("vfmacc.vv v8, v4, v4");
            __asm__("vfdiv.vv v0, v0, v8");
            __asm__("vfdiv.vv v4, v4, v8");
            __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(x), "r"(incx));
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        avl -= vl;
    }
    return;
}

#undef FLT_SIZE
#undef VLSEG2
#undef VLSSEG2
#undef VSSEG2
#undef VSSSEG2

#define FLT_SIZE 8
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "

void bli_zinvertv_sifive_x280_asm(dim_t n, void * restrict x_, inc_t incx,
                           const cntx_t *cntx) {
    (void)cntx;
    dcomplex* restrict x = x_;
    if (n <= 0)
        return;

    incx *= 2 * FLT_SIZE;
    size_t avl = n;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma"
                         : "=r"(vl)
                         : "r"(avl), "i"(8 * FLT_SIZE));
        if (incx == 2 * FLT_SIZE) {
            __asm__(VLSEG2 "v0, (%0)" : : "r"(x));
            __asm__("vfneg.v v4, v4");
            __asm__("vfmul.vv v8, v0, v0");
            __asm__("vfmacc.vv v8, v4, v4");
            __asm__("vfdiv.vv v0, v0, v8");
            __asm__("vfdiv.vv v4, v4, v8");
            __asm__(VSSEG2 "v0, (%0)" : : "r"(x));
        } else {
            __asm__(VLSSEG2 "v0, (%0), %1" : : "r"(x), "r"(incx));
            __asm__("vfneg.v v4, v4");
            __asm__("vfmul.vv v8, v0, v0");
            __asm__("vfmacc.vv v8, v4, v4");
            __asm__("vfdiv.vv v0, v0, v8");
            __asm__("vfdiv.vv v4, v4, v8");
            __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(x), "r"(incx));
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        avl -= vl;
    }
    return;
}
