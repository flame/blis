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

void bli_scopyv_sifive_x280_asm(conj_t conjx, dim_t n, const void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)conjx;
    (void)cntx;
    const float* restrict x = x_;
    float* restrict y = y_;

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
            __asm__(VSE "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

        inc_t tmp1 = vl * incx;
        inc_t tmp2 = vl * incy;
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
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

void bli_dcopyv_sifive_x280_asm(conj_t conjx, dim_t n, const void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)conjx;
    const double* restrict x = x_;
    double* restrict y = y_;

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
            __asm__(VSE "v0, (%0)" : : "r"(y));
        else
            __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

        inc_t tmp1 = vl * incx;
        inc_t tmp2 = vl * incy;
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
        __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
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
#define VLSEG2 "vlseg2e32.v "
#define VLSSEG2 "vlsseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "

void bli_ccopyv_sifive_x280_asm(conj_t conjx, dim_t n, const void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    const scomplex* restrict x = x_;
    scomplex* restrict y = y_;

    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    if (conjx == BLIS_NO_CONJUGATE) {
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
                __asm__(VSE "v0, (%0)" : : "r"(y));
            else
                __asm__(VSSE "v0, (%0), %1" : : "r"(y), "r"(incy));

            inc_t tmp1 = vl * incx;
            inc_t tmp2 = vl * incy;
            __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
            __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
            avl -= vl;
        }
    } else {
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

            __asm__("vfneg.v v4, v4");

            if (incy == 2 * FLT_SIZE)
                __asm__(VSSEG2 "v0, (%0)" : : "r"(y));
            else
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(y), "r"(incy));

            inc_t tmp1 = vl * incx;
            inc_t tmp2 = vl * incy;
            __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
            __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
            avl -= vl;
        }
        /*
        // After some benchmarks, it looks like using vl(s)e and vs(s)e with
        masked
        // instructions for conjugation is faster than using segment loads and
        stores.
        // We'll use the segment load/store version for now, but I'd like to
        leave this
        // code here (but commented out) for possible future use.
        size_t avl = n;
        // 0xA = 0b1010
        // this masks off the real parts, so only the imaginary parts are
        negated
        // this mask is large enough only for vl <= 64
        uint64_t mask[1] = {0xAAAAAAAAAAAAAAAA};
        __asm__("vsetivli zero, 1, e64, m1, ta, ma");
        __asm__("vle64.v v0, (%0)" : : "r"(mask));
        while (avl) {
          size_t vl;
          __asm__ volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) :
        "r"(avl)); if (incx == 8)
            __asm__("vle64.v v4, (%0)" : : "r"(x));
          else
            __asm__("vlse64.v v4, (%0), %1" : : "r"(x), "r"(incx));
          // set vl = VLMAX
          __asm__ volatile("vsetvli t0, zero, e32, m4, ta, ma");
          __asm__("vfneg.v v4, v4, v0.t");
          __asm__ volatile ("vsetvli zero, %0, e64, m4, ta, ma" : : "r"(avl));
          if (incy == 8)
            __asm__("vse64.v v4, (%0)" : : "r"(y));
          else
            __asm__("vsse64.v v4, (%0), %1" : : "r"(y), "r"(incy));
          inc_t tmp1 = vl * incx;
          inc_t tmp2 = vl * incy;
          __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
          __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
          avl -= vl;
        }
        */
    }
    return;
}

#undef FLT_SIZE
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE
#undef VLSEG2
#undef VLSSEG2
#undef VSSEG2
#undef VSSSEG2

#define FLT_SIZE 8
#define SH_ADD "sh3add "
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "

void bli_zcopyv_sifive_x280_asm(conj_t conjx, dim_t n, const void * restrict x_, inc_t incx,
                     void * restrict y_, inc_t incy, const cntx_t *cntx) {
    (void)cntx;
    const dcomplex* restrict x = x_;
    dcomplex* restrict y = y_;

    incx *= 2 * FLT_SIZE;
    incy *= 2 * FLT_SIZE;
    if (conjx == BLIS_NO_CONJUGATE && incx == 2 * FLT_SIZE &&
        incy == 2 * FLT_SIZE) {
        size_t avl = 2 * n;
        while (avl) {
            size_t vl;
            __asm__ volatile("vsetvli %0, %1, e%2, m8, ta, ma"
                             : "=r"(vl)
                             : "r"(avl), "i"(8 * FLT_SIZE));
            __asm__(VLE "v0, (%0)" : : "r"(x));
            __asm__(VSE "v0, (%0)" : : "r"(y));
            __asm__(SH_ADD "%0, %1, %0" : "+r"(x) : "r"(vl));
            __asm__(SH_ADD "%0, %1, %0" : "+r"(y) : "r"(vl));
            avl -= vl;
        }
    } else {
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

            if (conjx == BLIS_CONJUGATE)
                __asm__("vfneg.v v4, v4");

            if (incy == 2 * FLT_SIZE)
                __asm__(VSSEG2 "v0, (%0)" : : "r"(y));
            else
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(y), "r"(incy));

            inc_t tmp1 = vl * incx;
            inc_t tmp2 = vl * incy;
            __asm__("add %0, %0, %1" : "+r"(x) : "r"(tmp1));
            __asm__("add %0, %0, %1" : "+r"(y) : "r"(tmp2));
            avl -= vl;
        }
    }
    return;
}
