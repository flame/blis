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

void bli_samaxv_sifive_x280_asm(dim_t n, const void * restrict x_, inc_t incx,
                     dim_t *index, const cntx_t *cntx) {
    // assumes 64-bit index
    (void)cntx;
    const float* restrict x = x_;

    if (n <= 1) {
        *index = 0;
        return;
    }
    incx *= 4;
    size_t avl = n;
    size_t offset = 0;
    bool first = true;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e32, m4, tu, ma"
                         : "=r"(vl)
                         : "r"(avl));
        if (incx == 4)
            __asm__("vle32.v v24, (%0)" : : "r"(x));
        else
            __asm__("vlse32.v v24, (%0), %1" : : "r"(x), "r"(incx));
        // check for NaN
        __asm__ volatile("vmfne.vv v0, v24, v24");
        dim_t nan_index;
        __asm__ volatile("vfirst.m %0, v0" : "=r"(nan_index));
        if (nan_index != -1) {
            *index = nan_index + offset;
            return;
        }
        if (first) {
            __asm__("vfabs.v v8, v24");
            // keep vl same, change SEW and LMUL
            __asm__ volatile("vsetvli zero, zero, e64, m8, ta, ma");
            __asm__("vid.v v16");
            first = false;
        } else {
            __asm__("vfabs.v v24, v24");
            __asm__("vmflt.vv v0, v8, v24");
            __asm__("vmerge.vvm v8, v8, v24, v0");
            // keep vl same, change SEW and LMUL
            __asm__ volatile("vsetvli zero, zero, e64, m8, tu, ma");
            __asm__("vid.v v24");
            __asm__("vadd.vx v24, v24, %0" : : "r"(offset));
            __asm__("vmerge.vvm v16, v16, v24, v0");
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        offset += vl;
        avl -= vl;
    }
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(n));
    __asm__("vmv.s.x v0, zero");
    __asm__("vfredmax.vs v0, v8, v0");
    __asm__("vrgather.vi v24, v0, 0");
    __asm__("vmfeq.vv v0, v8, v24");
    __asm__ volatile("vsetvli zero, zero, e64, m8, ta, ma");
    uint64_t imax = -1;
    __asm__("vmv.s.x v24, %0" : : "r"(imax));
    __asm__("vredminu.vs v24, v16, v24, v0.t");
    __asm__ volatile("vsetivli zero, 1, e64, m1, ta, ma");
    __asm__("vse64.v v24, (%0)" : : "r"(index));
    return;
}

void bli_damaxv_sifive_x280_asm(dim_t n, const void * restrict x_, inc_t incx,
                     dim_t *index, const cntx_t *cntx) {
    // assumes 64-bit index
    (void)cntx;
    const double* restrict x = x_;

    if (n <= 1) {
        *index = 0;
        return;
    }
    incx *= 8;
    size_t avl = n;
    size_t offset = 0;
    bool first = true;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e64, m8, tu, ma"
                         : "=r"(vl)
                         : "r"(avl));
        if (incx == 8)
            __asm__("vle64.v v24, (%0)" : : "r"(x));
        else
            __asm__("vlse64.v v24, (%0), %1" : : "r"(x), "r"(incx));
        // check for NaN
        __asm__ volatile("vmfne.vv v0, v24, v24");
        dim_t nan_index;
        __asm__ volatile("vfirst.m %0, v0" : "=r"(nan_index));
        if (nan_index != -1) {
            *index = nan_index + offset;
            return;
        }
        if (first) {
            __asm__("vfabs.v v8, v24");
            __asm__("vid.v v16");
            first = false;
        } else {
            __asm__("vfabs.v v24, v24");
            __asm__("vmflt.vv v0, v8, v24");
            __asm__("vmerge.vvm v8, v8, v24, v0");
            __asm__("vid.v v24");
            __asm__("vadd.vx v24, v24, %0" : : "r"(offset));
            __asm__("vmerge.vvm v16, v16, v24, v0");
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        offset += vl;
        avl -= vl;
    }
    __asm__ volatile("vsetvli zero, %0, e64, m8, ta, ma" : : "r"(n));
    __asm__("vmv.s.x v0, zero");
    __asm__("vfredmax.vs v0, v8, v0");
    __asm__("vrgather.vi v24, v0, 0");
    __asm__("vmfeq.vv v0, v8, v24");
    uint64_t imax = -1;
    __asm__("vmv.s.x v24, %0" : : "r"(imax));
    __asm__("vredminu.vs v24, v16, v24, v0.t");
    __asm__ volatile("vsetivli zero, 1, e64, m1, ta, ma");
    __asm__("vse64.v v24, (%0)" : : "r"(index));
    return;
}

void bli_camaxv_sifive_x280_asm(dim_t n, const void * restrict x_, inc_t incx,
                     dim_t *index, const cntx_t *cntx) {
    // assumes 64-bit index
    (void)cntx;
    const scomplex* restrict x = x_;

    if (n <= 1) {
        *index = 0;
        return;
    }
    incx *= 8;
    size_t avl = n;
    size_t offset = 0;
    bool first = true;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e32, m4, tu, ma"
                         : "=r"(vl)
                         : "r"(avl));
        if (incx == 8)
            __asm__("vlseg2e32.v v24, (%0)" : : "r"(x));
        else
            __asm__("vlsseg2e32.v v24, (%0), %1" : : "r"(x), "r"(incx));
        __asm__("vfabs.v v24, v24");
        __asm__("vfabs.v v28, v28");
        __asm__("vfadd.vv v24, v24, v28");
        // check for NaN
        __asm__ volatile("vmfne.vv v0, v24, v24");
        dim_t nan_index;
        __asm__ volatile("vfirst.m %0, v0" : "=r"(nan_index));
        if (nan_index != -1) {
            *index = nan_index + offset;
            return;
        }
        if (first) {
            __asm__("vmv4r.v v8, v24");
            // keep vl same, change SEW and LMUL
            __asm__ volatile("vsetvli zero, zero, e64, m8, ta, ma");
            __asm__("vid.v v16");
            first = false;
        } else {
            __asm__("vmflt.vv v0, v8, v24");
            __asm__("vmerge.vvm v8, v8, v24, v0");
            // keep vl same, change SEW and LMUL
            __asm__ volatile("vsetvli zero, zero, e64, m8, tu, ma");
            __asm__("vid.v v24");
            __asm__("vadd.vx v24, v24, %0" : : "r"(offset));
            __asm__("vmerge.vvm v16, v16, v24, v0");
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        offset += vl;
        avl -= vl;
    }
    __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(n));
    __asm__("vmv.s.x v0, zero");
    __asm__("vfredmax.vs v0, v8, v0");
    __asm__("vrgather.vi v24, v0, 0");
    __asm__("vmfeq.vv v0, v8, v24");
    __asm__ volatile("vsetvli zero, zero, e64, m8, ta, ma");
    uint64_t imax = -1;
    __asm__("vmv.s.x v24, %0" : : "r"(imax));
    __asm__("vredminu.vs v24, v16, v24, v0.t");
    __asm__ volatile("vsetivli zero, 1, e64, m1, ta, ma");
    __asm__("vse64.v v24, (%0)" : : "r"(index));
    return;
}

void bli_zamaxv_sifive_x280_asm(dim_t n, const void * restrict x_, inc_t incx,
                     dim_t *index, const cntx_t *cntx) {
    // assumes 64-bit index
    (void)cntx;
    const dcomplex* restrict x = x_;

    if (n <= 1) {
        *index = 0;
        return;
    }
    incx *= 16;
    size_t avl = n;
    size_t offset = 0;
    bool first = true;
    while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e64, m4, tu, ma"
                         : "=r"(vl)
                         : "r"(avl));
        if (incx == 16)
            __asm__("vlseg2e64.v v24, (%0)" : : "r"(x));
        else
            __asm__("vlsseg2e64.v v24, (%0), %1" : : "r"(x), "r"(incx));
        __asm__("vfabs.v v24, v24");
        __asm__("vfabs.v v28, v28");
        __asm__("vfadd.vv v24, v24, v28");
        // check for NaN
        __asm__ volatile("vmfne.vv v0, v24, v24");
        dim_t nan_index;
        __asm__ volatile("vfirst.m %0, v0" : "=r"(nan_index));
        if (nan_index != -1) {
            *index = nan_index + offset;
            return;
        }
        if (first) {
            __asm__("vmv4r.v v8, v24");
            __asm__("vid.v v16");
            first = false;
        } else {
            __asm__("vmflt.vv v0, v8, v24");
            __asm__("vmerge.vvm v8, v8, v24, v0");
            __asm__("vid.v v24");
            __asm__("vadd.vx v24, v24, %0" : : "r"(offset));
            __asm__("vmerge.vvm v16, v16, v24, v0");
        }
        __asm__("add %0, %0, %1" : "+r"(x) : "r"(vl * incx));
        offset += vl;
        avl -= vl;
    }
    __asm__ volatile("vsetvli zero, %0, e64, m4, ta, ma" : : "r"(n));
    __asm__("vmv.s.x v0, zero");
    __asm__("vfredmax.vs v0, v8, v0");
    __asm__("vrgather.vi v24, v0, 0");
    __asm__("vmfeq.vv v0, v8, v24");
    uint64_t imax = -1;
    __asm__("vmv.s.x v24, %0" : : "r"(imax));
    __asm__("vredminu.vs v24, v16, v24, v0.t");
    __asm__ volatile("vsetivli zero, 1, e64, m1, ta, ma");
    __asm__("vse64.v v24, (%0)" : : "r"(index));
    return;
}
