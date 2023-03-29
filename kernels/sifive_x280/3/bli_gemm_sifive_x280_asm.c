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

// clang-format off
#include "blis.h"
#include "../riscv_cmul_macros_asm.h"
#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <riscv_vector.h>

// byte-size of the floating point type
#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define VLE "vle32.v "
#define VLSE "vlse32.v "
#define VSE "vse32.v "
#define VSSE "vsse32.v "
#define PACKMR 8
#define PACKNR 64

void bli_sgemm_7m4(dim_t N, dim_t K, const float* restrict alpha,
        const float* restrict a, const float* restrict b, const float* restrict beta,
        float* restrict c, inc_t rsc, inc_t csc) {
    // 7 x N x K sgemm, 0 < N <= 64 = vlmax, K > 0
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        __asm__(VLE "v28, (%0)" : : "r"(b));
        if (first) {
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__("vfmul.vf v0, v28, ft0");

            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            __asm__("vfmul.vf v4, v28, ft1");

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__("vfmul.vf v8, v28, ft2");

            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            __asm__("vfmul.vf v12, v28, ft3");

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__("vfmul.vf v16, v28, ft4");

            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            __asm__("vfmul.vf v20, v28, ft5");

            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a), "I"(6 * FLT_SIZE));
            __asm__("vfmul.vf v24, v28, ft6");

            first = false;
        }
        else {
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__("vfmacc.vf v0, ft0, v28");

            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            __asm__("vfmacc.vf v4, ft1, v28");

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__("vfmacc.vf v8, ft2, v28");

            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            __asm__("vfmacc.vf v12, ft3, v28");

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__("vfmacc.vf v16, ft4, v28");

            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            __asm__("vfmacc.vf v20, ft5, v28");

            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a), "I"(6 * FLT_SIZE));
            __asm__("vfmacc.vf v24, ft6, v28");
        }

        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * FLT_SIZE));
    }

    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
    
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    
    // compute alpha*a*b + beta*c
    if (*beta == 0.f) {
        __asm__("vfmul.vf v0, v0, ft10");
        __asm__("vfmul.vf v4, v4, ft10");
        __asm__("vfmul.vf v8, v8, ft10");
        __asm__("vfmul.vf v12, v12, ft10");
        __asm__("vfmul.vf v16, v16, ft10");
        __asm__("vfmul.vf v20, v20, ft10");
        __asm__("vfmul.vf v24, v24, ft10");
    }
    else { // beta != 0.f
        __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
        float *c_tmp = c;
        if (csc == FLT_SIZE) { // c unit column stride
            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v0, v0, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v0, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v4, v4, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v4, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v8, v8, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v8, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v12, v12, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v12, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v16, v16, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v16, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v20, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v24, v24, ft10");
            __asm__("vfmacc.vf v24, ft11, v28");
        } // end c unit column stride
        else { // c non-unit column stride
            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v0, v0, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v0, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v4, v4, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v4, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v8, v8, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v8, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v12, v12, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v12, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v16, v16, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v16, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v20, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v24, v24, ft10");
            __asm__("vfmacc.vf v24, ft11, v28");
        } // end c non-unit column stride
    } // end beta != 0.f

    // store c
    if (csc == FLT_SIZE) {
        __asm__(VSE "v0, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v4, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v8, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v12, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v16, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v20, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v24, (%0)" : : "r"(c));
    }
    else {
        __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v24, (%0), %1" : : "r"(c), "r"(csc));
    }

    return;
}

void bli_sgemm_7m4_cleanup(dim_t M, dim_t N, dim_t K,
        const float* restrict alpha, const float* restrict a, const float* restrict b,
        const float* restrict beta, float* restrict c, inc_t rsc, inc_t csc) {
    // M x N x K sgemm, 0 < M < 6, 0 < N <= 64 = vlmax, K > 0
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        __asm__(VLE "v28, (%0)" : : "r"(b));
        if (first) {
            switch (M) {
            case 6:
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
                __asm__("vfmul.vf v20, v28, ft5");
            case 5:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
                __asm__("vfmul.vf v16, v28, ft4");
            case 4:
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                __asm__("vfmul.vf v12, v28, ft3");
            case 3:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__("vfmul.vf v8, v28, ft2");
            case 2:
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                __asm__("vfmul.vf v4, v28, ft1");
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__("vfmul.vf v0, v28, ft0");
            }
            first = false;
        }
        else {
            switch (M) {
            case 6:
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
                __asm__("vfmacc.vf v20, ft5, v28");
            case 5:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
                __asm__("vfmacc.vf v16, ft4, v28");
            case 4:
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                __asm__("vfmacc.vf v12, ft3, v28");
            case 3:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__("vfmacc.vf v8, ft2, v28");
            case 2:
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                __asm__("vfmacc.vf v4, ft1, v28");
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__("vfmacc.vf v0, ft0, v28");
            }
        }
        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * FLT_SIZE));
    }

    c += (M - 1) * rsc;
    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
     
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    
    // compute alpha*a*b + beta*c
    if (*beta == 0.f) {
        switch (M) {
        case 6:
            __asm__("vfmul.vf v20, v20, ft10");
        case 5:
            __asm__("vfmul.vf v16, v16, ft10");
        case 4:
            __asm__("vfmul.vf v12, v12, ft10");
        case 3:
            __asm__("vfmul.vf v8, v8, ft10");
        case 2:
            __asm__("vfmul.vf v4, v4, ft10");
        case 1:
            __asm__("vfmul.vf v0, v0, ft10");
        }
    }
    else { // beta != 0.f
        __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
        float *c_tmp = c;
        if (csc == FLT_SIZE) {
            switch (M) {
            case 6:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v20, v20, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v20, ft11, v28");
            case 5:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v16, v16, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v16, ft11, v28");
            case 4:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v12, v12, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v12, ft11, v28");
            case 3:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v8, v8, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v8, ft11, v28");
            case 2:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v4, v4, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v4, ft11, v28");
            case 1:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v0, v0, ft10");
                __asm__("vfmacc.vf v0, ft11, v28");
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 6:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v20, v20, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v20, ft11, v28");
            case 5:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v16, v16, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v16, ft11, v28");
            case 4:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v12, v12, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v12, ft11, v28");
            case 3:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v8, v8, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v8, ft11, v28");
            case 2:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v4, v4, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v4, ft11, v28");
            case 1:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v0, v0, ft10");
                __asm__("vfmacc.vf v0, ft11, v28");
            }
        } // end c non-unit column stride
    } // end beta != 0.f

    // store c
    if (csc == FLT_SIZE) {
        switch (M) {
        case 6:
            __asm__(VSE "v20, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 5:
            __asm__(VSE "v16, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 4:
            __asm__(VSE "v12, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 3:
            __asm__(VSE "v8, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 2:
            __asm__(VSE "v4, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSE "v0, (%0)" : : "r"(c));
        }
    }
    else {
        switch (M) {
        case 6:
            __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 5:
            __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 4:
            __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 3:
            __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 2:
            __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
        }
    }
    return;
}

void bli_sgemm_7m4_k0(dim_t M, dim_t N, const float* restrict beta,
        float* restrict c, inc_t rsc, inc_t csc) {
    // 0 < M <= 7, 0 < N < 64 = vlmax, K = 0
    // This may not produce the same result as the reference kernel if alpha is infinite or NaN.
    __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    c += (M - 1) * rsc;
    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
    if (*beta == 0.f) {
        // set c to 0
        __asm__("vmv.v.i v0, 0");
        if (csc == FLT_SIZE) { // c unit column stride
            switch (M) {
            case 7:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSE "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 7:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
            }        
        } // end c non-unit column stride
    } // end beta == 0.f
    else { // beta != 0.f
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        if (csc == FLT_SIZE) { // c unit column stride
            switch (M) {
            case 7:
                __asm__(VLE "v24, (%0)" : : "r"(c));
                __asm__("vfmul.vf v24, v24, ft0");
                __asm__(VSE "v24, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VLE "v20, (%0)" : : "r"(c));
                __asm__("vfmul.vf v20, v20, ft0");
                __asm__(VSE "v20, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VLE "v16, (%0)" : : "r"(c));
                __asm__("vfmul.vf v16, v16, ft0");
                __asm__(VSE "v16, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VLE "v12, (%0)" : : "r"(c));
                __asm__("vfmul.vf v12, v12, ft0");
                __asm__(VSE "v12, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VLE "v8, (%0)" : : "r"(c));
                __asm__("vfmul.vf v8, v8, ft0");
                __asm__(VSE "v8, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLE "v4, (%0)" : : "r"(c));
                __asm__("vfmul.vf v4, v4, ft0");
                __asm__(VSE "v4, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLE "v0, (%0)" : : "r"(c));
                __asm__("vfmul.vf v0, v0, ft0");
                __asm__(VSE "v0, (%0)" : : "r"(c));
                
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 7:
                __asm__(VLSE "v24, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v24, v24, ft0");
                __asm__(VSSE "v24, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VLSE "v20, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v20, v20, ft0");
                __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VLSE "v16, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v16, v16, ft0");
                __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VLSE "v12, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v12, v12, ft0");
                __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VLSE "v8, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v8, v8, ft0");
                __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSE "v4, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v4, v4, ft0");
                __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v0, v0, ft0");
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta != 0.f
    return;
}

void bli_sgemm_sifive_x280_asm_7m4(dim_t M, dim_t N, dim_t K,
        const void* restrict alpha_, const void* restrict a_, const void* restrict b_,
        const void* restrict beta_, void* restrict c_, inc_t rsc, inc_t csc,
        auxinfo_t* restrict data, const cntx_t* restrict cntx) {
    (void) data;
    (void) cntx;
    const float* restrict alpha = alpha_;
    const float* restrict beta = beta_;
    const float* restrict a = a_;
    const float* restrict b = b_;
    float* restrict c = c_;

    // M x N x K sgemm
    if (M <= 0 || N <= 0 || K < 0)
        return;
    else if (K == 0)
        bli_sgemm_7m4_k0(M, N, beta, c, rsc, csc);
    else if (M == 7)
        bli_sgemm_7m4(N, K, alpha, a, b, beta, c, rsc, csc);
    else
        bli_sgemm_7m4_cleanup(M, N, K, alpha, a, b, beta, c, rsc, csc);
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE
#undef PACKMR
#undef PACKNR

// byte-size of the floating point type
#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define VLE "vle64.v "
#define VLSE "vlse64.v "
#define VSE "vse64.v "
#define VSSE "vsse64.v "
#define PACKMR 8
#define PACKNR 32 

void bli_dgemm_7m4(dim_t N, dim_t K, const double* restrict alpha,
        const double* restrict a, const double* restrict b, const double* restrict beta,
        double* restrict c, inc_t rsc, inc_t csc) {
    // 7 x N x K dgemm, 0 < N <= 64 = vlmax, K > 0
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        __asm__(VLE "v28, (%0)" : : "r"(b));
        if (first) {
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__("vfmul.vf v0, v28, ft0");

            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            __asm__("vfmul.vf v4, v28, ft1");

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__("vfmul.vf v8, v28, ft2");

            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            __asm__("vfmul.vf v12, v28, ft3");

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__("vfmul.vf v16, v28, ft4");

            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            __asm__("vfmul.vf v20, v28, ft5");

            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a), "I"(6 * FLT_SIZE));
            __asm__("vfmul.vf v24, v28, ft6");

            first = false;
        }
        else {
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__("vfmacc.vf v0, ft0, v28");

            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            __asm__("vfmacc.vf v4, ft1, v28");

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__("vfmacc.vf v8, ft2, v28");

            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            __asm__("vfmacc.vf v12, ft3, v28");

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__("vfmacc.vf v16, ft4, v28");

            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            __asm__("vfmacc.vf v20, ft5, v28");

            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a), "I"(6 * FLT_SIZE));
            __asm__("vfmacc.vf v24, ft6, v28");
        }

        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * FLT_SIZE));
    }

    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
    
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    
    // compute alpha*a*b + beta*c
    if (*beta == 0.) {
        __asm__("vfmul.vf v0, v0, ft10");
        __asm__("vfmul.vf v4, v4, ft10");
        __asm__("vfmul.vf v8, v8, ft10");
        __asm__("vfmul.vf v12, v12, ft10");
        __asm__("vfmul.vf v16, v16, ft10");
        __asm__("vfmul.vf v20, v20, ft10");
        __asm__("vfmul.vf v24, v24, ft10");
    }
    else { // beta != 0.
        __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
        double *c_tmp = c;
        if (csc == FLT_SIZE) { // c unit column stride
            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v0, v0, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v0, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v4, v4, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v4, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v8, v8, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v8, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v12, v12, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v12, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v16, v16, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v16, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v20, ft11, v28");

            __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
            __asm__("vfmul.vf v24, v24, ft10");
            __asm__("vfmacc.vf v24, ft11, v28");
        } // end c unit column stride
        else { // c non-unit column stride
            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v0, v0, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v0, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v4, v4, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v4, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v8, v8, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v8, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v12, v12, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v12, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v16, v16, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v16, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v20, v20, ft10");
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            __asm__("vfmacc.vf v20, ft11, v28");

            __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("vfmul.vf v24, v24, ft10");
            __asm__("vfmacc.vf v24, ft11, v28");
        } // end c non-unit column stride
    } // end beta != 0.

    // store c
    if (csc == FLT_SIZE) {
        __asm__(VSE "v0, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v4, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v8, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v12, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v16, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v20, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSE "v24, (%0)" : : "r"(c));
    }
    else {
        __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSE "v24, (%0), %1" : : "r"(c), "r"(csc));
    }

    return;
}

void bli_dgemm_7m4_cleanup(dim_t M, dim_t N, dim_t K,
        const double* restrict alpha, const double* restrict a, const double* restrict b,
        const double* restrict beta, double* restrict c, inc_t rsc, inc_t csc) {
    // M x N x K dgemm, 0 < M < 6, 0 < N <= 64 = vlmax, K > 0
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        __asm__(VLE "v28, (%0)" : : "r"(b));
        if (first) {
            switch (M) {
            case 6:
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
                __asm__("vfmul.vf v20, v28, ft5");
            case 5:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
                __asm__("vfmul.vf v16, v28, ft4");
            case 4:
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                __asm__("vfmul.vf v12, v28, ft3");
            case 3:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__("vfmul.vf v8, v28, ft2");
            case 2:
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                __asm__("vfmul.vf v4, v28, ft1");
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__("vfmul.vf v0, v28, ft0");
            }
            first = false;
        }
        else {
            switch (M) {
            case 6:
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
                __asm__("vfmacc.vf v20, ft5, v28");
            case 5:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
                __asm__("vfmacc.vf v16, ft4, v28");
            case 4:
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                __asm__("vfmacc.vf v12, ft3, v28");
            case 3:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__("vfmacc.vf v8, ft2, v28");
            case 2:
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                __asm__("vfmacc.vf v4, ft1, v28");
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__("vfmacc.vf v0, ft0, v28");
            }
        }
        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * FLT_SIZE));
    }

    c += (M - 1) * rsc;
    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
     
    __asm__(FLT_LOAD "ft10, (%0)" : : "r"(alpha));
    
    // compute alpha*a*b + beta*c
    if (*beta == 0.) {
        switch (M) {
        case 6:
            __asm__("vfmul.vf v20, v20, ft10");
        case 5:
            __asm__("vfmul.vf v16, v16, ft10");
        case 4:
            __asm__("vfmul.vf v12, v12, ft10");
        case 3:
            __asm__("vfmul.vf v8, v8, ft10");
        case 2:
            __asm__("vfmul.vf v4, v4, ft10");
        case 1:
            __asm__("vfmul.vf v0, v0, ft10");
        }
    }
    else { // beta != 0.
        __asm__(FLT_LOAD "ft11, (%0)" : : "r"(beta));
        double *c_tmp = c;
        if (csc == FLT_SIZE) {
            switch (M) {
            case 6:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v20, v20, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v20, ft11, v28");
            case 5:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v16, v16, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v16, ft11, v28");
            case 4:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v12, v12, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v12, ft11, v28");
            case 3:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v8, v8, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v8, ft11, v28");
            case 2:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v4, v4, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v4, ft11, v28");
            case 1:
                __asm__(VLE "v28, (%0)" : : "r"(c_tmp));
                __asm__("vfmul.vf v0, v0, ft10");
                __asm__("vfmacc.vf v0, ft11, v28");
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 6:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v20, v20, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v20, ft11, v28");
            case 5:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v16, v16, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v16, ft11, v28");
            case 4:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v12, v12, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v12, ft11, v28");
            case 3:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v8, v8, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v8, ft11, v28");
            case 2:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v4, v4, ft10");
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                __asm__("vfmacc.vf v4, ft11, v28");
            case 1:
                __asm__(VLSE "v28, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("vfmul.vf v0, v0, ft10");
                __asm__("vfmacc.vf v0, ft11, v28");
            }
        } // end c non-unit column stride
    } // end beta != 0.

    // store c
    if (csc == FLT_SIZE) {
        switch (M) {
        case 6:
            __asm__(VSE "v20, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 5:
            __asm__(VSE "v16, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 4:
            __asm__(VSE "v12, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 3:
            __asm__(VSE "v8, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 2:
            __asm__(VSE "v4, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSE "v0, (%0)" : : "r"(c));
        }
    }
    else {
        switch (M) {
        case 6:
            __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 5:
            __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 4:
            __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 3:
            __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 2:
            __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
        }
    }
    return;
}

void bli_dgemm_7m4_k0(dim_t M, dim_t N, const double* restrict beta,
        double* restrict c, inc_t rsc, inc_t csc) {
    // 0 < M <= 7, 0 < N < 64 = vlmax, K = 0
    // This may not produce the same result as the reference kernel if alpha is infinite or NaN.
    __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    c += (M - 1) * rsc;
    rsc *= FLT_SIZE;
    csc *= FLT_SIZE;
    if (*beta == 0.) {
        // set c to 0
        __asm__("vmv.v.i v0, 0");
        if (csc == FLT_SIZE) { // c unit column stride
            switch (M) {
            case 7:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSE "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 7:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
            }        
        } // end c non-unit column stride
    } // end beta == 0.
    else { // beta != 0.
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        if (csc == FLT_SIZE) { // c unit column stride
            switch (M) {
            case 7:
                __asm__(VLE "v24, (%0)" : : "r"(c));
                __asm__("vfmul.vf v24, v24, ft0");
                __asm__(VSE "v24, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VLE "v20, (%0)" : : "r"(c));
                __asm__("vfmul.vf v20, v20, ft0");
                __asm__(VSE "v20, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VLE "v16, (%0)" : : "r"(c));
                __asm__("vfmul.vf v16, v16, ft0");
                __asm__(VSE "v16, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VLE "v12, (%0)" : : "r"(c));
                __asm__("vfmul.vf v12, v12, ft0");
                __asm__(VSE "v12, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VLE "v8, (%0)" : : "r"(c));
                __asm__("vfmul.vf v8, v8, ft0");
                __asm__(VSE "v8, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLE "v4, (%0)" : : "r"(c));
                __asm__("vfmul.vf v4, v4, ft0");
                __asm__(VSE "v4, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLE "v0, (%0)" : : "r"(c));
                __asm__("vfmul.vf v0, v0, ft0");
                __asm__(VSE "v0, (%0)" : : "r"(c));
                
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 7:
                __asm__(VLSE "v24, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v24, v24, ft0");
                __asm__(VSSE "v24, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 6:
                __asm__(VLSE "v20, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v20, v20, ft0");
                __asm__(VSSE "v20, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 5:
                __asm__(VLSE "v16, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v16, v16, ft0");
                __asm__(VSSE "v16, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 4:
                __asm__(VLSE "v12, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v12, v12, ft0");
                __asm__(VSSE "v12, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 3:
                __asm__(VLSE "v8, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v8, v8, ft0");
                __asm__(VSSE "v8, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSE "v4, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v4, v4, ft0");
                __asm__(VSSE "v4, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSE "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("vfmul.vf v0, v0, ft0");
                __asm__(VSSE "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta != 0.
    return;
}

void bli_dgemm_sifive_x280_asm_7m4(dim_t M, dim_t N, dim_t K,
        const void* restrict alpha_, const void* restrict a_, const void* restrict b_,
        const void* restrict beta_, void* restrict c_, inc_t rsc, inc_t csc,
        auxinfo_t* restrict data, const cntx_t* restrict cntx) {
    (void) data;
    (void) cntx;
    const double* restrict alpha = alpha_;
    const double* restrict beta = beta_;
    const double* restrict a = a_;
    const double* restrict b = b_;
    double* restrict c = c_;

    // M x N x K dgemm
    if (M <= 0 || N <= 0 || K < 0)
        return;
    else if (K == 0)
        bli_dgemm_7m4_k0(M, N, beta, c, rsc, csc);
    else if (M == 7)
        bli_dgemm_7m4(N, K, alpha, a, b, beta, c, rsc, csc);
    else
        bli_dgemm_7m4_cleanup(M, N, K, alpha, a, b, beta, c, rsc, csc);
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VLE
#undef VLSE
#undef VSE
#undef VSSE
#undef PACKMR
#undef PACKNR

#define FLT_SIZE 4
#define FLT_LOAD "flw "
#define VSE "vse32.v "
#define VLSEG2 "vlseg2e32.v "
#define VLSSEG2 "vlsseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "
#define PACKMR 4
#define PACKNR 64

void bli_cgemm_3m4(dim_t N, dim_t K, const scomplex* restrict alpha,
        const scomplex* restrict a, const scomplex* restrict b, const scomplex* restrict beta,
        scomplex* restrict c, inc_t rsc, inc_t csc) {
    // 3 x N x K cgemm, 0 < N <= 64 = vlmax, K > 0
    // pairs of register groups hold the real and imag. parts of rows of c and b
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        // v24 = Re(b), v28 = Im(b)
        __asm__(VLSEG2 "v24, (%0)" : : "r"(b));
        if (first) {
            // ft0 = Re(*a), ft1 = Im(*a)
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            vcmul_vf(v0, v4, v24, v28, ft0, ft1); 

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            vcmul_vf(v8, v12, v24, v28, ft2, ft3);

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            vcmul_vf(v16, v20, v24, v28, ft4, ft5);

            first = false;
        }
        else {
            // ft0 = Re(*a), ft1 = Im(*a)
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            vcmacc_vf(v8, v12, ft2, ft3, v24, v28);

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            vcmacc_vf(v16, v20, ft4, ft5, v24, v28);
        }

        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * 2 * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * 2 * FLT_SIZE));
    }

    __asm__(FLT_LOAD "ft0, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));

    // scale a*b by alpha
    __asm__("vfmul.vf v24, v4, ft1");
    __asm__("vfmul.vf v28, v0, ft1");
    __asm__("vfmsub.vf v0, ft0, v24");
    __asm__("vfmadd.vf v4, ft0, v28");

    __asm__("vfmul.vf v24, v12, ft1");
    __asm__("vfmul.vf v28, v8, ft1");
    __asm__("vfmsub.vf v8, ft0, v24");
    __asm__("vfmadd.vf v12, ft0, v28");

    __asm__("vfmul.vf v24, v20, ft1");
    __asm__("vfmul.vf v28, v16, ft1");
    __asm__("vfmsub.vf v16, ft0, v24");
    __asm__("vfmadd.vf v20, ft0, v28");

    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;

    if (beta->real != 0.f || beta->imag != 0.f) {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        scomplex* c_tmp = c;

        if (csc == 2 * FLT_SIZE) {
            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v8, v12, ft0, ft1, v24, v28);

            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            vcmacc_vf(v16, v20, ft0, ft1, v24, v28);
        }
        else {
            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v8, v12, ft0, ft1, v24, v28);

            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            vcmacc_vf(v16, v20, ft0, ft1, v24, v28);
        }
    }

    if (csc == 2 * FLT_SIZE) {
        __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSEG2 "v8, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSEG2 "v16, (%0)" : : "r"(c));
    }
    else {
        __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSSEG2 "v16, (%0), %1" : : "r"(c), "r"(csc));
    }

    return;
}

void bli_cgemm_3m4_cleanup(dim_t M, dim_t N, dim_t K, const scomplex* restrict alpha,
        const scomplex* restrict a, const scomplex* restrict b, const scomplex* restrict beta,
        scomplex* restrict c, inc_t rsc, inc_t csc) {
    // M x N x K cgemm, 0 < M < 3, 0 < N <= 64 = vlmax, K > 0
    // pairs of register groups hold the real and imag. parts of rows of c and b
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "I"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        // v24 = Re(b), v28 = Im(b)
        __asm__(VLSEG2 "v24, (%0)" : : "r"(b));
        if (first) {
            // ft0 = Re(*a), ft1 = Im(*a)
            switch (M) {
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                vcmul_vf(v8, v12, v24, v28, ft2, ft3);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                vcmul_vf(v0, v4, v24, v28, ft0, ft1);
            }
            first = false;
        }
        else {
            // ft0 = Re(*a), ft1 = Im(*a)
            switch (M) {
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                vcmacc_vf(v8, v12, ft2, ft3, v24, v28);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * 2 * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * 2 * FLT_SIZE));
    }

    __asm__(FLT_LOAD "ft0, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));

    // scale a*b by alpha
    switch (M) {
    case 2:
        __asm__("vfmul.vf v24, v12, ft1");
        __asm__("vfmul.vf v28, v8, ft1");
        __asm__("vfmsub.vf v8, ft0, v24");
        __asm__("vfmadd.vf v12, ft0, v28");
    case 1:
        __asm__("vfmul.vf v24, v4, ft1");
        __asm__("vfmul.vf v28, v0, ft1");
        __asm__("vfmsub.vf v0, ft0, v24");
        __asm__("vfmadd.vf v4, ft0, v28");
    }

    c += (M - 1) * rsc;
    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;

    if (beta->real != 0.f || beta->imag != 0.f) {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        scomplex* c_tmp = c;

        if (csc == 2 * FLT_SIZE) {
            switch (M) {
            case 2:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                vcmacc_vf(v8, v12, ft0, ft1, v24, v28);
            case 1:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
        else {
            switch (M) {
            case 2:
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                vcmacc_vf(v8, v12, ft0, ft1, v24, v28);
            case 1:
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
    } // end beta != 0.f + 0.fi

    if (csc == 2 * FLT_SIZE) {
        switch (M) {
        case 2:
            __asm__(VSSEG2 "v8, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
        }
    }
    else {
        switch (M) {
        case 2:
            __asm__(VSSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
        }
    }
    return;
}

void bli_cgemm_3m4_k0(dim_t M, dim_t N, const scomplex* restrict beta,
        scomplex* restrict c, inc_t rsc, inc_t csc) {
    // 0 < M <= 3, 0 < N <= 64 = vlmax, K = 0
    // This may not produce the same result as the reference kernel if alpha is infinite or NaN.
    c += (M - 1) * rsc;
    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;
    if (beta->real == 0.f && beta->imag == 0.f) {
        if (csc == 2 * FLT_SIZE) { // c unit column stride
            // store 2*N 0.f's per row
            __asm__ volatile("vsetvli zero, %0, e%1, m8, ta, ma" : : "r"(2 * N), "i"(8 * FLT_SIZE));
            __asm__("vmv.v.i v0, 0");
            switch (M) {
            case 3:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSE "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
            __asm__("vmv.v.i v0, 0");
            __asm__("vmv.v.i v4, 0");
            switch (M) {
            case 3:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta == 0.f + 0.fi
    else { // beta != 0.f + 0.fi
        __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        if (csc == 2 * FLT_SIZE) { // c unit column stride
            switch (M) {
            case 3:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 3:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta != 0.f + 0.fi
    return;
}

void bli_cgemm_sifive_x280_asm_3m4(dim_t M, dim_t N, dim_t K,
        const void* restrict alpha_, const void* restrict a_, const void* restrict b_,
        const void* restrict beta_, void* restrict c_, inc_t rsc, inc_t csc,
        auxinfo_t* restrict data, const cntx_t* restrict cntx) {
    (void) data;
    (void) cntx;
    const scomplex* restrict alpha = alpha_;
    const scomplex* restrict beta = beta_;
    const scomplex* restrict a = a_;
    const scomplex* restrict b = b_;
    scomplex* restrict c = c_;
    
    // M x N x K cgemm, M <= 3, N <= 64
    if (M <= 0 || N <= 0 || K < 0)
        return;
    else if (K == 0)
        bli_cgemm_3m4_k0(M, N, beta, c, rsc, csc);
    else if (M == 3)
        bli_cgemm_3m4(N, K, alpha, a, b, beta, c, rsc, csc);
    else
        bli_cgemm_3m4_cleanup(M, N, K, alpha, a, b, beta, c, rsc, csc);
    return;
}

#undef FLT_SIZE
#undef FLT_LOAD
#undef VSE
#undef VLSEG2
#undef VLSSEG2
#undef VSSEG2
#undef VSSSEG2
#undef PACKMR
#undef PACKNR

#define FLT_SIZE 8
#define FLT_LOAD "fld "
#define VSE "vse64.v "
#define VLSEG2 "vlseg2e64.v "
#define VLSSEG2 "vlsseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "
#define PACKMR 4
#define PACKNR 32 

void bli_zgemm_3m4(dim_t N, dim_t K, const dcomplex* restrict alpha,
        const dcomplex* restrict a, const dcomplex* restrict b, const dcomplex* restrict beta,
        dcomplex* restrict c, inc_t rsc, inc_t csc) {
    // 3 x N x K zgemm, 0 < N <= 32 = vlmax, K > 0
    // pairs of register groups hold the real and imag. parts of rows of c and b
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        // v24 = Re(b), v28 = Im(b)
        __asm__(VLSEG2 "v24, (%0)" : : "r"(b));
        if (first) {
            // ft0 = Re(*a), ft1 = Im(*a)
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            vcmul_vf(v0, v4, v24, v28, ft0, ft1); 

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            vcmul_vf(v8, v12, v24, v28, ft2, ft3);

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            vcmul_vf(v16, v20, v24, v28, ft4, ft5);

            first = false;
        }
        else {
            // ft0 = Re(*a), ft1 = Im(*a)
            __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
            __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
            vcmacc_vf(v8, v12, ft2, ft3, v24, v28);

            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a), "I"(4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a), "I"(5 * FLT_SIZE));
            vcmacc_vf(v16, v20, ft4, ft5, v24, v28);
        }

        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * 2 * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * 2 * FLT_SIZE));
    }

    __asm__(FLT_LOAD "ft0, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));

    // scale a*b by alpha
    __asm__("vfmul.vf v24, v4, ft1");
    __asm__("vfmul.vf v28, v0, ft1");
    __asm__("vfmsub.vf v0, ft0, v24");
    __asm__("vfmadd.vf v4, ft0, v28");

    __asm__("vfmul.vf v24, v12, ft1");
    __asm__("vfmul.vf v28, v8, ft1");
    __asm__("vfmsub.vf v8, ft0, v24");
    __asm__("vfmadd.vf v12, ft0, v28");

    __asm__("vfmul.vf v24, v20, ft1");
    __asm__("vfmul.vf v28, v16, ft1");
    __asm__("vfmsub.vf v16, ft0, v24");
    __asm__("vfmadd.vf v20, ft0, v28");

    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;

    if (beta->real != 0. || beta->imag != 0.) {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        dcomplex* c_tmp = c;

        if (csc == 2 * FLT_SIZE) {
            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v8, v12, ft0, ft1, v24, v28);

            __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
            vcmacc_vf(v16, v20, ft0, ft1, v24, v28);
        }
        else {
            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v0, v4, ft0, ft1, v24, v28);

            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            __asm__("add %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
            vcmacc_vf(v8, v12, ft0, ft1, v24, v28);

            __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
            vcmacc_vf(v16, v20, ft0, ft1, v24, v28);
        }
    }

    if (csc == 2 * FLT_SIZE) {
        __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSEG2 "v8, (%0)" : : "r"(c));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSEG2 "v16, (%0)" : : "r"(c));
    }
    else {
        __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
        __asm__("add %0, %0, %1" : "+r"(c) : "r"(rsc));
        __asm__(VSSSEG2 "v16, (%0), %1" : : "r"(c), "r"(csc));
    }

    return;
}

void bli_zgemm_3m4_cleanup(dim_t M, dim_t N, dim_t K, const dcomplex* restrict alpha,
        const dcomplex* restrict a, const dcomplex* restrict b, const dcomplex* restrict beta,
        dcomplex* restrict c, inc_t rsc, inc_t csc) {
    // M x N x K zgemm, 0 < M < 3, 0 < N <= 32 = vlmax, K > 0
    // pairs of register groups hold the real and imag. parts of rows of c and b
    __asm__("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "I"(8 * FLT_SIZE));
    bool first = true;
    // compute a*b
    for (dim_t k = 0; k < K; ++k) {
        // v24 = Re(b), v28 = Im(b)
        __asm__(VLSEG2 "v24, (%0)" : : "r"(b));
        if (first) {
            // ft0 = Re(*a), ft1 = Im(*a)
            switch (M) {
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                vcmul_vf(v8, v12, v24, v28, ft2, ft3);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                vcmul_vf(v0, v4, v24, v28, ft0, ft1);
            }
            first = false;
        }
        else {
            // ft0 = Re(*a), ft1 = Im(*a)
            switch (M) {
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a), "I"(2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a), "I"(3 * FLT_SIZE));
                vcmacc_vf(v8, v12, ft2, ft3, v24, v28);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a), "I"(1 * FLT_SIZE));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
        __asm__("addi %0, %0, %1" : "+r"(a) : "I"(PACKMR * 2 * FLT_SIZE));
        __asm__("addi %0, %0, %1" : "+r"(b) : "I"(PACKNR * 2 * FLT_SIZE));
    }

    __asm__(FLT_LOAD "ft0, (%0)" : : "r"(alpha));
    __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(alpha), "I"(FLT_SIZE));

    // scale a*b by alpha
    switch (M) {
    case 2:
        __asm__("vfmul.vf v24, v12, ft1");
        __asm__("vfmul.vf v28, v8, ft1");
        __asm__("vfmsub.vf v8, ft0, v24");
        __asm__("vfmadd.vf v12, ft0, v28");
    case 1:
        __asm__("vfmul.vf v24, v4, ft1");
        __asm__("vfmul.vf v28, v0, ft1");
        __asm__("vfmsub.vf v0, ft0, v24");
        __asm__("vfmadd.vf v4, ft0, v28");
    }

    c += (M - 1) * rsc;
    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;

    if (beta->real != 0. || beta->imag != 0.) {
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        dcomplex* c_tmp = c;

        if (csc == 2 * FLT_SIZE) {
            switch (M) {
            case 2:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                vcmacc_vf(v8, v12, ft0, ft1, v24, v28);
            case 1:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(c_tmp));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
        else {
            switch (M) {
            case 2:
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c_tmp) : "r"(rsc));
                vcmacc_vf(v8, v12, ft0, ft1, v24, v28);
            case 1:
                __asm__(VLSSEG2 "v24, (%0), %1" : : "r"(c_tmp), "r"(csc));
                vcmacc_vf(v0, v4, ft0, ft1, v24, v28);
            }
        }
    } // end beta != 0. + 0.i

    if (csc == 2 * FLT_SIZE) {
        switch (M) {
        case 2:
            __asm__(VSSEG2 "v8, (%0)" : : "r"(c));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
        }
    }
    else {
        switch (M) {
        case 2:
            __asm__(VSSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
            __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
        case 1:
            __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
        }
    }
    return;
}

void bli_zgemm_3m4_k0(dim_t M, dim_t N, const dcomplex* restrict beta,
        dcomplex* restrict c, inc_t rsc, inc_t csc) {
    // 0 < M <= 3, 0 < N <= 32 = vlmax, K = 0
    // This may not produce the same result as the reference kernel if alpha is infinite or NaN.
    c += (M - 1) * rsc;
    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;
    if (beta->real == 0. && beta->imag == 0.) {
        if (csc == 2 * FLT_SIZE) { // c unit column stride
            // store 2*N 0.'s per row
            __asm__ volatile("vsetvli zero, %0, e%1, m8, ta, ma" : : "r"(2 * N), "i"(8 * FLT_SIZE));
            __asm__("vmv.v.i v0, 0");
            switch (M) {
            case 3:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSE "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSE "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
            __asm__("vmv.v.i v0, 0");
            __asm__("vmv.v.i v4, 0");
            switch (M) {
            case 3:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta == 0. + 0.i
    else { // beta != 0. + 0.i
        __asm__ volatile("vsetvli zero, %0, e%1, m4, ta, ma" : : "r"(N), "i"(8 * FLT_SIZE));
        __asm__(FLT_LOAD "ft0, (%0)" : : "r"(beta));
        __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(beta), "I"(FLT_SIZE));
        if (csc == 2 * FLT_SIZE) { // c unit column stride
            switch (M) {
            case 3:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSEG2 "v8, (%0)" : : "r"(c));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSEG2 "v0, (%0)" : : "r"(c));
            }
        } // end c unit column stride
        else { // c non-unit column stride
            switch (M) {
            case 3:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 2:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
                __asm__("sub %0, %0, %1" : "+r"(c) : "r"(rsc));
            case 1:
                __asm__(VLSSEG2 "v8, (%0), %1" : : "r"(c), "r"(csc));
                vcmul_vf(v0, v4, v8, v12, ft0, ft1);
                __asm__(VSSSEG2 "v0, (%0), %1" : : "r"(c), "r"(csc));
            }
        } // end c non-unit column stride
    } // end beta != 0. + 0.i
    return;
}

void bli_zgemm_sifive_x280_asm_3m4(dim_t M, dim_t N, dim_t K,
        const void* restrict alpha_, const void* restrict a_, const void* restrict b_,
        const void* restrict beta_, void* restrict c_, inc_t rsc, inc_t csc,
        auxinfo_t* restrict data, const cntx_t* restrict cntx) {
    (void) data;
    (void) cntx;
    const dcomplex* restrict alpha = alpha_;
    const dcomplex* restrict beta = beta_;
    const dcomplex* restrict a = a_;
    const dcomplex* restrict b = b_;
    dcomplex* restrict c = c_;

    // M x N x K zgemm, M <= 3, N <= 32
    if (M <= 0 || N <= 0 || K < 0)
        return;
    else if (K == 0)
        bli_zgemm_3m4_k0(M, N, beta, c, rsc, csc);
    else if (M == 3)
        bli_zgemm_3m4(N, K, alpha, a, b, beta, c, rsc, csc);
    else
        bli_zgemm_3m4_cleanup(M, N, K, alpha, a, b, beta, c, rsc, csc);
    return;
}
