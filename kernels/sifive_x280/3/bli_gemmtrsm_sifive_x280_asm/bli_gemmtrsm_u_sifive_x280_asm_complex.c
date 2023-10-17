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
#ifdef GEMMTRSM

GEMMTRSM(GEMMTRSM_U, PRECISION_CHAR, void)
{
    (void) data;
    (void) cntx;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a12 = a12_;
    const DATATYPE* restrict a11 = a11_;
    const DATATYPE* restrict b21 = b21_;
    const DATATYPE* restrict b11 = b11_;
    DATATYPE* restrict c11 = c11_;

    if (m <= 0 || n <= 0)
        return;

    __asm__ volatile("vsetvli zero, %0, e%1, m2, ta, ma" : : "r"(n), "i"(8 * FLT_SIZE));

    DATATYPE alpha_cast = *alpha;
    if (alpha_cast.real == 0 && alpha_cast.imag == 0) {
        switch (m) {
            case 6:
                __asm__("vmv.v.i v20, 0");
                __asm__("vmv.v.i v22, 0");
            case 5:
                __asm__("vmv.v.i v16, 0");
                __asm__("vmv.v.i v18, 0");
            case 4:
                __asm__("vmv.v.i v12, 0");
                __asm__("vmv.v.i v14, 0");
            case 3:
                __asm__("vmv.v.i v8, 0");
                __asm__("vmv.v.i v10, 0");
            case 2:
                __asm__("vmv.v.i v4, 0");
                __asm__("vmv.v.i v6, 0");
            case 1:
                __asm__("vmv.v.i v0, 0");
                __asm__("vmv.v.i v2, 0");
        }
    }
    else {
        const DATATYPE* b11_tmp = b11;
        switch (m) {
            case 6:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v20, v22, v24, v26, alpha_cast.real, alpha_cast.imag);
                __asm__("addi %0, %0, %1" : "+r"(b11_tmp) : "I"(PACKNR * 2 * FLT_SIZE));
            case 5:
                __asm__(VLSEG2 "v28, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v16, v18, v28, v30, alpha_cast.real, alpha_cast.imag);
                __asm__("addi %0, %0, %1" : "+r"(b11_tmp) : "I"(PACKNR * 2 * FLT_SIZE));
            case 4:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v12, v14, v24, v26, alpha_cast.real, alpha_cast.imag);
                __asm__("addi %0, %0, %1" : "+r"(b11_tmp) : "I"(PACKNR * 2 * FLT_SIZE));
            case 3:
                __asm__(VLSEG2 "v28, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v8, v10, v28, v30, alpha_cast.real, alpha_cast.imag);
                __asm__("addi %0, %0, %1" : "+r"(b11_tmp) : "I"(PACKNR * 2 * FLT_SIZE));
            case 2:
                __asm__(VLSEG2 "v24, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v4, v6, v24, v26, alpha_cast.real, alpha_cast.imag);
                __asm__("addi %0, %0, %1" : "+r"(b11_tmp) : "I"(PACKNR * 2 * FLT_SIZE));
            case 1:
                __asm__(VLSEG2 "v28, (%0)" : : "r"(b11_tmp));
                vcmul_vf2(v0, v2, v28, v30, alpha_cast.real, alpha_cast.imag);
        }
    }

    if (k >= 1) {
        __asm__(VLSEG2 "v24, (%0)" : : "r"(b21));
        __asm__("addi %0, %0, %1" : "+r"(b21) : "I"(PACKNR * 2 * FLT_SIZE));
    }
    if (k >= 2) {
        __asm__(VLSEG2 "v28, (%0)" : : "r"(b21));
        __asm__("addi %0, %0, %1" : "+r"(b21) : "I"(PACKNR * 2 * FLT_SIZE));
    }

    a12 += m - 1;

    while (k > 0) {
        switch (m) {
            case 6:
                __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a12), "I"(-10 * FLT_SIZE));
                __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a12), "I"(-9 * FLT_SIZE));
                vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);
            case 5:
                __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a12), "I"(-8 * FLT_SIZE));
                __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a12), "I"(-7 * FLT_SIZE));
                vcnmsac_vf(v16, v18, ft8, ft9, v24, v26);
            case 4:
                __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a12), "I"(-6 * FLT_SIZE));
                __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a12), "I"(-5 * FLT_SIZE));
                vcnmsac_vf(v12, v14, ft6, ft7, v24, v26);
            case 3:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a12), "I"(-4 * FLT_SIZE));
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a12), "I"(-3 * FLT_SIZE));
                vcnmsac_vf(v8, v10, ft4, ft5, v24, v26);
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a12), "I"(-2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a12), "I"(-1 * FLT_SIZE));
                vcnmsac_vf(v4, v6, ft2, ft3, v24, v26);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a12), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a12), "I"(1 * FLT_SIZE));
                vcnmsac_vf(v0, v2, ft0, ft1, v24, v26);
        }
        k -= 1;

        if (k == 0) { break; }

        if (k >= 2) {
            __asm__(VLSEG2 "v24, (%0)" : : "r"(b21));
            __asm__("addi %0, %0, %1" : "+r"(b21) : "I"(PACKNR * 2 * FLT_SIZE));
        }
        __asm__("addi %0, %0, %1" : "+r"(a12) : "I"(PACKMR * 2 * FLT_SIZE));

        switch (m) {
            case 6:
                __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a12), "I"(-10 * FLT_SIZE));
                __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a12), "I"(-9 * FLT_SIZE));
                vcnmsac_vf(v20, v22, ft10, ft11, v28, v30);
            case 5:
                __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a12), "I"(-8 * FLT_SIZE));
                __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a12), "I"(-7 * FLT_SIZE));
                vcnmsac_vf(v16, v18, ft8, ft9, v28, v30);
            case 4:
                __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a12), "I"(-6 * FLT_SIZE));
                __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a12), "I"(-5 * FLT_SIZE));
                vcnmsac_vf(v12, v14, ft6, ft7, v28, v30);
            case 3:
                __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a12), "I"(-4 * FLT_SIZE));
                __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a12), "I"(-3 * FLT_SIZE));
                vcnmsac_vf(v8, v10, ft4, ft5, v28, v30);
            case 2:
                __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a12), "I"(-2 * FLT_SIZE));
                __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a12), "I"(-1 * FLT_SIZE));
                vcnmsac_vf(v4, v6, ft2, ft3, v28, v30);
            case 1:
                __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a12), "I"(0 * FLT_SIZE));
                __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a12), "I"(1 * FLT_SIZE));
                vcnmsac_vf(v0, v2, ft0, ft1, v28, v30);
        }
        k -= 1;

        if (k >= 2) {
            __asm__(VLSEG2 "v28, (%0)" : : "r"(b21));
            __asm__("addi %0, %0, %1" : "+r"(b21) : "I"(PACKNR * 2 * FLT_SIZE));
        }
        __asm__("addi %0, %0, %1" : "+r"(a12) : "I"(PACKMR * 2 * FLT_SIZE));
    }

    a11 += (m - 1) * (PACKMR + 1); // (m - 1) + (m - 1) * PACKMR
    b11 += (m - 1) * PACKNR;
    c11 += (m - 1) * rsc;
    rsc *= 2 * FLT_SIZE;
    csc *= 2 * FLT_SIZE;

    __asm__(FLT_LOAD "ft0, %1(%0)" : : "r"(a11), "I"(0 * FLT_SIZE));
    __asm__(FLT_LOAD "ft1, %1(%0)" : : "r"(a11), "I"(1 * FLT_SIZE));
    vcmul_vf(v24, v26, v0, v2, ft0, ft1);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    if (m == 1) return;

    switch (m) {
        case 6:
            __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
            __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
            vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);
        case 5:
            __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a11), "I"(-8 * FLT_SIZE));
            __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a11), "I"(-7 * FLT_SIZE));
            vcnmsac_vf(v16, v18, ft8, ft9, v24, v26);
        case 4:
            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a11), "I"(-6 * FLT_SIZE));
            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a11), "I"(-5 * FLT_SIZE));
            vcnmsac_vf(v12, v14, ft6, ft7, v24, v26);
        case 3:
            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a11), "I"(-4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a11), "I"(-3 * FLT_SIZE));
            vcnmsac_vf(v8, v10, ft4, ft5, v24, v26);
        case 2:
            __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a11), "I"(-2 * FLT_SIZE));
            __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a11), "I"(-1 * FLT_SIZE));
            vcnmsac_vf(v4, v6, ft2, ft3, v24, v26);
    }
    __asm__("addi %0, %0, %1" : "+r"(a11) : "I"(-PACKMR * 2 * FLT_SIZE));
    __asm__("addi %0, %0, %1" : "+r"(b11) : "I"(-PACKNR * 2 * FLT_SIZE));
    __asm__("sub %0, %0, %1" : "+r"(c11) : "r"(rsc));

    __asm__(FLT_LOAD "ft2, %1(%0)" : : "r"(a11), "I"(-2 * FLT_SIZE));
    __asm__(FLT_LOAD "ft3, %1(%0)" : : "r"(a11), "I"(-1 * FLT_SIZE));
    vcmul_vf(v24, v26, v4, v6, ft2, ft3);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    if (m == 2) return;

    switch (m) {
        case 6:
            __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
            __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
            vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);
        case 5:
            __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a11), "I"(-8 * FLT_SIZE));
            __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a11), "I"(-7 * FLT_SIZE));
            vcnmsac_vf(v16, v18, ft8, ft9, v24, v26);
        case 4:
            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a11), "I"(-6 * FLT_SIZE));
            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a11), "I"(-5 * FLT_SIZE));
            vcnmsac_vf(v12, v14, ft6, ft7, v24, v26);
        case 3:
            __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a11), "I"(-4 * FLT_SIZE));
            __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a11), "I"(-3 * FLT_SIZE));
            vcnmsac_vf(v8, v10, ft4, ft5, v24, v26);
    }
    __asm__("addi %0, %0, %1" : "+r"(a11) : "I"(-PACKMR * 2 * FLT_SIZE));
    __asm__("addi %0, %0, %1" : "+r"(b11) : "I"(-PACKNR * 2 * FLT_SIZE));
    __asm__("sub %0, %0, %1" : "+r"(c11) : "r"(rsc));

    __asm__(FLT_LOAD "ft4, %1(%0)" : : "r"(a11), "I"(-4 * FLT_SIZE));
    __asm__(FLT_LOAD "ft5, %1(%0)" : : "r"(a11), "I"(-3 * FLT_SIZE));
    vcmul_vf(v24, v26, v8, v10, ft4, ft5);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    if (m == 3) return;

    switch (m) {
        case 6:
            __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
            __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
            vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);
        case 5:
            __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a11), "I"(-8 * FLT_SIZE));
            __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a11), "I"(-7 * FLT_SIZE));
            vcnmsac_vf(v16, v18, ft8, ft9, v24, v26);
        case 4:
            __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a11), "I"(-6 * FLT_SIZE));
            __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a11), "I"(-5 * FLT_SIZE));
            vcnmsac_vf(v12, v14, ft6, ft7, v24, v26);
    }
    __asm__("addi %0, %0, %1" : "+r"(a11) : "I"(-PACKMR * 2 * FLT_SIZE));
    __asm__("addi %0, %0, %1" : "+r"(b11) : "I"(-PACKNR * 2 * FLT_SIZE));
    __asm__("sub %0, %0, %1" : "+r"(c11) : "r"(rsc));

    __asm__(FLT_LOAD "ft6, %1(%0)" : : "r"(a11), "I"(-6 * FLT_SIZE));
    __asm__(FLT_LOAD "ft7, %1(%0)" : : "r"(a11), "I"(-5 * FLT_SIZE));
    vcmul_vf(v24, v26, v12, v14, ft6, ft7);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    if (m == 4) return;

    switch (m) {
        case 6:
            __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
            __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
            vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);
        case 5:
            __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a11), "I"(-8 * FLT_SIZE));
            __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a11), "I"(-7 * FLT_SIZE));
            vcnmsac_vf(v16, v18, ft8, ft9, v24, v26);
    }
    __asm__("addi %0, %0, %1" : "+r"(a11) : "I"(-PACKMR * 2 * FLT_SIZE));
    __asm__("addi %0, %0, %1" : "+r"(b11) : "I"(-PACKNR * 2 * FLT_SIZE));
    __asm__("sub %0, %0, %1" : "+r"(c11) : "r"(rsc));

    __asm__(FLT_LOAD "ft8, %1(%0)" : : "r"(a11), "I"(-8 * FLT_SIZE));
    __asm__(FLT_LOAD "ft9, %1(%0)" : : "r"(a11), "I"(-7 * FLT_SIZE));
    vcmul_vf(v24, v26, v16, v18, ft8, ft9);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    if (m == 5) return;

    __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
    vcnmsac_vf(v20, v22, ft10, ft11, v24, v26);

    __asm__("addi %0, %0, %1" : "+r"(a11) : "I"(-PACKMR * 2 * FLT_SIZE));
    __asm__("addi %0, %0, %1" : "+r"(b11) : "I"(-PACKNR * 2 * FLT_SIZE));
    __asm__("sub %0, %0, %1" : "+r"(c11) : "r"(rsc));

    __asm__(FLT_LOAD "ft10, %1(%0)" : : "r"(a11), "I"(-10 * FLT_SIZE));
    __asm__(FLT_LOAD "ft11, %1(%0)" : : "r"(a11), "I"(-9 * FLT_SIZE));
    vcmul_vf(v24, v26, v20, v22, ft10, ft11);
    __asm__(VSSEG2 "v24, (%0)" : : "r"(b11));
    __asm__(VSSSEG2 "v24, (%0), %1" : : "r"(c11), "r"(csc));

    return;
}
#endif
