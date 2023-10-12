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
#ifdef SCALV

SCALV(PRECISION_CHAR, void)
{
    // Computes x = conjalpha(alpha) * x
    //           ==           alpha  * x       (real case)
    
    (void) conjalpha; // Suppress unused parameter warnings
    const DATATYPE* restrict alpha = alpha_;
    DATATYPE* restrict x = x_;

    if (n <= 0 || *alpha == 1) return;

    if (*alpha == 0){
        SETV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, alpha, x, incx, cntx);
        return;
    }

    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_F(PREC, LMUL) xvec;

        if (incx == 1)
            xvec = VLE_V_F(PREC, LMUL) (x, vl);
        else
            xvec = VLSE_V_F(PREC, LMUL)(x, FLT_SIZE * incx, vl);

        xvec = VFMUL_VF(PREC, LMUL)(xvec, *alpha, vl);

        if (incx == 1)
            VSE_V_F(PREC, LMUL) (x, xvec, vl);
        else
            VSSE_V_F(PREC, LMUL)(x, FLT_SIZE * incx, xvec, vl);

        x += vl * incx;
        avl -= vl;
    }
}

#endif // SCALV
