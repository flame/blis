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
#ifdef AXPBYV

AXPBYV(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjx(x)
    //            == beta * y + alpha *    x        (real case)
    (void) conjx; // Suppress unused parameter warnings
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict beta = beta_;
    const DATATYPE* restrict x = x_;
    DATATYPE* restrict y = y_;

    if (n <= 0) return;
    
    if (*alpha == 0 && *beta == 0){
        SETV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, alpha, y, incy, cntx);
        return;
    }
    if (*alpha == 0){
        SCALV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, beta, y, incy, cntx);
        return;
    }
    if (*beta == 0){
        SCAL2V(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, alpha, x, incx, y, incy, cntx);
        return;
    }

    // Note: in the cases alpha = 0 && beta = 1, or alpha = 1 && beta = 0, we 
    // will canonicalize NaNs whereas the reference code will propagate NaN payloads.

    // TO DO (optimization): special cases for alpha = +-1, beta = +-1

    // alpha and beta are both nonzero
    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_F(PREC, LMUL) xvec, yvec;

        if (incx == 1)
            xvec = VLE_V_F(PREC, LMUL)(x, vl);
        else
            xvec = VLSE_V_F(PREC, LMUL)(x, FLT_SIZE * incx, vl);
        
        if (incy == 1)
            yvec = VLE_V_F(PREC, LMUL)(y, vl);
        else
            yvec = VLSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, vl);
        
        yvec = VFMUL_VF(PREC, LMUL) (yvec, *beta, vl);
        yvec = VFMACC_VF(PREC, LMUL)(yvec, *alpha, xvec, vl);
        
        if (incy == 1)
            VSE_V_F(PREC, LMUL)(y, yvec, vl);
        else
            VSSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, yvec, vl);
        
        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }
}

#endif // AXPYBV
