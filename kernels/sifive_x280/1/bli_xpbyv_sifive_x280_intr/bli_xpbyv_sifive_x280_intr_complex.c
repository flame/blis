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
#ifdef XPBYV

XPBYV(PRECISION_CHAR, void)
{
    // Computes y = beta * y + conjx(x)
    const DATATYPE* restrict beta = beta_;
    const DATATYPE* restrict x = x_;
    DATATYPE* restrict y = y_;
    
    if (n <= 0) return;

    if (beta->real == 0 && beta->imag == 0){
        COPYV(PRECISION_CHAR)(conjx, n, x, incx, y, incy, cntx);
        return;
    }

    // TO DO (optimization): beta = +-1, +-i special cases

    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_F(PREC, LMUL) xvec_real, xvec_imag, yvec_real, yvec_imag;

        if (incx == 1)
            VLSEG2_V_F(PREC, LMUL)( &xvec_real, &xvec_imag, (BASE_DT*) x, vl);
        else
            VLSSEG2_V_F(PREC, LMUL)(&xvec_real, &xvec_imag, (BASE_DT*) x, 2*FLT_SIZE*incx, vl);
        
        if (incy == 1)
            VLSEG2_V_F(PREC, LMUL)( &yvec_real, &yvec_imag, (BASE_DT*) y, vl);
        else
            VLSSEG2_V_F(PREC, LMUL)(&yvec_real, &yvec_imag, (BASE_DT*) y, 2*FLT_SIZE*incy, vl);
        
        // xpbyv is computed with FMAs as follows:
        // y[i].real = (      x[i].real + beta.real * y[i].real) - beta.imag * y[i].imag
        // y[i].imag = (conjx(x[i].imag + beta.imag * y[i].real) + beta.real * y[i].imag

        xvec_real = VFMACC_VF( PREC, LMUL)(xvec_real, beta->real, yvec_real, vl);
        xvec_real = VFNMSAC_VF(PREC, LMUL)(xvec_real, beta->imag, yvec_imag, vl);
        if (conjx == BLIS_NO_CONJUGATE)
            xvec_imag = VFMACC_VF(PREC, LMUL)(xvec_imag, beta->imag, yvec_real, vl);
        else
            xvec_imag = VFMSAC_VF(PREC, LMUL)(xvec_imag, beta->imag, yvec_real, vl);
        xvec_imag = VFMACC_VF(PREC, LMUL)(xvec_imag, beta->real, yvec_imag, vl);

        if (incy == 1)
            VSSEG2_V_F(PREC, LMUL)( (BASE_DT*) y, xvec_real, xvec_imag, vl);
        else
            VSSSEG2_V_F(PREC, LMUL)((BASE_DT*) y, 2*FLT_SIZE*incy, xvec_real, xvec_imag, vl);
        
        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }
}

#endif // XPBYV
