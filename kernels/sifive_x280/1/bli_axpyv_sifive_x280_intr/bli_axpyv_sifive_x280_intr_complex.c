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
#ifdef AXPYV

AXPYV(PRECISION_CHAR, void)
{
    // Computes y := y + alpha * conjx(x)
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict x = x_;
    DATATYPE* restrict y = y_;
    
    if (n <= 0) return;
    if (alpha->real == 0 && alpha->imag == 0) return;

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
        
        yvec_real = VFMACC_VF(PREC, LMUL)( yvec_real, alpha->real, xvec_real, vl);
        yvec_imag = VFMACC_VF(PREC, LMUL)( yvec_imag, alpha->imag, xvec_real, vl);
        if (conjx == BLIS_NO_CONJUGATE){
            yvec_real = VFNMSAC_VF(PREC, LMUL)(yvec_real, alpha->imag, xvec_imag, vl);
            yvec_imag = VFMACC_VF(PREC, LMUL) (yvec_imag, alpha->real, xvec_imag, vl);
        } else {
            yvec_real = VFMACC_VF(PREC, LMUL) (yvec_real, alpha->imag, xvec_imag, vl);
            yvec_imag = VFNMSAC_VF(PREC, LMUL)(yvec_imag, alpha->real, xvec_imag, vl);
        }

        if (incy == 1)
            VSSEG2_V_F(PREC, LMUL)( (BASE_DT*) y, yvec_real, yvec_imag, vl);
        else
            VSSSEG2_V_F(PREC, LMUL)((BASE_DT*) y, 2*FLT_SIZE*incy, yvec_real, yvec_imag, vl);
        
        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }

}

#endif // AXPYV
