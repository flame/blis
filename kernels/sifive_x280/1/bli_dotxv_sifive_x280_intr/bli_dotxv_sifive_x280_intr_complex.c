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
#ifdef DOTXV

DOTXV(PRECISION_CHAR, void)
{
    // Computes rho = beta * rho + alpha * conjxt(x)^T * conjy(y)
    (void) cntx;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict beta = beta_;
    DATATYPE* restrict rho = rho_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict y = y_;
    
    if (beta->real == 0 && beta->imag == 0){
        rho->real = 0;
        rho->imag = 0;
    } else if (!(beta->real == 1 && beta->imag == 0)) {
        DATATYPE temp = *rho;
        rho->real =  rho->real * beta->real - rho->imag * beta->imag;
        rho->imag =  temp.real * beta->imag + rho->imag * beta->real;
    }

    if (n <= 0 || (alpha->real == 0 && alpha->imag == 0))
        return;

    // Instead of conjugating x, switch conjugation on y
    //  and conjugate dot product at the end
    conj_t conjsum = conjxt;
    if (conjxt == BLIS_CONJUGATE)
        bli_toggle_conj(&conjy); // Switch conjugation of y

    // Compute dot product
    RVV_TYPE_F(PREC, LMUL) acc_real, acc_imag;
    size_t avl = n;
    bool first = true;
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
        
        if (first) {
            acc_real = VFMUL_VV(PREC, LMUL)(xvec_real, yvec_real, vl);
            acc_imag = VFMUL_VV(PREC, LMUL)(xvec_imag, yvec_real, vl);
            first = false;
        } else {
            acc_real = VFMACC_VV_TU(PREC, LMUL)(acc_real, xvec_real, yvec_real, vl);
            acc_imag = VFMACC_VV_TU(PREC, LMUL)(acc_imag, xvec_imag, yvec_real, vl);
        }
        if (conjy == BLIS_NO_CONJUGATE) {
            acc_real = VFNMSAC_VV_TU(PREC, LMUL)(acc_real, xvec_imag, yvec_imag, vl);
            acc_imag = VFMACC_VV_TU(PREC, LMUL)( acc_imag, xvec_real, yvec_imag, vl);
        } else {
            acc_real = VFMACC_VV_TU(PREC, LMUL)( acc_real, xvec_imag, yvec_imag, vl);
            acc_imag = VFNMSAC_VV_TU(PREC, LMUL)(acc_imag, xvec_real, yvec_imag, vl);
        }

        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }


    RVV_TYPE_F(PREC, m1) sum_real = VFMV_S_F(PREC, m1)(0.f, 1);
    RVV_TYPE_F(PREC, m1) sum_imag = VFMV_S_F(PREC, m1)(0.f, 1);
    sum_real = VF_REDUSUM_VS(PREC, LMUL)(acc_real, sum_real, n);
    sum_imag = VF_REDUSUM_VS(PREC, LMUL)(acc_imag, sum_imag, n);

    if (conjsum == BLIS_CONJUGATE) {
        sum_imag = VFNEG_VF(PREC, m1)(sum_imag, 1);
    }
    DATATYPE dot = {VFMV_F_S(PREC)(sum_real), VFMV_F_S(PREC)(sum_imag)};

    // Accumulate alpha * dot
    rho->real = fma( alpha->real, dot.real, rho->real);
    rho->real = fma(-alpha->imag, dot.imag, rho->real);
    rho->imag = fma( alpha->imag, dot.real, rho->imag);
    rho->imag = fma( alpha->real, dot.imag, rho->imag);

}

#endif // DOTXV
