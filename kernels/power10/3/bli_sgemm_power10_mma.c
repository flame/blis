/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "vector_int_macros.h"

#define S_ACCUMULATE \
        __builtin_mma_xvf32gerpp (&acc0, ca[0], rb[0]); \
        __builtin_mma_xvf32gerpp (&acc1, ca[0], rb[1]); \
        __builtin_mma_xvf32gerpp (&acc2, ca[0], rb[2]); \
        __builtin_mma_xvf32gerpp (&acc3, ca[0], rb[3]); \
        __builtin_mma_xvf32gerpp (&acc4, ca[1], rb[0]); \
        __builtin_mma_xvf32gerpp (&acc5, ca[1], rb[1]); \
        __builtin_mma_xvf32gerpp (&acc6, ca[1], rb[2]); \
        __builtin_mma_xvf32gerpp (&acc7, ca[1], rb[3]); 

#define S_INCREMENT \
        A0+=8; \
        B0+=16;

#define S_AB_PRODUCT \
        LOAD_VECTORS \
        S_INCREMENT \
        S_ACCUMULATE 

void bli_sgemm_power10_mma_8x16
    (
        dim_t               k0,
        float*     restrict alpha,
        float*     restrict a,
        float*     restrict b,
        float*     restrict beta,
        float*     restrict c, inc_t rs_c0, inc_t cs_c0,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
    )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    // (1 is subtracted from k0 because 1 iteration of the k loop is pulled out)
    uint64_t k_iter = (k0-1) / 4;
    uint64_t k_left = (k0-1) % 4;
    
    uint64_t rs_c   = rs_c0;

    fv4sf_t result[4];
      fv4sf_t *rowC;

    // accumulators that will hold the matrix product
    __vector_quad acc0, acc1, acc2, acc3, 
                  acc4, acc5, acc6, acc7;

    float* restrict A0 = a;
    float* restrict B0 = b;
    float* restrict C0 = c;

    float alpha_ = *alpha,
          beta_  = *beta;

    /* Load elements into vector registers */
    vec_t *ca = (vec_t *) A0;
    vec_t *rb = (vec_t *) B0;

    /* Compute accumulate outer products and override accumulators with result */
    __builtin_mma_xvf32ger (&acc0, ca[0], rb[0]);
    __builtin_mma_xvf32ger (&acc1, ca[0], rb[1]);
    __builtin_mma_xvf32ger (&acc2, ca[0], rb[2]);
    __builtin_mma_xvf32ger (&acc3, ca[0], rb[3]);
    __builtin_mma_xvf32ger (&acc4, ca[1], rb[0]);
    __builtin_mma_xvf32ger (&acc5, ca[1], rb[1]);
    __builtin_mma_xvf32ger (&acc6, ca[1], rb[2]);
    __builtin_mma_xvf32ger (&acc7, ca[1], rb[3]);

    S_INCREMENT

    // k loop (unrolled by 4)
    for (int k = 0; k<k_iter; k++)
    {
        S_AB_PRODUCT
        S_AB_PRODUCT
        S_AB_PRODUCT
        S_AB_PRODUCT
    }
    
    // edge loop
    for (int k = 0; k<k_left; k++)
    {
        S_AB_PRODUCT
    }

    // handle beta cases
    if (beta_ != 0.0)
    {
        SAVE_ACC(fv4sf_t, &acc0, rs_c, 0      );
        SAVE_ACC(fv4sf_t, &acc1, rs_c, 4      );
        SAVE_ACC(fv4sf_t, &acc2, rs_c, 8      );
        SAVE_ACC(fv4sf_t, &acc3, rs_c, 12     );
        SAVE_ACC(fv4sf_t, &acc4, rs_c,    4*rs_c);
        SAVE_ACC(fv4sf_t, &acc5, rs_c,  4+4*rs_c);
        SAVE_ACC(fv4sf_t, &acc6, rs_c,  8+4*rs_c);
        SAVE_ACC(fv4sf_t, &acc7, rs_c, 12+4*rs_c);
    }
    else
    {
        SAVE_ACC_bz(fv4sf_t, &acc0, rs_c,  0     );
        SAVE_ACC_bz(fv4sf_t, &acc1, rs_c,  4     );
        SAVE_ACC_bz(fv4sf_t, &acc2, rs_c,  8     );
        SAVE_ACC_bz(fv4sf_t, &acc3, rs_c, 12     );
        SAVE_ACC_bz(fv4sf_t, &acc4, rs_c,    4*rs_c);
        SAVE_ACC_bz(fv4sf_t, &acc5, rs_c,  4+4*rs_c);
        SAVE_ACC_bz(fv4sf_t, &acc6, rs_c,  8+4*rs_c);
        SAVE_ACC_bz(fv4sf_t, &acc7, rs_c, 12+4*rs_c);
    }
}
