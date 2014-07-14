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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

extern void bli_sgemm_kernel_4x4(dim_t              k,
                        float* alpha,
                        float*   restrict a,
                        float*   restrict b,
                        float* beta,
                        float*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      );


void bli_sgemm_opt_4x4(
                        dim_t              k,
                        float*    restrict alpha,
                        float*    restrict a,
                        float*    restrict b,
                        float*    restrict beta,
                        float*    restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{

	bli_sgemm_kernel_4x4(k, alpha, a, b, beta, c, rs_c, cs_c, data);

}

extern void bli_dgemm_kernel_4x4(dim_t              k,
                        double* alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double* beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      );


void bli_dgemm_opt_4x4(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double*   restrict beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	bli_dgemm_kernel_4x4(k, alpha, a, b, beta, c, rs_c, cs_c, data);
}

extern void bli_cgemm_kernel_2x2(dim_t              k,
                        scomplex* alpha,
                        scomplex*   restrict a,
                        scomplex*   restrict b,
                        scomplex* beta,
                        scomplex*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      );




void bli_cgemm_opt_4x4(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a,
                        scomplex* restrict b,
                        scomplex* restrict beta,
                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{

	bli_cgemm_kernel_2x2(k, alpha, a, b, beta, c, rs_c, cs_c, data);
}

extern void bli_zgemm_kernel_2x2(dim_t              k,
                        dcomplex* alpha,
                        dcomplex*   restrict a,
                        dcomplex*   restrict b,
                        dcomplex* beta,
                        dcomplex*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      );


void bli_zgemm_opt_4x4(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a,
                        dcomplex* restrict b,
                        dcomplex* restrict beta,
                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{

	bli_zgemm_kernel_2x2(k, alpha, a, b, beta, c, rs_c, cs_c, data);
}

