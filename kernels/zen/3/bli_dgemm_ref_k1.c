/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include "blis.h"

#define C(i,j) *(c+j*ldc+i)
#define A(i,j) *(a+j*lda+i)
#define B(i,j) *(b+j*ldb+i)


void bli_dgemm_ref_k1_nn
     (
	dim_t  m,
	dim_t  n,
	dim_t  k,
	double*    alpha,
	double*    a, const inc_t lda,
	double*    b, const inc_t ldb,
	double*    beta,
	double*    c, const inc_t ldc
       )
{

	double alpha_val, beta_val, temp;
	dim_t i, j, K;

	beta_val = *beta;
	alpha_val = *alpha;

	if((m == 0) || (n == 0) || (((alpha_val == 0.0) || (k == 0)) && (beta_val == 1.0))){
		return;
	}

	/* If alpha = 0 */

	if(alpha_val == 0.0)
	{
		if(beta_val == 0.0)
		{
			for(j = 0; j < n; j++){
				for(i = 0; i < m; i++){
					C(i,j) = 0.0;
				}
			}
		}
		else
		{
			for(j = 0; j < n; j++){
				for(i = 0; i < m; i++){
					C(i,j) = beta_val * C(i,j);
				}
			}
		}
		return;
	}


	/* Start the operation */


	/* Form C = alpha*A*B + beta*c */
	if(beta_val == 0.0){
		for(j =0; j < n; j++){
			for(i = 0; i < m; i++){
				C(i,j) = 0.0;
			}
		}
	}
	else if(beta_val != 1.0){
		for(j = 0 ; j < n; j++){
			for(i = 0; i < m; i++){
				C(i,j) = beta_val * C(i,j);
			}
		}
	}

	for(j = 0; j < n; j++){
		for(K = 0; K < k; K++)
		{
			temp = alpha_val * B(K,j);
			for(i = 0; i < m; i++){
				C(i,j) = C(i,j) + temp*A(i,K);
			}
		}
	}
	return;
}
