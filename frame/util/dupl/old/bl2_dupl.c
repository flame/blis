/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"

#define NDUP_S         BLIS_DEFAULT_NUM_DUPL_S
#define NDUP_D         BLIS_DEFAULT_NUM_DUPL_D
#define NDUP_C         BLIS_DEFAULT_NUM_DUPL_C
#define NDUP_Z         BLIS_DEFAULT_NUM_DUPL_Z

#define UNROLL_FAC_S   1
#define UNROLL_FAC_D   8
#define UNROLL_FAC_C   1
#define UNROLL_FAC_Z   1

void bl2_sdupl(
                dim_t     n_elem,
                float*    b,
                float*    bd
              )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bl2_ddupl(
                dim_t     n_elem,
                double*   b,
                double*   bd
              )
{
	dim_t         n_iter  = n_elem / UNROLL_FAC_D;
	dim_t         n_left  = n_elem % UNROLL_FAC_D;

	const inc_t   rstep_b = UNROLL_FAC_D;
	const inc_t   step_bd = UNROLL_FAC_D * NDUP_D;

	dim_t         i;

	for ( i = 0; i < n_iter; ++i )
	{
		*(bd +  0) = *(b + 0);
		*(bd +  1) = *(b + 0);

		*(bd +  2) = *(b + 1);
		*(bd +  3) = *(b + 1);

		*(bd +  4) = *(b + 2);
		*(bd +  5) = *(b + 2);

		*(bd +  6) = *(b + 3);
		*(bd +  7) = *(b + 3);

		*(bd +  8) = *(b + 4);
		*(bd +  9) = *(b + 4);

		*(bd + 10) = *(b + 5);
		*(bd + 11) = *(b + 5);

		*(bd + 12) = *(b + 6);
		*(bd + 13) = *(b + 6);

		*(bd + 14) = *(b + 7);
		*(bd + 15) = *(b + 7);

		b  += rstep_b;
		bd += step_bd;
	}

	for ( i = 0; i < n_left; ++i )
	{
		*(bd +  0) = *(b + 0);
		*(bd +  1) = *(b + 0);

		b  += 1;
		bd += NDUP;
	}
}

void bl2_cdupl(
                dim_t     n_elem,
                scomplex* b,
                scomplex* bd
              )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bl2_zdupl(
                dim_t     n_elem,
                dcomplex* b,
                dcomplex* bd
              )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

