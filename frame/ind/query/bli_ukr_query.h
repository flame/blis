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
      derived from this software without specific prior written permission.

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

#ifndef BLIS_UKR_QUERY_H
#define BLIS_UKR_QUERY_H


typedef enum
{
    BLIS_GEMM_UKR = 0,
    BLIS_GEMMTRSM_L_UKR,
    BLIS_GEMMTRSM_U_UKR,
    BLIS_TRSM_L_UKR,
    BLIS_TRSM_U_UKR,
} l3ukr_t;

#define BLIS_NUM_LEVEL3_UKRS 5

typedef enum
{
	BLIS_REFERENCE_UKERNEL = 0,
	BLIS_VIRTUAL_UKERNEL,
	BLIS_OPTIMIZED_UKERNEL,
	BLIS_NOTAPPLIC_UKERNEL,
} kimpl_t;

#define BLIS_NUM_UKR_IMPL_TYPES 4

// -----------------------------------------------------------------------------

char*   bli_ukr_impl_string( l3ukr_t ukr, ind_t method, num_t dt );
char*   bli_ukr_avail_impl_string( l3ukr_t ukr, num_t dt );
kimpl_t bli_ukr_impl_type( l3ukr_t ukr, ind_t method, num_t dt );

func_t* bli_ukr_get_funcs( l3ukr_t ukr, ind_t method );
func_t* bli_ukr_get_ref_funcs( l3ukr_t ukr );


#endif

