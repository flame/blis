/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include <x86intrin.h>
#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"


void bli_dpackm_zen4_asm_16xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       double*    restrict kappa,
       double*    restrict a, inc_t inca0, inc_t lda0,
       double*    restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
#if 0
	bli_dpackm_16xk_zen4_ref
	(
	  conja, schema, cdim0, k0, k0_max,
	  kappa, a, inca0, lda0, p, ldp0, cntx
	);
	return;
#endif

	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mnr   = 16;

	// This is the "packing" dimension assumed by the packm kernel.
	// This should be equal to ldp.
	//const dim_t    packmnr = 16;

	// NOTE: For the purposes of the comments in this packm kernel, we
	// interpret inca and lda as rs_a and cs_a, respectively, and similarly
	// interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
	// this packm kernel, you should think of the operation as packing an
	// m x n micropanel, where m and n are tiny and large, respectively, and
	// where elements of each column of the packed matrix P are contiguous.
	// (This packm kernel can still be used to pack micropanels of matrix B
	// in a gemm operation.)
	const uint64_t inca   = inca0;
	const uint64_t lda    = lda0;
	const uint64_t ldp    = ldp0;

	// NOTE: If/when this kernel ever supports scaling by kappa within the
	// assembly region, this constraint should be lifted.
	const bool     unitk  = bli_deq1( *kappa );


	// -------------------------------------------------------------------------

	if ( cdim0 == mnr )
	{
		if ( unitk )
		{
			if ( bli_is_conj( conja ) )
			{
				if ( inca == 1 )
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dcopyjs( *(a + i), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
				else
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dcopyjs( *(a + i*inca), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
			}
			else
			{
				if ( inca == 1 )
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						_mm_prefetch( a + (8*lda), _MM_HINT_T0 );
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dcopys( *(a + i), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
				else
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dcopys( *(a + i*inca), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
			}
		}
		else
		{
			if ( bli_is_conj( conja ) )
			{
				if ( inca == 1 )
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dscal2js( *kappa, *(a + i), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
				else
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dscal2js( *kappa, *(a + i*inca), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
			}
			else
			{
				if ( inca == 1 )
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dscal2s( *kappa, *(a + i), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
				else
				{
					for ( dim_t k = k0; k != 0; --k )
					{
						for ( dim_t i = 0 ; i < 16 ; i++ ) {
							bli_dscal2s( *kappa, *(a + i*inca), *(p + i) );
						}
						a += lda;
						p    += ldp;
					}
				}
			}
		}
	}
	else // if ( cdim0 < mnr )
	{
		PASTEMAC(dscal2m,BLIS_TAPI_EX_SUF)
		(
		  0,
		  BLIS_NONUNIT_DIAG,
		  BLIS_DENSE,
		  ( trans_t )conja,
		  cdim0,
		  k0,
		  kappa,
		  a, inca0, lda0,
		  p,     1, ldp0,
		  cntx,
		  NULL
		);

		if ( cdim0 < mnr )
		{
			// Handle zero-filling along the "long" edge of the micropanel.

			const dim_t      i      = cdim0;
			const dim_t      m_edge = mnr - cdim0;
			const dim_t      n_edge = k0_max;
			double* restrict p_edge = p + (i  )*1;

			bli_dset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, ldp
			);
		}
	}

	if ( k0 < k0_max )
	{
		// Handle zero-filling along the "short" (far) edge of the micropanel.

		const dim_t      j      = k0;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = k0_max - k0;
		double* restrict p_edge = p + (j  )*ldp;

		bli_dset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, ldp
		);
	}
}

