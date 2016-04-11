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

#include "blis.h"

void bli_hemv_cntx_init( cntx_t* cntx )
{
	// Perform basic setup on the context.
	bli_cntx_obj_create( cntx );

	// Initialize the context with kernels for the current architecture.
	bli_gks_cntx_set_l1v_ker( BLIS_AXPYV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_DOTXV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SCALV_KER, cntx );

	bli_gks_cntx_set_l1f_ker( BLIS_DOTAXPYV_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_AXPYF_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_DOTXF_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_DOTXAXPYF_KER, cntx );

	// Set the register and cache blocksizes and multiples, as well
	// as the execution method.
	bli_gks_cntx_set_blkszs( BLIS_NAT, 5,
	                         BLIS_N2, BLIS_N2,
	                         BLIS_M2, BLIS_M2,
	                         BLIS_AF, BLIS_AF,
	                         BLIS_DF, BLIS_DF,
	                         BLIS_XF, BLIS_XF,
	                         cntx );
}

void bli_hemv_cntx_finalize( cntx_t* cntx )
{
	// Free the context and all memory allocated to it.
	bli_cntx_obj_free( cntx );
}

