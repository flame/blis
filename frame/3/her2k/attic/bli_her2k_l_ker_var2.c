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

void bli_her2k_l_ker_var2( obj_t*   a,
                           obj_t*   bh,
                           obj_t*   b,
                           obj_t*   ah,
                           obj_t*   c,
                           her2k_t* cntl )
{
	herk_t herk_cntl;
	obj_t  c_local;

	// Implement her2k kernel in terms of two calls to the corresponding
	// herk kernel.

	// Note we have to use BLIS_ONE for the second rank-k product since we
	// only want to apply beta once. (And beta might be unit anyway if this
	// is not the first iteration of variant 3.)

	cntl_gemm_ukrs( (&herk_cntl) ) = cntl_gemm_ukrs( cntl );

	bli_obj_alias_to( *c, c_local );

	bli_herk_l_ker_var2( a,
	                     bh,
	                     &c_local,
	                     &herk_cntl );

	bli_obj_scalar_reset( &c_local );

	bli_herk_l_ker_var2( b,
	                     ah,
	                     &c_local,
	                     &herk_cntl );
}

