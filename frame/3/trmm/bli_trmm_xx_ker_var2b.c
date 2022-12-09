/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#include "blis.h"

static l3_var_oft vars[2][2] =
{
	{ bli_trmm_ll_ker_var2b, bli_trmm_lu_ker_var2b },
	{ bli_trmm_rl_ker_var2b, bli_trmm_ru_ker_var2b }
};

void bli_trmm_xx_ker_var2b
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	dim_t side;
	dim_t uplo;

	// Set two bools: one based on the implied side parameter (the structure
	// of the root object) and one based on the uplo field of the triangular
	// matrix's root object (whether that is matrix A or matrix B).
	if ( bli_obj_root_is_triangular( a ) )
	{
		side = 0;
		if ( bli_obj_root_is_lower( a ) ) uplo = 0;
		else                              uplo = 1;
	}
	else // if ( bli_obj_root_is_triangular( b ) )
	{
		side = 1;
		if ( bli_obj_root_is_lower( b ) ) uplo = 0;
		else                              uplo = 1;
	}

	// Index into the variant array to extract the correct function pointer.
	l3_var_oft f = vars[side][uplo];

	// Call the macrokernel.
	f
	(
	  a,
	  b,
	  c,
	  cntx,
	  cntl,
	  thread_par
	);
}

