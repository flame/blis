/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#ifndef BLIS_CHOL_H
#define BLIS_CHOL_H

#include "bli_chol_var.h"
#include "bli_chol_int.h"
#include "bli_chol_blksz.h"
#include "bli_chol_cntl.h"

#ifdef BLIS_ENABLE_LEVEL4

BLIS_EXPORT_BLIS err_t bli_chol
     (
       const obj_t*  a
     );

BLIS_EXPORT_BLIS err_t bli_chol_ex
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm
     );

#else

BLIS_INLINE err_t bli_chol
     (
       const obj_t*  a
     )
{
	return BLIS_SUCCESS;
}

BLIS_INLINE err_t bli_chol_ex
     (
       const obj_t*  a,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	return BLIS_SUCCESS;
}

#endif

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_chol_tweak_rntm
     (
       rntm_t*  rntm,
       rntm_t** rntm_chol,
       rntm_t** rntm_trsm,
       rntm_t** rntm_herk
     )
{
	*rntm_chol = rntm;
	*rntm_herk = rntm;

	**rntm_trsm = *rntm;

	dim_t ic = bli_rntm_ic_ways( rntm );

	if ( 1 < ic )
	{
		dim_t jr = bli_rntm_jr_ways( rntm );

		bli_rntm_set_ic_ways_only(       1, *rntm_trsm );
		bli_rntm_set_jr_ways_only( ic * jr, *rntm_trsm );
	}
}

#endif
