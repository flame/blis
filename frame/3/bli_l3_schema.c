/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

void bli_l3_set_schemas
     (
             num_t   dt,
             pack_t* schema_a,
             pack_t* schema_b,
       const cntx_t* cntx
     )
{
	// Begin with pack schemas for native execution.
	*schema_a = BLIS_PACKED_ROW_PANELS;
	*schema_b = BLIS_PACKED_COL_PANELS;

	// When executing the 1m method, choose the appropriate pack schemas based
	// on the microkernel preference encoded within the current cntx_t (which
	// was presumably returned by the gks).
	if ( bli_cntx_method( cntx ) == BLIS_1M )
	{
		// Note that bli_cntx_l3_vir_ukr_prefers_cols_dt() will use the real
		// projection of dt to query the preference of the corresponding native
		// real-domain microkernel. This is what ultimately determines which
		// variant of 1m is applicable.
		if ( bli_cntx_ukr_prefers_cols_dt( bli_dt_proj_to_real( dt ), BLIS_GEMM_UKR, cntx ) )
		{
			*schema_a = BLIS_PACKED_ROW_PANELS_1E;
			*schema_b = BLIS_PACKED_COL_PANELS_1R;
		}
		else
		{
			*schema_a = BLIS_PACKED_ROW_PANELS_1R;
			*schema_b = BLIS_PACKED_COL_PANELS_1E;
		}
	}
}

