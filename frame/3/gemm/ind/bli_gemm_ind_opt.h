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

BLIS_INLINE void bli_gemm_ind_recast_1m_params
     (
             num_t*  dt_exec,
             num_t*  dt_c,
             pack_t  schema_a,
       const obj_t*  c,
             dim_t*  m,
             dim_t*  n,
             dim_t*  k,
             inc_t*  pd_a, inc_t* ps_a,
             inc_t*  pd_b, inc_t* ps_b,
             inc_t*  rs_c, inc_t* cs_c,
       const cntx_t* cntx
     )
{
	obj_t beta;

	// Detach the beta scalar from c so that we can test its imaginary
	// component.
	bli_obj_scalar_detach( c, &beta );

#if 1
	// Determine whether the storage of C matches the IO preference of the
	// microkernel. (We cannot utilize the optimization below if there is a
	// mismatch.)
	const ukr_t ukr_id     = BLIS_GEMM_VIR_UKR;

	const bool  row_stored = bli_is_row_stored( *rs_c, *cs_c );
	const bool  col_stored = !row_stored;
	const bool  row_pref   = bli_cntx_ukr_prefers_rows_dt( *dt_c, ukr_id, cntx );
	const bool  col_pref   = !row_pref;

	const bool  is_match   = ( row_stored && row_pref ) ||
	                         ( col_stored && col_pref );
#else
	// This was the previous behavior, which resulted in buggy behavior
	// when executing right-sided hemm, and:
	// - the 1m method is enabled,
	// - BLIS_DISABLE_HEMM_RIGHT is #defined, and
	// - the storage of C matches the microkernel IO preference PRIOR to
	//   detecting the right-sidedness of the operation.
	// See Issue #621 for details.
	const bool is_match = TRUE;
#endif

	// If (a) the storage of C matches the IO pref of the ukernel, (b) beta is
	// in the real domain, and (c) C is row- or column-stored, then we may
	// proceed with the optimization below, which allows 1m to be induced by
	// executing the real-domain macrokernel with the real-domain microkernel
	// plus a few tweaked parameters. Otherwise, we must skip the optimization
	// and allow 1m to execute via the complex-domain macrokernel calling the
	// 1m virtual microkernel function, which will incur a little extra
	// overhead.
	if ( is_match &&
	     bli_obj_imag_is_zero( &beta ) &&
	     !bli_is_gen_stored( *rs_c, *cs_c ) )
	{
		*dt_exec = bli_dt_proj_to_real( *dt_exec );
		*dt_c    = bli_dt_proj_to_real( *dt_c );

		if ( bli_is_1e_packed( schema_a ) )
		{
			*m    *= 2;
			*n    *= 1;
			*k    *= 2;
			*pd_a *= 2; *ps_a *= 2;
			*pd_b *= 1; *ps_b *= 2;
			*rs_c *= 1; *cs_c *= 2;
		}
		else // if ( bli_is_1r_packed( schema_a ) )
		{
			*m    *= 1;
			*n    *= 2;
			*k    *= 2;
			*pd_a *= 1; *ps_a *= 2;
			*pd_b *= 2; *ps_b *= 2;
			*rs_c *= 2; *cs_c *= 1;
		}
	}
}

