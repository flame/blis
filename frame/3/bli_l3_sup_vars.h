/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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


//
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
       trans_t trans, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       stor3_t eff_id, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     );

GENPROT( gemmsup_ref_var1 )
GENPROT( gemmsup_ref_var2 )

GENPROT( gemmsup_ref_var1n )
GENPROT( gemmsup_ref_var2m )


//
// Prototype BLAS-like interfaces with void pointer operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          eff_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     );

INSERT_GENTPROT_BASIC0( gemmsup_ref_var1 )
INSERT_GENTPROT_BASIC0( gemmsup_ref_var2 )

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          eff_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     );

INSERT_GENTPROT_BASIC0( gemmsup_ref_var1n )
INSERT_GENTPROT_BASIC0( gemmsup_ref_var2m )

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemmsup_ref_var1n2m_opt_cases
     (
       num_t    dt,
       trans_t* trans,
       bool     packa,
       bool     packb,
       stor3_t* eff_id,
       cntx_t*  cntx
     )
{
	const bool row_pref = bli_cntx_l3_sup_ker_prefers_rows_dt( dt, *eff_id, cntx );

	// Handle row- and column-preferrential kernels separately.
	if ( row_pref )
	{
		if      ( packa && packb )
		{
			if      ( *eff_id == BLIS_RRC )
			{
				// Since C is already row-stored, we can use BLIS_RRR kernel instead.
				*eff_id = BLIS_RRR;
			}
			else if ( *eff_id == BLIS_CRC )
			{
				// BLIS_RRC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*eff_id = BLIS_CCC; // BLIS_RRR when transposed below.
			}
			else if ( *eff_id == BLIS_CRR )
			{
				// Induce a transpose to make C row-stored.
				// BLIS_RCC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_CCC; // BLIS_RRR when transposed below.
			}
		}
		else if ( packb )
		{
			if      ( *eff_id == BLIS_RRC )
			{
				// Since C is already row-stored, we can use BLIS_RRR kernel instead.
				*eff_id = BLIS_RRR;
			}
			else if ( *eff_id == BLIS_CRC )
			{
				// BLIS_RRC when transposed below (with packa instead of packb).
				// No transformation is beneficial here.
			}
			else if ( *eff_id == BLIS_RCC )
			{
				// C is already row-stored; cancel transposition and use BLIS_RCR
				// kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_RCR;
			}
			#if 0
			// This transformation performs poorly. Theory: packing A (formerly B)
			// when eff_id == BLIS_RCC (formerly BLIS_CRR) to row storage is slow
			// and kills the performance?
			else if ( eff_id == BLIS_CRR )
			{
				trans = bli_trans_toggled( trans );
				eff_id = BLIS_CRC; // BLIS_RRC when transposed below.
			}
			#endif
		}
		else if ( packa )
		{
			if      ( *eff_id == BLIS_CRR )
			{
				// Induce a transpose to make C row-stored.
				// BLIS_RCC when transposed below (both matrices still packed).
				// This allows us to use the BLIS_RRR kernel instead.
				*trans = bli_trans_toggled( *trans );
				*eff_id = BLIS_CCR; // BLIS_RCR when transposed below.
			}
		}
	}
	else
	{
		if ( ( dt == BLIS_DOUBLE ) || ( dt == BLIS_DCOMPLEX ) || ( dt == BLIS_SCOMPLEX ) )
		{
			// The optimizations are only done for CRC and RRC storage schemes to avoid RD kernels.
			// Optimizations for other storage schemes is yet to be done.
			if ( packa )
			{
				if( *eff_id == BLIS_CRC )
				{
					*eff_id = BLIS_CCC;
				}
				else if ( *eff_id == BLIS_RRC )
				{
					*trans = bli_trans_toggled( *trans );
					*eff_id = BLIS_RCC;
				}
			}
		}
		else
		{
			printf( "libblis: sup var1n2m_opt_cases not yet implemented for column-preferential kernels for S, C and Z datatypes.\n" );
			bli_abort();
		}
	}
}

