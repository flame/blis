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

#include "blis.h"


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, arch, suf, diagop ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       const void*      a0, \
             void*      b0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
             auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*      a      = a0; \
	      ctype*      b      = b0; \
	      ctype*      c      = c0; \
\
	const num_t       dt     = PASTEMAC(ch,type); \
	const num_t       dt_r   = PASTEMAC(chr,type); \
\
	const dim_t       mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t       nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t       packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t       packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t       m      = mr; \
	const dim_t       n      = nr; \
\
	const inc_t       rs_a   = bli_cntx_get_blksz_def_dt( dt_r, BLIS_BBM, cntx ); \
	const inc_t       cs_a   = packmr; \
\
	const inc_t       rs_b   = packnr; \
	const inc_t       cs_b   = bli_cntx_get_blksz_def_dt( dt_r, BLIS_BBN, cntx ); \
\
	const inc_t       ld_a  = cs_a; \
	const inc_t       ld_b  = rs_b; \
\
	const pack_t      schema_b = bli_auxinfo_schema_b( data ); \
\
	if ( bli_is_1e_packed( schema_b ) ) \
	{ \
		const inc_t rs_a2 = 1 * rs_a; \
		const inc_t cs_a2 = 2 * cs_a; \
		const inc_t rs_b2 = 2 * rs_b; \
		const inc_t cs_b2 = 2 * cs_b; \
\
		const ctype_r* restrict a_r  = ( ctype_r* )a; \
		const ctype_r* restrict a_i  = ( ctype_r* )a + ld_a; \
\
		      ctype_r* restrict b_ri = ( ctype_r* )b; \
		      ctype_r* restrict b_ir = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = iter; \
			dim_t n_behind  = i; \
\
			const ctype_r* restrict alpha11_r = a_r  + (i  )*rs_a2 + (i  )*cs_a2; \
			const ctype_r* restrict alpha11_i = a_i  + (i  )*rs_a2 + (i  )*cs_a2; \
			const ctype_r* restrict a10t_r    = a_r  + (i  )*rs_a2 + (0  )*cs_a2; \
			const ctype_r* restrict a10t_i    = a_i  + (i  )*rs_a2 + (0  )*cs_a2; \
			      ctype_r* restrict b1_ri     = b_ri + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict b1_ir     = b_ir + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B0_ri     = b_ri + (0  )*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a10t * B0; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_ri_r = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ri_i = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict beta11_ir_r = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ir_i = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict b01_ri      = B0_ri + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11     = c     + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r   = *beta11_ri_r; \
				ctype_r           beta11c_i   = *beta11_ri_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				PASTEMAC(ch,set0ris)( rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					const ctype_r* restrict alpha10_r = a10t_r  + (l  )*cs_a2; \
					const ctype_r* restrict alpha10_i = a10t_i  + (l  )*cs_a2; \
					const ctype_r* restrict beta01_r  = b01_ri  + (l  )*rs_b2 + 0*cs_b; \
					const ctype_r* restrict beta01_i  = b01_ri  + (l  )*rs_b2 + 1*cs_b; \
\
					PASTEMAC(ch,axpyris)( *alpha10_r, \
					                      *alpha10_i, \
					                      *beta01_r, \
					                      *beta01_i, \
					                      rho11_r, \
					                      rho11_i ); \
				} \
				PASTEMAC(ch,subris)( rho11_r, \
				                     rho11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(ch,diagop)( *alpha11_r, \
				                     *alpha11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				PASTEMAC(ch,sets)(  beta11c_r, beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
				{ \
					PASTEMAC(ch,copyris)(  beta11c_r, beta11c_i, *(beta11_ri_r + d), *(beta11_ri_i + d) ); \
					PASTEMAC(ch,copyris)( -beta11c_i, beta11c_r, *(beta11_ir_r + d), *(beta11_ir_i + d) ); \
                } \
			} \
		} \
	} \
	else /* ( bli_is_1r_packed( schema_b ) ) */ \
	{ \
		const inc_t rs_b2 = 2 * rs_b; \
		const inc_t cs_b2 = 1 * cs_b; \
\
		const ctype*   restrict a_ri  = ( ctype*   )a; \
		    /*ctype*   restrict a_ir  = ( ctype*   )a + ld_a/2;*/ \
\
		      ctype_r* restrict b_r   = ( ctype_r* )b; \
		      ctype_r* restrict b_i   = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = iter; \
			dim_t n_behind  = i; \
\
			const ctype*   restrict alpha11_ri = a_ri + (i  )*rs_a  + (i  )*cs_a; \
			const ctype_r* restrict alpha11_r  = &PASTEMAC(ch,real)( *alpha11_ri ); \
			const ctype_r* restrict alpha11_i  = &PASTEMAC(ch,imag)( *alpha11_ri ); \
			const ctype*   restrict a10t_ri    = a_ri + (i  )*rs_a  + (0  )*cs_a; \
			      ctype_r* restrict b1_r       = b_r  + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict b1_i       = b_i  + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B0_r       = b_r  + (0  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B0_i       = b_i  + (0  )*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a10t * B0; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_r  = b1_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict beta11_i  = b1_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b01_r     = B0_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b01_i     = B0_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11   = c    + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r = *beta11_r; \
				ctype_r           beta11c_i = *beta11_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				PASTEMAC(ch,set0ris)( rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					const ctype*   restrict alpha10_ri = a10t_ri + (l  )*cs_a; \
					const ctype_r* restrict alpha10_r  = &PASTEMAC(ch,real)( *alpha10_ri ); \
					const ctype_r* restrict alpha10_i  = &PASTEMAC(ch,imag)( *alpha10_ri ); \
					      ctype_r* restrict beta01_r   = b01_r   + (l  )*rs_b2; \
					      ctype_r* restrict beta01_i   = b01_i   + (l  )*rs_b2; \
\
					PASTEMAC(ch,axpyris)( *alpha10_r, \
					                      *alpha10_i, \
					                      *beta01_r, \
					                      *beta01_i, \
					                      rho11_r, \
					                      rho11_i ); \
				} \
				PASTEMAC(ch,subris)( rho11_r, \
				                     rho11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(ch,diagop)( *alpha11_r, \
				                     *alpha11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				PASTEMAC(ch,sets)( beta11c_r, \
				                   beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
					PASTEMAC(ch,copyris)( beta11c_r, \
					                      beta11c_i, \
					                      *(beta11_r + d), \
					                      *(beta11_i + d) ); \
			} \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCCO( trsm1m_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCCO( trsm1m_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, arch, suf, diagop ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       const void*      a0, \
             void*      b0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
             auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*      a      = a0; \
	      ctype*      b      = b0; \
	      ctype*      c      = c0; \
\
	const num_t       dt     = PASTEMAC(ch,type); \
	const num_t       dt_r   = PASTEMAC(chr,type); \
\
	const dim_t       mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t       nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t       packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t       packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t       m      = mr; \
	const dim_t       n      = nr; \
\
	const inc_t       rs_a   = bli_cntx_get_blksz_def_dt( dt_r, BLIS_BBM, cntx ); \
	const inc_t       cs_a   = packmr; \
\
	const inc_t       rs_b   = packnr; \
	const inc_t       cs_b   = bli_cntx_get_blksz_def_dt( dt_r, BLIS_BBN, cntx ); \
\
	const inc_t       ld_a  = cs_a; \
	const inc_t       ld_b  = rs_b; \
\
	const pack_t      schema_b = bli_auxinfo_schema_b( data ); \
\
	if ( bli_is_1e_packed( schema_b ) ) \
	{ \
		const inc_t rs_a2 = 1 * rs_a; \
		const inc_t cs_a2 = 2 * cs_a; \
		const inc_t rs_b2 = 2 * rs_b; \
		const inc_t cs_b2 = 2 * cs_b; \
\
		const ctype_r* restrict a_r  = ( ctype_r* )a; \
		const ctype_r* restrict a_i  = ( ctype_r* )a + ld_a; \
\
		      ctype_r* restrict b_ri = ( ctype_r* )b; \
		      ctype_r* restrict b_ir = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = m - iter - 1; \
			dim_t n_behind  = iter; \
\
			const ctype_r* restrict alpha11_r  = a_r  + (i  )*rs_a2 + (i  )*cs_a2; \
			const ctype_r* restrict alpha11_i  = a_i  + (i  )*rs_a2 + (i  )*cs_a2; \
			const ctype_r* restrict a12t_r     = a_r  + (i  )*rs_a2 + (i+1)*cs_a2; \
			const ctype_r* restrict a12t_i     = a_i  + (i  )*rs_a2 + (i+1)*cs_a2; \
			      ctype_r* restrict b1_ri      = b_ri + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict b1_ir      = b_ir + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B2_ri      = b_ri + (i+1)*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a12t * B2; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_ri_r = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ri_i = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict beta11_ir_r = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ir_i = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict b21_ri      = B2_ri + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11     = c     + (i  )*rs_c + (j  )*cs_c; \
				ctype_r           beta11c_r   = *beta11_ri_r; \
				ctype_r           beta11c_i   = *beta11_ri_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				PASTEMAC(ch,set0ris)( rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					const ctype_r* restrict alpha12_r = a12t_r + (l  )*cs_a2; \
					const ctype_r* restrict alpha12_i = a12t_i + (l  )*cs_a2; \
					      ctype_r* restrict beta21_r  = b21_ri + (l  )*rs_b2 + 0*cs_b; \
					      ctype_r* restrict beta21_i  = b21_ri + (l  )*rs_b2 + 1*cs_b; \
\
					PASTEMAC(ch,axpyris)( *alpha12_r, \
					                      *alpha12_i, \
					                      *beta21_r, \
					                      *beta21_i, \
					                      rho11_r, \
					                      rho11_i ); \
				} \
				PASTEMAC(ch,subris)( rho11_r, \
				                     rho11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(ch,diagop)( *alpha11_r, \
				                     *alpha11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				PASTEMAC(ch,sets)(  beta11c_r, beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
				{ \
					PASTEMAC(ch,copyris)(  beta11c_r, beta11c_i, *(beta11_ri_r + d), *(beta11_ri_i + d) ); \
					PASTEMAC(ch,copyris)( -beta11c_i, beta11c_r, *(beta11_ir_r + d), *(beta11_ir_i + d) ); \
                } \
			} \
		} \
	} \
	else /* if ( bli_is_1r_packed( schema_b ) ) */ \
	{ \
		const inc_t rs_b2 = 2 * rs_b; \
		const inc_t cs_b2 = 1 * cs_b; \
\
		const ctype*   restrict a_ri  = ( ctype*   )a; \
		    /*ctype*   restrict a_ir  = ( ctype*   )a + ld_a/2;*/ \
\
		      ctype_r* restrict b_r   = ( ctype_r* )b; \
		      ctype_r* restrict b_i   = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = m - iter - 1; \
			dim_t n_behind  = iter; \
\
			const ctype*   restrict alpha11_ri = a_ri + (i  )*rs_a  + (i  )*cs_a; \
			const ctype_r* restrict alpha11_r  = &PASTEMAC(ch,real)( *alpha11_ri ); \
			const ctype_r* restrict alpha11_i  = &PASTEMAC(ch,imag)( *alpha11_ri ); \
			const ctype*   restrict a12t_ri    = a_ri + (i  )*rs_a  + (i+1)*cs_a; \
			      ctype_r* restrict b1_r       = b_r  + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict b1_i       = b_i  + (i  )*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B2_r       = b_r  + (i+1)*rs_b2 + (0  )*cs_b2; \
			      ctype_r* restrict B2_i       = b_i  + (i+1)*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a12t * B2; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_r  = b1_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict beta11_i  = b1_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b21_r     = B2_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b21_i     = B2_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11   = c    + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r = *beta11_r; \
				ctype_r           beta11c_i = *beta11_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				PASTEMAC(ch,set0ris)( rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					const ctype*   restrict alpha12_ri = a12t_ri + (l  )*cs_a; \
					const ctype_r* restrict alpha12_r  = &PASTEMAC(ch,real)( *alpha12_ri ); \
					const ctype_r* restrict alpha12_i  = &PASTEMAC(ch,imag)( *alpha12_ri ); \
					      ctype_r* restrict beta21_r   = b21_r   + (l  )*rs_b2; \
					      ctype_r* restrict beta21_i   = b21_i   + (l  )*rs_b2; \
\
					PASTEMAC(ch,axpyris)( *alpha12_r, \
					                      *alpha12_i, \
					                      *beta21_r, \
					                      *beta21_i, \
					                      rho11_r, \
					                      rho11_i ); \
				} \
				PASTEMAC(ch,subris)( rho11_r, \
				                     rho11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(ch,diagop)( *alpha11_r, \
				                     *alpha11_i, \
				                     beta11c_r, \
				                     beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				PASTEMAC(ch,sets)( beta11c_r, \
				                   beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
					PASTEMAC(ch,copyris)( beta11c_r, \
					                      beta11c_i, \
					                      *(beta11_r + d), \
					                      *(beta11_i + d) ); \
			} \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCCO( trsm1m_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCCO( trsm1m_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif
