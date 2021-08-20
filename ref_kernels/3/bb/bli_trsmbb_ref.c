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

// An implementation that indexes through B with the assumption that all
// elements were broadcast (duplicated) by a factor of NP/NR.

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf, diagop ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const num_t     dt     = PASTEMAC(ch,type); \
\
	const dim_t     mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t     nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t     packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t     m      = mr; \
	const dim_t     n      = nr; \
\
	const inc_t     rs_a   = 1; \
	const inc_t     cs_a   = packmr; \
\
	const inc_t     rs_b   = packnr; \
\
	/* Assume that the degree of duplication is equal to packnr / nr. */ \
	const inc_t     cs_b   = packnr / nr; \
\
	dim_t           iter, i, j, l; \
	dim_t           n_behind; \
\
	for ( iter = 0; iter < m; ++iter ) \
	{ \
		i        = iter; \
		n_behind = i; \
\
		ctype* restrict alpha11  = a + (i  )*rs_a + (i  )*cs_a; \
		ctype* restrict a10t     = a + (i  )*rs_a + (0  )*cs_a; \
		ctype* restrict B0       = b + (0  )*rs_b + (0  )*cs_b; \
		ctype* restrict b1       = b + (i  )*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a10t * B0; */ \
		/* b1 = b1 / alpha11; */ \
		for ( j = 0; j < n; ++j ) \
		{ \
			ctype* restrict b01     = B0 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict beta11  = b1 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict gamma11 = c  + (i  )*rs_c + (j  )*cs_c; \
			ctype           beta11c = *beta11; \
			ctype           rho11; \
\
			/* beta11 = beta11 - a10t * b01; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( l = 0; l < n_behind; ++l ) \
			{ \
				ctype* restrict alpha10 = a10t + (l  )*cs_a; \
				ctype* restrict beta01  = b01  + (l  )*rs_b; \
\
				PASTEMAC(ch,axpys)( *alpha10, *beta01, rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: When preinversion is enabled, the INVERSE of alpha11
			   (1.0/alpha11) is stored during packing instead alpha11 so we
			   can multiply rather than divide. When preinversion is disabled,
			   alpha11 is stored and division happens below explicitly. */ \
			PASTEMAC(ch,scals)( *alpha11, beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, *gamma11 ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, *beta11 ); \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNC_BASIC3( trsmbb_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scals )
#else
INSERT_GENTFUNC_BASIC3( trsmbb_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscals )
#endif


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf, diagop ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const num_t     dt     = PASTEMAC(ch,type); \
\
	const dim_t     mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t     nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t     packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t     m      = mr; \
	const dim_t     n      = nr; \
\
	const inc_t     rs_a   = 1; \
	const inc_t     cs_a   = packmr; \
\
	const inc_t     rs_b   = packnr; \
\
	/* Assume that the degree of duplication is equal to packnr / nr. */ \
	const inc_t     cs_b   = packnr / nr; \
\
	dim_t           iter, i, j, l; \
	dim_t           n_behind; \
\
	for ( iter = 0; iter < m; ++iter ) \
	{ \
		i        = m - iter - 1; \
		n_behind = iter; \
\
		ctype* restrict alpha11  = a + (i  )*rs_a + (i  )*cs_a; \
		ctype* restrict a12t     = a + (i  )*rs_a + (i+1)*cs_a; \
		ctype* restrict b1       = b + (i  )*rs_b + (0  )*cs_b; \
		ctype* restrict B2       = b + (i+1)*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a12t * B2; */ \
		/* b1 = b1 / alpha11; */ \
		for ( j = 0; j < n; ++j ) \
		{ \
			ctype* restrict beta11  = b1 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict b21     = B2 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict gamma11 = c  + (i  )*rs_c + (j  )*cs_c; \
			ctype           beta11c = *beta11; \
			ctype           rho11; \
\
			/* beta11 = beta11 - a12t * b21; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( l = 0; l < n_behind; ++l ) \
			{ \
				ctype* restrict alpha12 = a12t + (l  )*cs_a; \
				ctype* restrict beta21  = b21  + (l  )*rs_b; \
\
				PASTEMAC(ch,axpys)( *alpha12, *beta21, rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: When preinversion is enabled, the INVERSE of alpha11
			   (1.0/alpha11) is stored during packing instead alpha11 so we
			   can multiply rather than divide. When preinversion is disabled,
			   alpha11 is stored and division happens below explicitly. */ \
			PASTEMAC(ch,diagop)( *alpha11, beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, *gamma11 ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, *beta11 ); \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNC_BASIC3( trsmbb_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scals )
#else
INSERT_GENTFUNC_BASIC3( trsmbb_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscals )
#endif

