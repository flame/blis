/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

// Guard the function definitions so that they are only compiled when
// #included from files that define the typed API macros.
#ifdef BLIS_ENABLE_TAPI

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   n, \
	   x, incx, \
	   y, incy, \
	   cntx  \
	); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( addv,  BLIS_ADDV_KER )
INSERT_GENTFUNC_BASIC( copyv, BLIS_COPYV_KER )
INSERT_GENTFUNC_BASIC( subv,  BLIS_SUBV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       dim_t*  index  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   n, \
	   x, incx, \
	   index, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( amaxv, BLIS_AMAXV_KER )
INSERT_GENTFUNC_BASIC( aminv, BLIS_AMINV_KER )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	/* Early exit in case n is 0, or alpha is 0 and beta is 1 */ \
	if ( bli_zero_dim1( n ) || \
		 ( PASTEMAC( ch, eq0 )( *alpha ) && PASTEMAC( ch, eq1 )( *beta ) ) ) \
	{ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return; \
	} \
\
	/* 
		Setting all the required booleans based on special
	  cases of alpha and beta
	*/ \
	bool is_alpha_zero = PASTEMAC( ch, eq0 )( *alpha );	\
	bool is_alpha_one = PASTEMAC( ch, eq1 )( *alpha ); \
	bool is_beta_zero = PASTEMAC( ch, eq0 )( *beta ); \
	bool is_beta_one = PASTEMAC( ch, eq1 )( *beta ); \
	bool is_alpha_gen = !( is_alpha_zero || is_alpha_one );	\
	bool is_beta_gen = !( is_beta_zero || is_beta_one ); \
\
	/*
		Setting a map that would correspond to a distinct value
		based on any particular special case pair of alpha and beta.
		The map is a weighted sum of the booleans in powers of two.
	*/ \
	dim_t compute_map = is_alpha_zero + 2 * is_alpha_one + 4 * is_alpha_gen \
										+ 8 * is_beta_zero + 16 * is_beta_one + 32 * is_beta_gen;	\
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Reroute to other L1 kernels based on the compute type */	\
	switch ( compute_map ) \
	{ \
	  	/* When beta is 0 and alpha is 0 */ \
		case 9 : \
		{ \
			PASTECH2(ch,setv,_ker_ft) setv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_SETV_KER, cntx ); \
			setv_kf \
			( \
				BLIS_NO_CONJUGATE, \
				n, \
				beta, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
	  	/* When beta is 0 and alpha is 1 */ \
		case 10 : \
		{ \
			PASTECH2(ch,copyv,_ker_ft) copyv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_COPYV_KER, cntx ); \
			copyv_kf \
			( \
				conjx, \
				n, \
				x, incx, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
	  	/* When beta is 0 and alpha is not 0 or 1 */ \
		case 12 : \
		{ \
			PASTECH2(ch,scal2v,_ker_ft) scal2v_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_SCAL2V_KER, cntx ); \
			scal2v_kf \
			( \
				conjx, \
				n, \
				alpha, \
				x, incx, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
	  	/* When beta is 1 and alpha is 1 */ \
		case 18 : \
		{ \
			PASTECH2(ch,addv,_ker_ft) addv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_ADDV_KER, cntx ); \
			addv_kf \
			( \
				conjx, \
				n, \
				x, incx, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
	  	/* When beta is 1 and alpha is not 0 or 1 */ \
		case 20 : \
		{ \
			PASTECH2(ch,axpyv,_ker_ft) axpyv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
			axpyv_kf \
			( \
				conjx, \
				n, \
				alpha, \
				x, incx, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
	  	/* When beta is not 0 or 1 and alpha is 0 */ \
		case 33 : \
		{ \
			PASTECH2(ch,scalv,_ker_ft) scalv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_SCALV_KER, cntx ); \
			scalv_kf \
			( \
				BLIS_NO_CONJUGATE, \
				n, \
				beta, \
				y, incy, \
				cntx  \
			); \
			break; \
		} \
\
		/* The remaining cases of beta and alpha. I.e, beta != 0 or 1 and alpha != 0 or 1 */ \
		default : \
		{ \
			PASTECH2(ch,axpbyv,_ker_ft) axpbyv_kf = \
			bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPBYV_KER, cntx ); \
			axpbyv_kf \
			( \
				conjx, \
				n, \
				alpha, \
				x, incx, \
				beta, \
				y, incy, \
				cntx  \
			); \
		} \
	} \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( axpbyv, BLIS_AXPBYV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) \
		cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   n, \
	   alpha, \
	   x, incx, \
	   y, incy, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( axpyv,  BLIS_AXPYV_KER )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	/* The behaviour is undefined when increments are negative or 0 */ \
	/* So, return early */ \
	if( ( incx <= 0 ) || ( incy <= 0 ) ) \
	{ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return; \
	} \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) \
		cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   n, \
	   alpha, \
	   x, incx, \
	   y, incy, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( scal2v, BLIS_SCAL2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   conjy, \
	   n, \
	   x, incx, \
	   y, incy, \
	   rho, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( dotv, BLIS_DOTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  beta, \
       ctype*  rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   conjy, \
	   n, \
	   alpha, \
	   x, incx, \
	   y, incy, \
	   beta, \
	   rho, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( dotxv, BLIS_DOTXV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   n, \
	   x, incx, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( invertv, BLIS_INVERTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjalpha, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjalpha, \
	   n, \
	   alpha, \
	   x, incx, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( scalv, BLIS_SCALV_KER )
INSERT_GENTFUNC_BASIC( setv,  BLIS_SETV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   n, \
	   x, incx, \
	   y, incy, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( swapv, BLIS_SWAPV_KER )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC2(ch,opname,EX_SUF) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
\
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH2(ch,opname,_ker_ft) f = bli_cntx_get_l1v_ker_dt( dt, kerid, cntx ); \
\
	f \
	( \
	   conjx, \
	   n, \
	   x, incx, \
	   beta, \
	   y, incy, \
	   cntx  \
	); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
}

INSERT_GENTFUNC_BASIC( xpbyv, BLIS_XPBYV_KER )


#endif

