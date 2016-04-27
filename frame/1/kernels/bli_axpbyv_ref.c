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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t          conjx, \
       dim_t           n, \
       ctype* restrict alpha, \
       ctype* restrict x, inc_t incx, \
       ctype* restrict beta, \
       ctype* restrict y, inc_t incy, \
       cntx_t*         cntx  \
     ) \
{ \
	ctype* restrict chi1; \
	ctype* restrict psi1; \
	dim_t  i; \
\
	if ( bli_zero_dim1( n ) ) return; \
\
	if ( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		/* If alpha is zero and beta is zero, set to zero. */ \
		if ( PASTEMAC(ch,eq0)( *beta ) ) \
		{ \
			ctype* zero = PASTEMAC(ch,0); \
\
			/* Query the context for the kernel function pointer. */ \
			const num_t         dt     = PASTEMAC(ch,type); \
			PASTECH(ch,setv_ft) setv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SETV_KER, cntx ); \
\
			setv_p \
			( \
			  BLIS_NO_CONJUGATE, \
			  n, \
			  zero, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
		/* If alpha is zero and beta is one, return. */ \
		else if ( PASTEMAC(ch,eq1)( *beta ) ) \
		{ \
			return; \
		} \
		/* If alpha is zero, scale by beta. */ \
		else \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t          dt      = PASTEMAC(ch,type); \
			PASTECH(ch,scalv_ft) scalv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SCALV_KER, cntx ); \
\
			scalv_p \
			( \
			  BLIS_NO_CONJUGATE, \
			  n, \
			  beta, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
\
	} \
	else if ( PASTEMAC(ch,eq1)( *alpha ) ) \
	{ \
		/* If alpha is one and beta is zero, copy. */ \
		if ( PASTEMAC(ch,eq0)( *beta ) ) \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t          dt      = PASTEMAC(ch,type); \
			PASTECH(ch,copyv_ft) copyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_COPYV_KER, cntx ); \
\
			copyv_p \
			( \
			  conjx, \
			  n, \
			  x, incx, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
		/* If alpha is one and beta is one, add. */ \
		else if ( PASTEMAC(ch,eq1)( *beta ) ) \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t         dt     = PASTEMAC(ch,type); \
			PASTECH(ch,addv_ft) addv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_ADDV_KER, cntx ); \
\
			addv_p \
			( \
			  conjx, \
			  n, \
			  x, incx, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
		/* If alpha is one, call xpby. */ \
		else \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t          dt      = PASTEMAC(ch,type); \
			PASTECH(ch,xpbyv_ft) xpbyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_XPBYV_KER, cntx ); \
\
			xpbyv_p \
			( \
			  conjx, \
			  n, \
			  x, incx, \
			  beta, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
	} \
	else \
	{ \
		/* If beta is zero, call scal2. */ \
		if ( PASTEMAC(ch,eq0)( *beta ) ) \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t           dt       = PASTEMAC(ch,type); \
			PASTECH(ch,scal2v_ft) scal2v_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SCAL2V_KER, cntx ); \
\
			scal2v_p \
			( \
			  conjx, \
			  n, \
			  alpha, \
			  x, incx, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
		/* If beta is one, call axpy. */ \
		else if ( PASTEMAC(ch,eq1)( *beta ) ) \
		{ \
			/* Query the context for the kernel function pointer. */ \
			const num_t          dt      = PASTEMAC(ch,type); \
			PASTECH(ch,axpyv_ft) axpyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
			axpyv_p \
			( \
			  conjx, \
			  n, \
			  alpha, \
			  x, incx, \
			  y, incy, \
			  cntx  \
			); \
			return; \
		} \
	\
	} \
\
	chi1 = x; \
	psi1 = y; \
\
	if ( bli_is_conj( conjx ) ) \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				PASTEMAC(ch,axpbyjs)( *alpha, chi1[i], *beta, psi1[i] ); \
			} \
		} \
		else \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				PASTEMAC(ch,axpbyjs)( *alpha, *chi1, *beta, *psi1 ); \
\
				chi1 += incx; \
				psi1 += incy; \
			} \
		} \
	} \
	else \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				PASTEMAC(ch,axpbys)( *alpha, chi1[i], *beta, psi1[i] ); \
			} \
		} \
		else \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				PASTEMAC(ch,axpbys)( *alpha, *chi1, *beta, *psi1 ); \
\
				chi1 += incx; \
				psi1 += incy; \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( axpbyv_ref )

