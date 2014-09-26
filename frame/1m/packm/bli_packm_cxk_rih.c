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

#define FUNCPTR_T packm_cxk_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conja,
                           pack_t  schema,
                           dim_t   panel_len,
                           void*   kappa,
                           void*   a, inc_t inca, inc_t lda,
                           void*   p,             inc_t ldp
                         );

#undef  FUNCPTR_ARRAY_LENGTH
#define FUNCPTR_ARRAY_LENGTH 32

static FUNCPTR_T ftypes_rih[FUNCPTR_ARRAY_LENGTH][BLIS_NUM_FP_TYPES] =
{
	/* micro-panel width = 0 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 1 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 2 */
	{
		NULL, BLIS_CPACKM_2XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_2XK_RIH_KERNEL,
	},
	/* micro-panel width = 3 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 4 */
	{
		NULL, BLIS_CPACKM_4XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_4XK_RIH_KERNEL,
	},
	/* micro-panel width = 5 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 6 */
	{
		NULL, BLIS_CPACKM_6XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_6XK_RIH_KERNEL,
	},
	/* micro-panel width = 7 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 8 */
	{
		NULL, BLIS_CPACKM_8XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_8XK_RIH_KERNEL,
	},
	/* micro-panel width = 9 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 10 */
	{
		NULL, BLIS_CPACKM_10XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_10XK_RIH_KERNEL,
	},
	/* micro-panel width = 11 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 12 */
	{
		NULL, BLIS_CPACKM_12XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_12XK_RIH_KERNEL,
	},
	/* micro-panel width = 13 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 14 */
	{
		NULL, BLIS_CPACKM_14XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_14XK_RIH_KERNEL,
	},
	/* micro-panel width = 15 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 16 */
	{
		NULL, BLIS_CPACKM_16XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_16XK_RIH_KERNEL,
	},
	/* micro-panel width = 17 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 18 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 19 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 20 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 21 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 22 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 23 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 24 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 25 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 26 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 27 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 28 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 29 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* micro-panel width = 30 */
	{
		NULL, BLIS_CPACKM_30XK_RIH_KERNEL,
		NULL, BLIS_ZPACKM_30XK_RIH_KERNEL,
	},
	/* micro-panel width = 31 */
	{
		NULL, NULL, NULL, NULL,
	},
};



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conja, \
                           pack_t  schema, \
                           dim_t   panel_dim, \
                           dim_t   panel_len, \
                           void*   kappa, \
                           void*   a, inc_t inca, inc_t lda, \
                           void*   p,             inc_t ldp  \
                         ) \
{ \
	num_t     dt; \
	FUNCPTR_T f; \
\
	/* Acquire the datatype for the current function. */ \
	dt = PASTEMAC(ch,type); \
\
	/* Index into the array to extract the correct function pointer.
	   If the micro-panel dimension is too big to be within the array of
	   explicitly handled kernels, then we treat that kernel the same
	   as if it were in range but unimplemented. */ \
	if ( panel_dim < FUNCPTR_ARRAY_LENGTH ) f = ftypes_rih[panel_dim][dt]; \
	else                                    f = NULL; \
\
	/* If there exists a kernel implementation for the micro-panel dimension
	   provided, we invoke the implementation. Otherwise, we use scal2m. */ \
	if ( f != NULL ) \
	{ \
		f( conja, \
		   schema, \
		   panel_len, \
		   kappa, \
		   a, inca, lda, \
		   p,       ldp ); \
	} \
	else \
	{ \
		ctype*   restrict kappa_cast = ( ctype*   )kappa; \
		ctype*   restrict a_r        = ( ctype*   )a; \
		ctype_r* restrict p_r        = ( ctype_r* )p; \
		const dim_t       inca1      = inca; \
		const dim_t       lda1       = lda; \
		const dim_t       ldp1       = ldp; \
		dim_t             i, j; \
\
		/* Treat the micro-panel as panel_dim x panel_len and column-stored
		   (unit row stride). */ \
\
		if ( bli_is_ro_packed( schema ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jros)( *kappa_cast, \
						                        *alpha11, \
						                        *pi11_r ); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2ros)( *kappa_cast, \
						                       *alpha11, \
						                       *pi11_r ); \
					} \
				} \
			} \
		} \
		else if ( bli_is_io_packed( schema ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jios)( *kappa_cast, \
						                        *alpha11, \
						                        *pi11_r ); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2ios)( *kappa_cast, \
						                       *alpha11, \
						                       *pi11_r ); \
					} \
				} \
			} \
		} \
		else /* if ( bli_is_rpi_packed( schema ) ) */ \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jrpis)( *kappa_cast, \
						                         *alpha11, \
						                         *pi11_r ); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2rpis)( *kappa_cast, \
						                        *alpha11, \
						                        *pi11_r ); \
					} \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_cxk_rih )

