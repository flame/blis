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

#define FUNCPTR_T scalm_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conjbeta,
                           doff_t  diagoffx,
                           uplo_t  uplox,
                           dim_t   m,
                           dim_t   n,
                           void*   beta,
                           void*   x, inc_t rs_x, inc_t cs_x
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,scalm_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,scalm_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,scalm_unb_var1);
#endif
#endif


void bli_scalm_unb_var1( obj_t*  x )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	uplo_t    uplox     = bli_obj_uplo( *x );

	dim_t     m         = bli_obj_length( *x );
	dim_t     n         = bli_obj_width( *x );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );

	void*     buf_beta;

	FUNCPTR_T f;


	// Grab the address of the internal scalar buffer for the scalar
	// attached to x.
	buf_beta  = bli_obj_internal_scalar_buffer( *x );

	// Index into the type combination array to extract the correct
	// function pointer.
	// NOTE: We use dt_x for both beta and x because beta was obtained
	// from the attached scalar of x, which is guaranteed to be of the
	// same datatype as x.
	f = ftypes[dt_x][dt_x];

	// Invoke the function.
	// NOTE: We unconditionally pass in BLIS_NO_CONJUGATE for beta
	// because it would have already been conjugated by the front-end.
	f( BLIS_NO_CONJUGATE,
	   diagoffx,
	   uplox,
	   m,
	   n,
	   buf_beta,
	   buf_x, rs_x, cs_x );
}


#undef  GENTFUNC2
#define GENTFUNC2( ctype_b, ctype_x, chb, chx, varname, kername ) \
\
void PASTEMAC2(chb,chx,varname)( \
                                 conj_t  conjbeta, \
                                 doff_t  diagoffx, \
                                 uplo_t  uplox, \
                                 dim_t   m, \
                                 dim_t   n, \
                                 void*   beta, \
                                 void*   x, inc_t rs_x, inc_t cs_x \
                               ) \
{ \
	ctype_b* beta_cast = beta; \
	ctype_x* x_cast    = x; \
	ctype_x* x1; \
	uplo_t   uplox_eff; \
	dim_t    n_iter; \
	dim_t    n_elem, n_elem_max; \
	inc_t    ldx, incx; \
	dim_t    j, i; \
	dim_t    ij0, n_shift; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If beta is unit, the entire operation is a no-op. */ \
	if ( PASTEMAC(chb,eq1)( *beta_cast ) ) return; \
\
	/* Set various loop parameters. Here, we assume diagx is BLIS_NONUNIT_DIAG
	   because in _check() we disallow scalm on unit diagonal matrices. */ \
	bli_set_dims_incs_uplo_1m( diagoffx, BLIS_NONUNIT_DIAG, \
	                           uplox, m, n, rs_x, cs_x, \
	                           uplox_eff, n_elem_max, n_iter, incx, ldx, \
	                           ij0, n_shift ); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x1     = x_cast + (j  )*ldx + (0  )*incx; \
\
			PASTEMAC2(chb,chx,kername)( conjbeta, \
			                            n_elem, \
			                            beta_cast, \
			                            x1, incx ); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				x1     = x_cast + (ij0+j  )*ldx + (0  )*incx; \
\
				PASTEMAC2(chb,chx,kername)( conjbeta, \
				                            n_elem, \
				                            beta_cast, \
				                            x1, incx ); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				x1     = x_cast + (j  )*ldx + (ij0+i  )*incx; \
\
				PASTEMAC2(chb,chx,kername)( conjbeta, \
				                            n_elem, \
				                            beta_cast, \
				                            x1, incx ); \
			} \
		} \
	} \
}


// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( scalm_unb_var1, SCALV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( scalm_unb_var1, SCALV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( scalm_unb_var1, SCALV_KERNEL )
#endif

