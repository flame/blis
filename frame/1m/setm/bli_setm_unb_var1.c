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

#define FUNCPTR_T setm_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffx,
                           diag_t  diagx,
                           uplo_t  uplox,
                           dim_t   m,
                           dim_t   n,
                           void*   beta,
                           void*   x, inc_t rs_x, inc_t cs_x
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY2_ALL(ftypes,setm_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY2_EXT(ftypes,setm_unb_var1);
#else
static FUNCPTR_T GENARRAY2_MIN(ftypes,setm_unb_var1);
#endif
#endif


void bli_setm_unb_var1( obj_t*  beta,
                        obj_t*  x )
{
	num_t     dt_x      = bli_obj_datatype( *x );

	doff_t    diagoffx  = bli_obj_diag_offset( *x );
	diag_t    diagx     = bli_obj_diag( *x );
	uplo_t    uplox     = bli_obj_uplo( *x );

	dim_t     m         = bli_obj_length( *x );
	dim_t     n         = bli_obj_width( *x );

	void*     buf_x     = bli_obj_buffer_at_off( *x );
	inc_t     rs_x      = bli_obj_row_stride( *x );
	inc_t     cs_x      = bli_obj_col_stride( *x );

	void*     buf_beta;
	num_t     dt_beta;

	FUNCPTR_T f;

	// If beta is a scalar constant, use dt_x to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the beta object and extract the buffer at the beta offset.
	bli_set_scalar_dt_buffer( beta, dt_x, dt_beta, buf_beta );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_beta][dt_x];

	// Invoke the function.
	f( diagoffx,
	   diagx,
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
                                 doff_t  diagoffx, \
                                 diag_t  diagx, \
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
	/* When the diagonal of x is implicitly unit, we update ONLY the stored
	   region strictly above or below the diagonal. */ \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_1m( diagoffx, diagx, \
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
			PASTEMAC2(chb,chx,kername)( n_elem, \
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
				PASTEMAC2(chb,chx,kername)( n_elem, \
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
				PASTEMAC2(chb,chx,kername)( n_elem, \
				                            beta_cast, \
				                            x1, incx ); \
			} \
		} \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( setm_unb_var1, SETV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( setm_unb_var1, SETV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( setm_unb_var1, SETV_KERNEL )
#endif

