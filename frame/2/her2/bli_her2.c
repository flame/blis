/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

extern her2_t* her2_cntl_bs_ke_row;
extern her2_t* her2_cntl_bs_ke_col;
extern her2_t* her2_cntl_ge_row;
extern her2_t* her2_cntl_ge_col;

void bli_her2( obj_t*  alpha,
               obj_t*  x,
               obj_t*  y,
               obj_t*  c )
{
	her2_t* her2_cntl;
	num_t   dt_targ_x;
	num_t   dt_targ_y;
	num_t   dt_targ_c;
	bool_t  x_is_contig;
	bool_t  y_is_contig;
	bool_t  c_is_contig;
	obj_t   alpha_local;
	obj_t   alpha_conj_local;
	num_t   dt_alpha;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_her2_check( BLIS_CONJUGATE, alpha, x, y, c );


	// Query the target datatypes of each object.
	dt_targ_x = bli_obj_target_datatype( *x );
	dt_targ_y = bli_obj_target_datatype( *y );
	dt_targ_c = bli_obj_target_datatype( *c );

	// Determine whether each operand is stored contiguously.
	x_is_contig = ( bli_obj_vector_inc( *x ) == 1 );
	y_is_contig = ( bli_obj_vector_inc( *y ) == 1 );
	c_is_contig = ( bli_obj_is_row_stored( *c ) ||
	                bli_obj_is_col_stored( *c ) );


	// Create an object to hold a copy-cast of alpha. Notice that we use
	// the type union of the datatypes of x and y.
	dt_alpha = bli_datatype_union( dt_targ_x, dt_targ_y );
	bli_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_NO_CONJUGATE,
	                             alpha,
	                             &alpha_local );

	// Also create a conjugated copy of alpha.
	bli_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_CONJUGATE,
	                             alpha,
	                             &alpha_conj_local );


	// If all operands are contiguous, we choose a control tree for calling
	// the unblocked implementation directly without any blocking.
	if ( x_is_contig &&
	     y_is_contig &&
	     c_is_contig )
	{
		// Use different control trees depending on storage of the matrix
		// operand.
		if ( bli_obj_is_row_stored( *c ) ) her2_cntl = her2_cntl_bs_ke_row;
		else                               her2_cntl = her2_cntl_bs_ke_col;
	}
	else
	{
		// Mark objects with unit stride as already being packed. This prevents
		// unnecessary packing from happening within the blocked algorithm.
		if ( x_is_contig ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, *x );
		if ( y_is_contig ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, *y );
		if ( c_is_contig ) bli_obj_set_pack_schema( BLIS_PACKED_UNSPEC, *c );

		// Here, we make a similar choice as above, except that (1) we look
		// at storage tilt, and (2) we choose a tree that performs blocking.
		if ( bli_obj_is_row_tilted( *c ) ) her2_cntl = her2_cntl_ge_row;
		else                               her2_cntl = her2_cntl_ge_col;
	}


	// Invoke the internal back-end with the copy-cast scalar and the
	// chosen control tree. Set conjh to BLIS_CONJUGATE to invoke the
	// Hermitian (and not symmetric) algorithms.
	bli_her2_int( BLIS_CONJUGATE,
	              &alpha_local,
	              &alpha_conj_local,
	              x,
	              y,
	              c,
	              her2_cntl );
}


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          uplo_t    uploc, \
                          conj_t    conjx, \
                          conj_t    conjy, \
                          dim_t     m, \
                          ctype*    alpha, \
                          ctype*    x, inc_t incx, \
                          ctype*    y, inc_t incy, \
                          ctype*    c, inc_t rs_c, inc_t cs_c \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, xo, yo, co; \
\
	inc_t       rs_x, cs_x; \
	inc_t       rs_y, cs_y; \
	err_t       init_result; \
\
	bli_init_safe( &init_result ); \
\
	rs_x = incx; cs_x = m * incx; \
	rs_y = incy; cs_y = m * incy; \
\
	bli_obj_create_scalar_with_attached_buffer( dt, alpha, &alphao ); \
\
	bli_obj_create_with_attached_buffer( dt, m, 1, x, rs_x, cs_x, &xo ); \
	bli_obj_create_with_attached_buffer( dt, m, 1, y, rs_y, cs_y, &yo ); \
	bli_obj_create_with_attached_buffer( dt, m, m, c, rs_c, cs_c, &co ); \
\
	bli_obj_set_conj( conjx, xo ); \
	bli_obj_set_conj( conjy, yo ); \
	bli_obj_set_uplo( uploc, co ); \
\
	bli_obj_set_struc( BLIS_HERMITIAN, co ); \
\
	PASTEMAC0(opname)( &alphao, \
	                   &xo, \
	                   &yo, \
	                   &co ); \
\
	bli_finalize_safe( init_result ); \
}

INSERT_GENTFUNC_BASIC( her2, her2 )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_c, ctype_xy, chx, chy, chc, chxy, opname, varname ) \
\
void PASTEMAC3(chx,chy,chc,opname)( \
                                    uplo_t    uploc, \
                                    conj_t    conjx, \
                                    conj_t    conjy, \
                                    dim_t     m, \
                                    ctype_xy* alpha, \
                                    ctype_x*  x, inc_t incx, \
                                    ctype_y*  y, inc_t incy, \
                                    ctype_c*  c, inc_t rs_c, inc_t cs_c \
                                  ) \
{ \
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC3U12_BASIC( her2, her2 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( her2, her2 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( her2, her2 )
#endif

