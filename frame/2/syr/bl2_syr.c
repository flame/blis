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

#include "blis2.h"

extern her_t* her_cntl_bs_ke_row;
extern her_t* her_cntl_bs_ke_col;
extern her_t* her_cntl_ge_row;
extern her_t* her_cntl_ge_col;

void bl2_syr( obj_t*  alpha,
              obj_t*  x,
              obj_t*  c )
{
	her_t*  her_cntl;
	num_t   dt_targ_x;
	num_t   dt_targ_c;
	bool_t  x_is_contig;
	bool_t  c_is_contig;
	obj_t   alpha_local;
	num_t   dt_alpha;

	// Check parameters.
	if ( bl2_error_checking_is_enabled() )
		bl2_syr_check( alpha, x, c );


	// Query the target datatypes of each object.
	dt_targ_x = bl2_obj_target_datatype( *x );
	dt_targ_c = bl2_obj_target_datatype( *c );

	// Determine whether each operand is stored contiguously.
	x_is_contig = ( bl2_obj_vector_inc( *x ) == 1 );
	c_is_contig = ( bl2_obj_is_row_stored( *c ) ||
	                bl2_obj_is_col_stored( *c ) );


	// Create an object to hold a copy-cast of alpha. Notice that we use
	// the type union of the target datatypes of x and c to prevent any
	// unnecessary loss of information during the computation.
	dt_alpha = bl2_datatype_union( dt_targ_x, dt_targ_c );
	bl2_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_NO_CONJUGATE,
	                             alpha,
	                             &alpha_local );


	// If all operands are contiguous, we choose a control tree for calling
	// the unblocked implementation directly without any blocking.
	if ( x_is_contig &&
	     c_is_contig )
	{
		// Use different control trees depending on storage of the matrix
		// operand.
		if ( bl2_obj_is_row_stored( *c ) ) her_cntl = her_cntl_bs_ke_row;
		else                               her_cntl = her_cntl_bs_ke_col;
	}
    else
    {
		// Mark objects with unit stride as already being packed. This prevents
		// unnecessary packing from happening within the blocked algorithm.
        if ( x_is_contig ) bl2_obj_set_pack_schema( BLIS_PACKED_VECTOR, *x );
        if ( c_is_contig ) bl2_obj_set_pack_schema( BLIS_PACKED_UNSPEC, *c );

		// Here, we make a similar choice as above, except that (1) we look
		// at storage tilt, and (2) we choose a tree that performs blocking.
        if ( bl2_obj_is_row_tilted( *c ) ) her_cntl = her_cntl_ge_row;
        else                               her_cntl = her_cntl_ge_col;
    }


	// Invoke the internal back-end with the copy-cast scalar and the
	// chosen control tree. Set conjh to BLIS_NO_CONJUGATE to invoke the
	// symmetric (and not Hermitian) algorithms.
	bl2_her_int( BLIS_NO_CONJUGATE,
	             &alpha_local,
	             x,
	             c,
	             her_cntl );
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
                          dim_t     m, \
                          ctype*    alpha, \
                          ctype*    x, inc_t incx, \
                          ctype*    c, inc_t rs_c, inc_t cs_c \
                        ) \
{ \
    const num_t dt = PASTEMAC(ch,type); \
\
    obj_t       alphao, xo, co; \
\
    inc_t       rs_x, cs_x; \
\
    rs_x = incx; cs_x = m * incx; \
\
    bl2_obj_create_scalar_with_attached_buffer( dt, alpha, &alphao ); \
\
    bl2_obj_create_with_attached_buffer( dt, m, 1, x, rs_x, cs_x, &xo ); \
    bl2_obj_create_with_attached_buffer( dt, m, m, c, rs_c, cs_c, &co ); \
\
    bl2_obj_set_conj( conjx, xo ); \
    bl2_obj_set_uplo( uploc, co ); \
\
    PASTEMAC0(opname)( &alphao, \
                       &xo, \
                       &co ); \
}

INSERT_GENTFUNC_BASIC( syr, syr )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_c, chx, chc, opname, varname ) \
\
void PASTEMAC2(chx,chc,opname)( \
                                uplo_t    uploc, \
                                conj_t    conjx, \
                                dim_t     m, \
                                ctype_x*  alpha, \
                                ctype_x*  x, inc_t incx, \
                                ctype_c*  c, inc_t rs_c, inc_t cs_c \
                              ) \
{ \
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC2_BASIC( syr, syr )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( syr, syr )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( syr, syr )
#endif
