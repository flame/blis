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

extern her2k_t* her2k_cntl;
extern herk_t*  herk_cntl;

//
// Define object-based interface.
//
void bli_syr2k( obj_t*  alpha,
                obj_t*  a,
                obj_t*  b,
                obj_t*  beta,
                obj_t*  c )
{
	her2k_t* cntl;
	obj_t    alpha_local;
	obj_t    beta_local;
	obj_t    c_local;
	obj_t    a_local;
	obj_t    bt_local;
	obj_t    b_local;
	obj_t    at_local;
	num_t    dt_alpha;
	num_t    dt_beta;
	bool_t   pack_c;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syr2k_check( alpha, a, b, beta, c );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_scalar_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( *a, a_local );
	bli_obj_alias_to( *b, b_local );
	bli_obj_alias_to( *c, c_local );
	bli_obj_set_as_root( c_local );

	// For syr2k, the first and second right-hand "B" operands are simply B'
	// and A'.
	bli_obj_alias_to( *b, bt_local );
	bli_obj_induce_trans( bt_local );
	bli_obj_alias_to( *a, at_local );
	bli_obj_induce_trans( at_local );

	// An optimization: If C is row-stored, transpose the entire operation
	// so as to allow the macro-kernel more favorable access patterns
	// through C. (The effect of the transposition of A and A' is negligible
	// because those operands are always packed to contiguous memory.)
	if ( bli_obj_is_row_stored( c_local ) )
	{
		bli_obj_induce_trans( c_local );
	}

	// Set the target and execution datatypes of the objects, and apply
	// any transformations necessary to handle mixed domain computation.
	bli_her2k_set_targ_exec_datatypes( &a_local,
	                                   &bt_local,
	                                   &b_local,
	                                   &at_local,
	                                   &c_local,
	                                   &dt_alpha,
	                                   &dt_beta,
	                                   &pack_c );

	// Create an object to hold a copy-cast of alpha.
	bli_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_NO_CONJUGATE,
	                             alpha,
	                             &alpha_local );

	// Create an object to hold a copy-cast of beta.
	bli_obj_init_scalar_copy_of( dt_beta,
	                             BLIS_NO_CONJUGATE,
	                             beta,
	                             &beta_local );

	if ( pack_c ) bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	// Choose the control tree. We can just use her2k since the algorithm
	// is nearly identical to that of syr2k.
	cntl = her2k_cntl;

	// Invoke the internal back-end.
	bli_her2k_int( &alpha_local,
	               &a_local,
	               &bt_local,
	               &alpha_local,
	               &b_local,
	               &at_local,
	               &beta_local,
	               &c_local,
	               cntl );
/*
	bli_herk_int( &alpha_local,
	              a,
	              &bt,
	              &beta_local,
	              &c_local,
	              herk_cntl );
	bli_herk_int( &alpha_local,
	              b,
	              &at,
	              &BLIS_ONE,
	              &c_local,
	              herk_cntl );
*/
}

//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          uplo_t  uploc, \
                          trans_t transa, \
                          trans_t transb, \
                          dim_t   m, \
                          dim_t   k, \
                          ctype*  alpha, \
                          ctype*  a, inc_t rs_a, inc_t cs_a, \
                          ctype*  b, inc_t rs_b, inc_t cs_b, \
                          ctype*  beta, \
                          ctype*  c, inc_t rs_c, inc_t cs_c  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, bo, betao, co; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
	err_t       init_result; \
\
	bli_init_safe( &init_result ); \
\
	bli_set_dims_with_trans( transa, m, k, m_a, n_a ); \
	bli_set_dims_with_trans( transb, m, k, m_b, n_b ); \
\
	bli_obj_create_scalar_with_attached_buffer( dt, alpha, &alphao ); \
	bli_obj_create_scalar_with_attached_buffer( dt, beta,  &betao  ); \
\
	bli_obj_create_with_attached_buffer( dt, m_a, n_a, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, m_b, n_b, b, rs_b, cs_b, &bo ); \
	bli_obj_create_with_attached_buffer( dt, m,   m,   c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, co ); \
	bli_obj_set_conjtrans( transa, ao ); \
	bli_obj_set_conjtrans( transb, bo ); \
\
	bli_obj_set_struc( BLIS_SYMMETRIC, co ); \
\
	PASTEMAC0(opname)( &alphao, \
	                   &ao, \
	                   &bo, \
	                   &betao, \
	                   &co ); \
\
	bli_finalize_safe( init_result ); \
}

INSERT_GENTFUNC_BASIC( syr2k, syr2k )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, opname, varname ) \
\
void PASTEMAC3(cha,chb,chc,opname)( \
                                    uplo_t    uploc, \
                                    trans_t   transa, \
                                    trans_t   transb, \
                                    dim_t     m, \
                                    dim_t     k, \
                                    ctype_ab* alpha, \
                                    ctype_a*  a, inc_t rs_a, inc_t cs_a, \
                                    ctype_b*  b, inc_t rs_b, inc_t cs_b, \
                                    ctype_c*  beta, \
                                    ctype_c*  c, inc_t rs_c, inc_t cs_c  \
                                  ) \
{ \
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC3U12_BASIC( syr2k, syr2k )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( syr2k, syr2k )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( syr2k, syr2k )
#endif

