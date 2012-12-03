/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

extern her2k_t* her2k_cntl;

//
// Define object-based interface.
//
void bl2_her2k( obj_t*  alpha,
                obj_t*  a,
                obj_t*  bh,
                obj_t*  beta,
                obj_t*  c )
{
	her2k_t* cntl;
	obj_t    alpha_local;
	obj_t    alpha_conj_local;
	obj_t    beta_local;
	obj_t    b;
	obj_t    ah;
	num_t    dt_targ_a;
	num_t    dt_targ_bh;
	num_t    dt_targ_c;
	num_t    dt_exec;
	num_t    dt_alpha;
	num_t    dt_beta;
	bool_t   pack_c = FALSE;

	// Check parameters.
	if ( bl2_error_checking_is_enabled() )
		bl2_her2k_check( alpha, a, bh, beta, c );

	// If alpha is zero, scale by beta and return.
	if ( bl2_obj_scalar_equals( alpha, &BLIS_ZERO ) )
	{
		bl2_scalm( beta, c );
		return;
	}

	// Determine the target datatype of each matrix object.
	//bl2_her2k_get_target_datatypes( a,
	//                                b,
	//                                c,
	//                                &dt_targ_a,
	//                                &dt_targ_b,
	//                                &dt_targ_c,
	//                                &pack_c );

	dt_targ_a  = bl2_obj_datatype( *a  );
	dt_targ_bh = bl2_obj_datatype( *bh );
	dt_targ_c  = bl2_obj_datatype( *c  );

	// Set the target datatypes for each matrix object.
	bl2_obj_set_target_datatype( dt_targ_a,  *a  );
	bl2_obj_set_target_datatype( dt_targ_bh, *bh );
	bl2_obj_set_target_datatype( dt_targ_c,  *c  );

	// Determine the execution datatype.
	dt_exec = dt_targ_a;

	// Embed the execution datatype in all matrix operands.
	bl2_obj_set_execution_datatype( dt_exec, *a  );
	bl2_obj_set_execution_datatype( dt_exec, *bh );
	bl2_obj_set_execution_datatype( dt_exec, *c  );

	// Create objects to track B and A' (for the second rank-k update).
	bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, *bh, b );
	bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, *a, ah );


	// Create an object to hold a copy-cast of alpha. Notice that we use
	// the target datatype of matrix a. By inspecting the table above,
	// this clearly works for cases (0) through (4), (6), and (7). It
	// Also works for case (5) since it is transformed into case (6) by
	// the above code.
	dt_alpha = dt_targ_a;
	bl2_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_NO_CONJUGATE,
	                             alpha,
	                             &alpha_local );

	// Create an object to hold a copy-cast of conj(alpha).
	dt_alpha = dt_targ_bh;
	bl2_obj_init_scalar_copy_of( dt_alpha,
	                             BLIS_CONJUGATE,
	                             alpha,
	                             &alpha_conj_local );

	// Create an object to hold a copy-cast of beta. Notice that we use
	// the datatype of c. Here's why: If c is real and beta is complex,
	// there is no reason to keep beta_local in the complex domain since
	// the complex part of beta*c will not be stored. If c is complex and
	// beta is real then beta is harmlessly promoted to complex.
	dt_beta = bl2_obj_datatype( *c );
	bl2_obj_init_scalar_copy_of( dt_beta,
	                             BLIS_NO_CONJUGATE,
	                             beta,
	                             &beta_local );

	// Choose the control tree based on whether it was determined we need
	// to pack c.
	//if ( pack_c ) her2k_cntl = her2k_cntl_packabc;
	//else          her2k_cntl = her2k_cntl_packab;
	cntl = her2k_cntl;
	if ( pack_c ) bl2_abort();

	// Invoke the internal back-end.
	bl2_her2k_int( &alpha_local,
	               a,
	               bh,
	               &alpha_conj_local,
	               &b,
	               &ah,
	               &beta_local,
	               c,
	               cntl );
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
\
    bl2_set_dims_with_trans( transa, m, k, m_a, n_a ); \
    bl2_set_dims_with_trans( transb, m, k, m_b, n_b ); \
\
    bl2_obj_create_scalar_with_attached_buffer( dt, alpha, &alphao ); \
    bl2_obj_create_scalar_with_attached_buffer( dt, beta,  &betao  ); \
\
    bl2_obj_create_with_attached_buffer( dt, m_a, n_a, a, rs_a, cs_a, &ao ); \
    bl2_obj_create_with_attached_buffer( dt, m_b, n_b, b, rs_b, cs_b, &bo ); \
    bl2_obj_create_with_attached_buffer( dt, m,   m,   c, rs_c, cs_c, &co ); \
\
    bl2_obj_set_uplo( uploc, co ); \
    bl2_obj_set_conjtrans( transa, ao ); \
    bl2_obj_set_conjtrans( transb, bo ); \
\
    bl2_obj_set_struc( BLIS_HERMITIAN, co ); \
\
    PASTEMAC0(opname)( &alphao, \
                       &ao, \
                       &bo, \
                       &betao, \
                       &co ); \
}

INSERT_GENTFUNC_BASIC( her2k, her2k )


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
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC3U12_BASIC( her2k, her2k )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( her2k, her2k )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( her2k, her2k )
#endif

