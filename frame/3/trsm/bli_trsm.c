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

extern trsm_t* trsm_l_cntl;
extern trsm_t* trsm_r_cntl;

//
// Define object-based interface.
//
void bli_trsm( side_t  side,
               obj_t*  alpha,
               obj_t*  a,
               obj_t*  b )
{
	trsm_t* cntl;
	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( alpha, b );
		return;
	}

	// Alias A and B so we can tweak the objects if necessary.
	bli_obj_alias_to( *a, a_local );
	bli_obj_alias_to( *b, b_local );
	bli_obj_alias_to( *b, c_local );

	// We do not explicitly implement the cases where A is transposed.
	// However, we can still handle them. Specifically, if A is marked as
	// needing a transposition, we simply induce a transposition. This
	// allows us to only explicitly implement the no-transpose cases. Once
	// the transposition is induced, the correct algorithm will be called,
	// since, for example, an algorithm over a transposed lower triangular
	// matrix A moves in the same direction (forwards) as a non-transposed
	// upper triangular matrix. And with the transposition induced, the
	// matrix now appears to be upper triangular, so the upper triangular
	// algorithm will grab the correct partitions, as if it were upper
	// triangular (with no transpose) all along.
	if ( bli_obj_has_trans( a_local ) )
	{
		bli_obj_induce_trans( a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, a_local );
	}

#if 0
	if ( bli_is_right( side ) )
	{
		bli_obj_induce_trans( a_local );
		bli_obj_induce_trans( b_local );
		bli_obj_induce_trans( c_local );

		bli_toggle_side( side );
	}
#endif

#if 1
	// If A is being solved against from the right, swap A and B so that
	// the matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( a_local, b_local );
	}

	// An optimization: If C is row-stored, transpose the entire operation
	// so as to allow the macro-kernel more favorable access patterns
	// through C. (The effect of the transposition of A and B is negligible
	// because those operands are always packed to contiguous memory.)
	if ( bli_obj_is_row_stored( c_local ) )
	{
		bli_obj_swap( a_local, b_local );

		bli_obj_induce_trans( a_local );
		bli_obj_induce_trans( b_local );
		bli_obj_induce_trans( c_local );

		bli_toggle_side( side );
	}
#endif

	// Set each alias as the root object.
	// NOTE: We MUST wait until we are done potentially swapping the objects
	// before setting the root fields!
	bli_obj_set_as_root( a_local );
	bli_obj_set_as_root( b_local );
	bli_obj_set_as_root( c_local );

	// Choose the control tree.
	if ( bli_is_left( side ) ) cntl = trsm_l_cntl;
	else                       cntl = trsm_r_cntl;

	// Invoke the internal back-end.
	bli_trsm_int( alpha,
	              &a_local,
	              &b_local,
	              &BLIS_ZERO,
	              &c_local,
	              cntl );
}

//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          side_t  side, \
                          uplo_t  uploa, \
                          trans_t transa, \
                          diag_t  diaga, \
                          dim_t   m, \
                          dim_t   n, \
                          ctype*  alpha, \
                          ctype*  a, inc_t rs_a, inc_t cs_a, \
                          ctype*  b, inc_t rs_b, inc_t cs_b  \
                        ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, bo; \
\
	dim_t       mn_a; \
\
	bli_set_dim_with_side( side, m, n, mn_a ); \
\
	bli_obj_create_1x1_with_attached_buffer( dt, alpha, &alphao ); \
\
	bli_obj_create_with_attached_buffer( dt, mn_a, mn_a, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, m,    n,    b, rs_b, cs_b, &bo ); \
\
	bli_obj_set_uplo( uploa, ao ); \
	bli_obj_set_diag( diaga, ao ); \
	bli_obj_set_conjtrans( transa, ao ); \
\
	bli_obj_set_struc( BLIS_TRIANGULAR, ao ); \
\
	PASTEMAC0(opname)( side, \
	                   &alphao, \
	                   &ao, \
	                   &bo ); \
}

INSERT_GENTFUNC_BASIC( trsm, trsm )


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC2
#define GENTFUNC2( ctype_a, ctype_b, cha, chb, opname, varname ) \
\
void PASTEMAC2(cha,chb,opname)( \
                                side_t    side, \
                                uplo_t    uploa, \
                                trans_t   transa, \
                                diag_t    diaga, \
                                dim_t     m, \
                                dim_t     n, \
                                ctype_a*  alpha, \
                                ctype_a*  a, inc_t rs_a, inc_t cs_a, \
                                ctype_b*  b, inc_t rs_b, inc_t cs_b  \
                              ) \
{ \
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC2_BASIC( trsm, trsm )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( trsm, trsm )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( trsm, trsm )
#endif

