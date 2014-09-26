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

void bli_gemm_set_targ_exec_datatypes( obj_t*  a,
                                       obj_t*  b,
                                       obj_t*  c,
                                       bool_t* pack_c )
{
	num_t   dt_targ_a;
	num_t   dt_targ_b;
	num_t   dt_targ_c;
	num_t   dt_exec;

	// Determine the target datatype of each matrix object.
	bli_gemm_get_target_datatypes( a,
	                               b,
	                               c,
	                               &dt_targ_a,
	                               &dt_targ_b,
	                               &dt_targ_c,
	                               pack_c );
	
	// Set the target datatypes for each matrix object.
	bli_obj_set_target_datatype( dt_targ_a, *a );
	bli_obj_set_target_datatype( dt_targ_b, *b );
	bli_obj_set_target_datatype( dt_targ_c, *c );

	// Determine the execution datatype. Generally speaking, the
	// execution datatype is the real projection of the target datatype
	// of c. This rule holds unless the target datatypes of a and b
	// are both complex, in which case the execution datatype is also
	// complex.
	if ( bli_is_complex( dt_targ_a ) && bli_is_complex( dt_targ_b ) )
		dt_exec = dt_targ_c;
	else
		dt_exec = bli_datatype_proj_to_real( dt_targ_c );

	// Embed the execution datatype in all matrix operands.
	bli_obj_set_execution_datatype( dt_exec, *a );
	bli_obj_set_execution_datatype( dt_exec, *b );
	bli_obj_set_execution_datatype( dt_exec, *c );

	// Note that the precisions of the target datatypes of a, b, and c
	// match. The domains, however, are not necessarily the same. There
	// are eight possible combinations of target domains:
	//
	//   case  input     target    exec    pack  notes  
	//         domain    domain    domain  c?      
	//         c+=a*b    c+=a*b     
	//   (0)   r  r r    r  r r    r              
	//   (1)   r  r c    r  r r    r             b demoted to real
	//   (2)   r  c r    r  r r    r             a demoted to real
	//   (3)   r  c c    c  c c    c       yes   a*b demoted to real
	//   (4)   c  r r    r  r r    r       yes   copynzm used to update c
	//   (5)   c  r c    c  r c    r       yes   transposed to induce (6)
	//   (6)   c  c r    c  c r    r       ~     c and a treated as real
	//   (7)   c  c c    c  c c    c              
	//   ~ Must pack c if not column-stored (ie: row or general storage).
	//
	// There are two special cases: (5) and (6). Because the inner kernels
	// assume column storage, it is easy to implement (6) since we can
	// simply treat matrices c and a as real matrices with inflated m
	// dimension and column stride and then proceed with a kernel for real
	// computation. We cannot pull the same trick with case (5) because it
	// would result in a mismatch in the k dimension. But we can transform
	// case (5) into case (6) by transposing all arguments and swapping the
	// a and b operands. Also, we will need to pack matrix c. That is what
	// we do here.
	if ( bli_is_real( dt_targ_a ) && bli_is_complex( dt_targ_b ) )
	{
		bli_obj_swap( *a, *b );
		bli_swap_types( dt_targ_a, dt_targ_b );
		bli_obj_toggle_trans( *c );
		bli_obj_toggle_trans( *a );
		bli_obj_toggle_trans( *b );
	}

	// For now disable packing of C.
	*pack_c = FALSE;
}


void bli_gemm_get_target_datatypes( obj_t*  a,
                                    obj_t*  b,
                                    obj_t*  c,
                                    num_t*  dt_a,
                                    num_t*  dt_b,
                                    num_t*  dt_c,
                                    bool_t* pack_c )
{
	prec_t tp_a, tp_b, tp_c;
	dom_t  td_a, td_b, td_c;

	// Determine the target domains for each object.
	bli_gemm_get_target_domain( a,
	                            b,
	                            c,
	                            &td_a,
	                            &td_b,
	                            &td_c,
	                            pack_c );

	// Determine the target precisions for each object.
	bli_gemm_get_target_prec( a,
	                          b,
	                          c,
	                          &tp_a,
	                          &tp_b,
	                          &tp_c,
	                          pack_c );

	// The target datatype of an object is simply the union of its
	// target domain and target precision.
	*dt_a = td_a | tp_a;
	*dt_b = td_b | tp_b;
	*dt_c = td_c | tp_c;
}

void bli_gemm_get_target_domain( obj_t*  a,
                                 obj_t*  b,
                                 obj_t*  c,
                                 dom_t*  td_a,
                                 dom_t*  td_b,
                                 dom_t*  td_c,
                                 bool_t* pack_c )
/*
    input       target     packing
    domain      domain     needed
    c+=a*b      c+=a*b
    r  r r      r  r r     ab
    r  r c      r  r r     ab
    r  c r      r  r r     ab
    r  c c      c  c c     abc
    c  r r      r  r r     abc
    c  r c      c  r c     abc
    c  c r      c  c r     ab
    c  c c      c  c c     ab
*/
{
	num_t d_a = bli_obj_datatype( *a );
	num_t d_b = bli_obj_datatype( *b );
	num_t d_c = bli_obj_datatype( *c );

	if ( bli_is_real( d_c ) )
	{
		if ( bli_is_complex( d_a ) &&
		     bli_is_complex( d_b ) )
		{
			*td_c = *td_a = *td_b = BLIS_COMPLEX;
			*pack_c = TRUE;
		}
		else
		{
			*td_c = *td_a = *td_b = BLIS_REAL;
		}
	}
	else // if ( bli_is_complex( d_c ) )
	{
		*td_a = bli_domain_of_dt( d_a );
		*td_b = bli_domain_of_dt( d_b );

		if ( bli_is_real( d_a ) &&
		     bli_is_real( d_b ) )
		{
			*td_c = BLIS_REAL;
			*pack_c = TRUE;
		}
		else
		{
			*td_c = BLIS_COMPLEX;

			if ( bli_obj_is_real( *a ) &&
			     bli_obj_is_complex( *b ) ) *pack_c = TRUE;

			if ( bli_obj_is_complex( *a ) &&
			     bli_obj_is_real( *b ) )
			{
				if ( !bli_obj_is_col_stored( *c ) ) *pack_c = TRUE;
			}
		}
	}
}

void bli_gemm_get_target_prec( obj_t*  a,
                               obj_t*  b,
                               obj_t*  c,
                               prec_t* tp_a,
                               prec_t* tp_b,
                               prec_t* tp_c,
                               bool_t* pack_c )
/*
    input       target      packing
    precision   precision   needed
    c+=a*b      c+=a*b
    s  s s      s  s s      ab
    s  s d      s  s s      ab
    s  d s      s  s s      ab
    s  d d      d  d d      abc
    d  s s      s  s s      abc
    d  s d      d  d d      ab
    d  d s      d  d d      ab
    d  d d      d  d d      ab
*/
{
	num_t p_a = bli_obj_datatype( *a );
	num_t p_b = bli_obj_datatype( *b );
	num_t p_c = bli_obj_datatype( *c );

	if ( bli_is_single_prec( p_c ) )
	{
		if ( bli_is_double_prec( p_a ) &&
		     bli_is_double_prec( p_b ) )
		{
			*tp_c = *tp_a = *tp_b = BLIS_DOUBLE_PREC;
			*pack_c = TRUE;
		}
		else
		{
			*tp_c = *tp_a = *tp_b = BLIS_SINGLE_PREC;
		}
	}
	else // if ( bli_is_double_prec( p_c ) )
	{
		if ( bli_is_single_prec( p_a ) &&
		     bli_is_single_prec( p_b ) )
		{
			*tp_c = *tp_a = *tp_b = BLIS_SINGLE_PREC;
			*pack_c = TRUE;
		}
		else
		{
			*tp_c = *tp_a = *tp_b = BLIS_DOUBLE_PREC;
		}
	}
}

