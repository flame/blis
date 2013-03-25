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
	dom_t d_a = bli_obj_domain( *a );
	dom_t d_b = bli_obj_domain( *b );
	dom_t d_c = bli_obj_domain( *c );

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
		*td_a = d_a;
		*td_b = d_b;

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
	prec_t p_a = bli_obj_precision( *a );
	prec_t p_b = bli_obj_precision( *b );
	prec_t p_c = bli_obj_precision( *c );

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

