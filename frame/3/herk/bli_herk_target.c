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

void bli_herk_set_targ_exec_datatypes( obj_t*  a,
                                       obj_t*  ah,
                                       obj_t*  c,
                                       bool_t* pack_c )
{
	num_t   dt_targ_a;
	num_t   dt_targ_ah;
	num_t   dt_targ_c;
	num_t   dt_exec;

	// Determine the target datatype of each matrix object.
/*
	bli_herk_get_target_datatypes( a,
	                               c,
	                               &dt_targ_a,
	                               &dt_targ_c,
	                               pack_c );
*/
	dt_targ_a  = bli_obj_datatype( *a );
	dt_targ_ah = bli_obj_datatype( *ah );
	dt_targ_c  = bli_obj_datatype( *c );
	dt_exec    = dt_targ_a;
	
	// Set the target datatypes for each matrix object.
	bli_obj_set_target_datatype( dt_targ_a,  *a );
	bli_obj_set_target_datatype( dt_targ_ah, *ah );
	bli_obj_set_target_datatype( dt_targ_c,  *c );

	// Embed the execution datatype in all matrix operands.
	bli_obj_set_execution_datatype( dt_exec, *a );
	bli_obj_set_execution_datatype( dt_exec, *ah );
	bli_obj_set_execution_datatype( dt_exec, *c );

	// For now disable packing of C.
	*pack_c = FALSE;
}

/*
void bli_herk_get_target_datatypes( obj_t*  a,
                                    obj_t*  c,
                                    num_t*  dt_a,
                                    num_t*  dt_c,
                                    bool_t* pack_c )
{
	prec_t tp_a, tp_c;
	dom_t  td_a, td_c;

	// Determine the target domains for each object.
	bli_herk_get_target_domain( a,
	                            c,
	                            &td_a,
	                            &td_c,
	                            pack_c );

	// Determine the target precisions for each object.
	bli_herk_get_target_prec( a,
	                          c,
	                          &tp_a,
	                          &tp_c,
	                          pack_c );

	// The target datatype of an object is simply the union of its
	// target domain and target precision.
	*dt_a = td_a | tp_a;
	*dt_c = td_c | tp_c;
}
*/
/*
void bli_herk_get_target_domain( obj_t*  a,
                                 obj_t*  c,
                                 dom_t*  td_a,
                                 dom_t*  td_c,
                                 bool_t* pack_c )
{
	dom_t d_a = bli_obj_domain( *a );
	dom_t d_c = bli_obj_domain( *c );

	// Note that the precisions of the target datatypes of a and c
	// match. The domains, however, are not necessarily the same. There
	// are four possible combinations of target domains:
	//
	//   case  input     target    exec    pack  notes  
	//         domain    domain    domain  c?      
	//         c+=a*a'   c+=a*a'     
	//   (0)   r  r r    r  r r    r              
	//   (1)   r  c c    c  c c    c       yes   a*a' demoted to real
	//   (2)   c  r r    r  r r    r       yes   copynzm used to update c
	//   (3)   c  c c    c  c c    c

	if ( bli_is_real( d_c ) )
	{
		if ( bli_is_complex( d_a ) )
		{
			*td_c = *td_a = BLIS_COMPLEX;
			*pack_c = TRUE;
		}
		else
		{
			*td_c = *td_a = BLIS_REAL;
		}
	}
	else // if ( bli_is_complex( d_c ) )
	{
		*td_a = d_a;

		if ( bli_is_real( d_a ) )
		{
			*td_c = BLIS_REAL;
			*pack_c = TRUE;
		}
		else
		{
			*td_c = BLIS_COMPLEX;

			if ( bli_obj_is_real( *a ) ) *pack_c = TRUE;

			if ( bli_obj_is_complex( *a ) )
			{
				if ( !bli_obj_is_col_stored( *c ) ) *pack_c = TRUE;
			}
		}
	}
}
*/

/*
void bli_herk_get_target_prec( obj_t*  a,
                               obj_t*  c,
                               prec_t* tp_a,
                               prec_t* tp_c,
                               bool_t* pack_c )
{
	prec_t p_a = bli_obj_precision( *a );
	prec_t p_c = bli_obj_precision( *c );

	if ( bli_is_single_prec( p_c ) )
	{
		if ( bli_is_double_prec( p_a ) )
		{
			*tp_c = *tp_a = BLIS_DOUBLE_PREC;
			*pack_c = TRUE;
		}
		else
		{
			*tp_c = *tp_a = BLIS_SINGLE_PREC;
		}
	}
	else // if ( bli_is_double_prec( p_c ) )
	{
		if ( bli_is_single_prec( p_a ) )
		{
			*tp_c = *tp_a = BLIS_SINGLE_PREC;
			*pack_c = TRUE;
		}
		else
		{
			*tp_c = *tp_a = BLIS_DOUBLE_PREC;
		}
	}
}
*/
