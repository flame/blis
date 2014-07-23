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

void bli_trmm_set_targ_exec_datatypes( obj_t*  a,
                                       obj_t*  b,
                                       obj_t*  c )
{
	num_t   dt_targ_a;
	num_t   dt_targ_b;
	num_t   dt_targ_c;
	num_t   dt_exec;

	dt_targ_a  = bli_obj_datatype( *a );
	dt_targ_b  = bli_obj_datatype( *b );
	dt_targ_c  = bli_obj_datatype( *c );
	dt_exec    = dt_targ_a;

	// Set the target datatypes for each matrix object.
	bli_obj_set_target_datatype( dt_targ_a, *a );
	bli_obj_set_target_datatype( dt_targ_b, *b );
	bli_obj_set_target_datatype( dt_targ_c, *c );

	// Embed the execution datatype in all matrix operands.
	bli_obj_set_execution_datatype( dt_exec, *a );
	bli_obj_set_execution_datatype( dt_exec, *b );
	bli_obj_set_execution_datatype( dt_exec, *c );
}

/*
void bli_trmm_get_target_datatypes( obj_t*  a,
                                    obj_t*  b,
                                    obj_t*  c,
                                    num_t*  dt_a,
                                    num_t*  dt_b,
                                    num_t*  dt_c,
                                    bool_t* pack_c )
{
}

void bli_trmm_get_target_domain( obj_t*  a,
                                 obj_t*  b,
                                 obj_t*  c,
                                 dom_t*  td_a,
                                 dom_t*  td_b,
                                 dom_t*  td_c,
                                 bool_t* pack_c )
{
}

void bli_trmm_get_target_prec( obj_t*  a,
                               obj_t*  b,
                               obj_t*  c,
                               prec_t* tp_a,
                               prec_t* tp_b,
                               prec_t* tp_c,
                               bool_t* pack_c )
{
}
*/
