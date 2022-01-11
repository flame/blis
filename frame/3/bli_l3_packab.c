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
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

void bli_l3_packa
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	obj_t a_local, a_pack;

	bli_obj_alias_to( a, &a_local );
	if ( bli_obj_has_trans( a ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

	// Pack matrix A according to the control tree node.
	bli_packm_int
	(
	  &a_local,
	  &a_pack,
	  cntx,
	  rntm,
	  cntl,
	  thread
	);

	// Proceed with execution using packed matrix A.
	bli_l3_int
	(
	  &BLIS_ONE,
	  &a_pack,
	  b,
	  &BLIS_ONE,
	  c,
	  cntx,
	  rntm,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);
}

// -----------------------------------------------------------------------------

void bli_l3_packb
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	obj_t bt_local, bt_pack;

	// We always pass B^T to bli_l3_packm.
	bli_obj_alias_to( b, &bt_local );
	if ( bli_obj_has_trans( b ) )
	{
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &bt_local );
	}
	else
	{
		bli_obj_induce_trans( &bt_local );
	}

	// Pack matrix B according to the control tree node.
	bli_packm_int
	(
	  &bt_local,
	  &bt_pack,
	  cntx,
	  rntm,
	  cntl,
	  thread
	);

	// Transpose packed object back to B.
	bli_obj_induce_trans( &bt_pack );

	// Proceed with execution using packed matrix B.
	bli_l3_int
	(
	  &BLIS_ONE,
	  a,
	  &bt_pack,
	  &BLIS_ONE,
	  c,
	  cntx,
	  rntm,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);
}

