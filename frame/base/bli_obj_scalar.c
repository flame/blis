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


void bli_obj_scalar_init_detached( num_t  dt,
                                   obj_t* beta )
{
	void* p;

	// Initialize beta without a buffer and then attach its internal buffer.
	bli_obj_create_without_buffer( dt, 1, 1, beta );

	// Query the address of the object's internal scalar buffer.
	p = bli_obj_internal_scalar_buffer( *beta );

	// Update the object.
	bli_obj_set_buffer( p, *beta );
	bli_obj_set_incs( 1, 1, *beta );
}

void bli_obj_scalar_init_detached_copy_of( num_t  dt,
                                           conj_t conj,
                                           obj_t* alpha,
                                           obj_t* beta )
{
	obj_t alpha_local;

	// Make a local copy of alpha so we can apply the conj parameter.
	bli_obj_alias_to( *alpha, alpha_local );
	bli_obj_apply_conj( conj, alpha_local );

	// Initialize beta without a buffer and then attach its internal buffer.
	bli_obj_scalar_init_detached( dt, beta );

	// Copy the scalar value in a to object b, conjugating and/or
	// typecasting if needed.
	bli_copysc( &alpha_local, beta );
}

void bli_obj_scalar_detach( obj_t* a,
                            obj_t* alpha )
{
	num_t dt_a = bli_obj_datatype( *a );

	// Initialize alpha to be a bufferless internal scalar of the same
	// datatype as A.
	bli_obj_scalar_init_detached( dt_a, alpha );

	// Copy the internal scalar in A to alpha.
	bli_obj_copy_internal_scalar( *a, *alpha );
}

void bli_obj_scalar_attach( conj_t conj,
                            obj_t* alpha,
                            obj_t* a )
{
	obj_t alpha_cast;

	// Make a copy-cast of alpha of the same datatype as A. This step
	// gives us the opportunity to conjugate and/or typecast alpha.
	bli_obj_scalar_init_detached_copy_of( bli_obj_datatype( *a ),
	                                      conj,
	                                      alpha,
	                                      &alpha_cast );

	// Copy the internal scalar in alpha_cast to A.
	bli_obj_copy_internal_scalar( alpha_cast, *a );
}

void bli_obj_scalar_apply_scalar( obj_t* alpha,
                                  obj_t* a )
{
	obj_t alpha_cast;
	obj_t scalar_a;

	// Make a copy-cast of alpha of the same datatype as A. This step
	// gives us the opportunity to typecast alpha.
	bli_obj_scalar_init_detached_copy_of( bli_obj_datatype( *a ),
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_cast );
	// Detach the scalar from A.
	bli_obj_scalar_detach( a, &scalar_a );

	// Scale the detached scalar by alpha.
	bli_mulsc( &alpha_cast, &scalar_a );

	// Copy the internal scalar in scalar_a to A.
	bli_obj_copy_internal_scalar( scalar_a, *a );
}

void bli_obj_scalar_reset( obj_t* a )
{
	num_t dt       = bli_obj_datatype( *a );
	void* scalar_a = bli_obj_internal_scalar_buffer( *a );
	void* one      = bli_obj_buffer_for_const( dt, BLIS_ONE );

	if      ( bli_is_float( dt )    ) *(( float*    )scalar_a) = *(( float*    )one);
	else if ( bli_is_double( dt )   ) *(( double*   )scalar_a) = *(( double*   )one);
	else if ( bli_is_scomplex( dt ) ) *(( scomplex* )scalar_a) = *(( scomplex* )one);
	else if ( bli_is_dcomplex( dt ) ) *(( dcomplex* )scalar_a) = *(( dcomplex* )one);

	// Alternate implementation:
	//bli_obj_scalar_attach( BLIS_NO_CONJUGATE, &BLIS_ONE, a );
}

bool_t bli_obj_scalar_has_nonzero_imag( obj_t* a )
{
	bool_t r_val     = FALSE;
	num_t  dt        = bli_obj_datatype( *a );
	void*  scalar_a  = bli_obj_internal_scalar_buffer( *a );

	if      ( bli_is_real( dt ) )
	{
		r_val = FALSE;
	}
	else if ( bli_is_scomplex( dt ) )
	{
		r_val = ( bli_cimag( *(( scomplex* )scalar_a) ) != 0.0F );
	}
	else if ( bli_is_dcomplex( dt ) )
	{
		r_val = ( bli_zimag( *(( dcomplex* )scalar_a) ) != 0.0  );
	}

	return r_val;
}

bool_t bli_obj_scalar_equals( obj_t* a,
                              obj_t* beta )
{
	obj_t  scalar_a;
	bool_t r_val;

	bli_obj_scalar_detach( a, &scalar_a );
	
	r_val = bli_obj_equals( &scalar_a, beta );

	return r_val;
}

