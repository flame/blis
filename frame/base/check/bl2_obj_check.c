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

void bl2_obj_create_check( num_t  dt,
                           dim_t  m,
                           dim_t  n,
                           inc_t  rs,
                           inc_t  cs,
                           obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_valid_datatype( dt );
	bl2_check_error_code( e_val );

	e_val = bl2_check_matrix_strides( m, n, rs, cs );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_create_without_buffer_check( num_t  dt,
                                          dim_t  m,
                                          dim_t  n,
                                          obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_valid_datatype( dt );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_alloc_buffer_check( inc_t  rs,
                                 inc_t  cs,
                                 obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_matrix_strides( bl2_obj_length( *obj ),
	                                  bl2_obj_width( *obj ),
	                                  rs, cs );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_attach_buffer_check( void*  p,
                                  inc_t  rs,
                                  inc_t  cs,
                                  obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_null_pointer( p );
	bl2_check_error_code( e_val );

	e_val = bl2_check_matrix_strides( bl2_obj_length( *obj ),
	                                  bl2_obj_width( *obj ),
	                                  rs, cs );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_create_scalar_check( num_t  dt,
                                  obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_valid_datatype( dt );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_free_check( obj_t* obj )
{
	//err_t e_val;

	// We don't bother checking for null-ness since bl2_obj_free()
	// handles null pointers safely.
	//e_val = bl2_check_null_pointer( obj );
	//bl2_check_error_code( e_val );
}

void bl2_obj_create_const_check( double value, obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

void bl2_obj_create_const_copy_of_check( obj_t* a, obj_t* b )
{
	err_t e_val;

	e_val = bl2_check_null_pointer( a );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( b );
	bl2_check_error_code( e_val );

	e_val = bl2_check_scalar_object( a );
	bl2_check_error_code( e_val );
}

void bl2_datatype_size_check( num_t dt )
{
	err_t e_val;

	e_val = bl2_check_valid_datatype( dt );
	bl2_check_error_code( e_val );
}

void bl2_datatype_union_check( num_t dt1, num_t dt2 )
{
	err_t e_val;

	e_val = bl2_check_floating_datatype( dt1 );
	bl2_check_error_code( e_val );

	e_val = bl2_check_floating_datatype( dt2 );
	bl2_check_error_code( e_val );
}

void bl2_obj_print_check( char* label, obj_t* obj )
{
	err_t e_val;

	e_val = bl2_check_null_pointer( label );
	bl2_check_error_code( e_val );

	e_val = bl2_check_null_pointer( obj );
	bl2_check_error_code( e_val );
}

