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

void bl2_obj_create( num_t  dt,
                     dim_t  m,
                     dim_t  n,
                     inc_t  rs,
                     inc_t  cs,
                     obj_t* obj )
{
	bl2_obj_create_without_buffer( dt, m, n, obj );

	bl2_obj_alloc_buffer( rs, cs, obj );
}

void bl2_obj_create_with_attached_buffer( num_t  dt,
                                          dim_t  m,
                                          dim_t  n,
                                          void*  p,
                                          inc_t  rs,
                                          inc_t  cs,
                                          obj_t* obj )
{
	bl2_obj_create_without_buffer( dt, m, n, obj );

	bl2_obj_attach_buffer( p, rs, cs, obj );
}

void bl2_obj_create_without_buffer( num_t  dt,
                                    dim_t  m,
                                    dim_t  n,
                                    obj_t* obj )
{
	siz_t  elem_size;
	mem_t* pack_mem;
	mem_t* cast_mem;

	if ( bl2_error_checking_is_enabled() )
		bl2_obj_create_without_buffer_check( dt, m, n, obj );

	// Query the size of one element of the object's pre-set datatype.
	elem_size = bl2_datatype_size( dt );

	// Set any default properties that are appropriate.
	bl2_obj_set_defaults( *obj );

	// Set the object root to itself, since obj is not presumed to be a view
	// into a larger matrix. This is the ONLY time this field is ever set;
	// henceforth, subpartitions and aliases to this object will get copies
	// of this field, and thus always have access to its "greatest-grand"
	// parent (ie: the original parent, or "root", object).
	bl2_obj_set_as_root( *obj );

	// Set individual fields.
	bl2_obj_set_buffer( NULL, *obj );
	bl2_obj_set_datatype( dt, *obj );
	bl2_obj_set_elem_size( elem_size, *obj );
	bl2_obj_set_target_datatype( dt, *obj );
	bl2_obj_set_execution_datatype( dt, *obj );
	bl2_obj_set_dims( m, n, *obj );
	bl2_obj_set_offs( 0, 0, *obj );
	bl2_obj_set_diag_offset( 0, *obj );

	pack_mem = bl2_obj_pack_mem( *obj );
	cast_mem = bl2_obj_cast_mem( *obj );
	bl2_mem_set_buffer( NULL, pack_mem );
	bl2_mem_set_buffer( NULL, cast_mem );
}

void bl2_obj_alloc_buffer( inc_t  rs,
                           inc_t  cs,
                           obj_t* obj )
{
	dim_t  n_elem = 0;
	dim_t  m, n;
	siz_t  elem_size;
	siz_t  buffer_size;
	void*  p;

	m  = bl2_obj_length( *obj );
	n  = bl2_obj_width( *obj );

	// Adjust the strides, if needed, before doing anything else
	// (particularly, before doing any error checking).
	bl2_adjust_strides( m, n, &rs, &cs );

	if ( bl2_error_checking_is_enabled() )
		bl2_obj_alloc_buffer_check( rs, cs, obj );

	// Query the size of one element.
	elem_size = bl2_obj_elem_size( *obj );

	// Determine how much object to allocate.
	if ( rs == 1 )
	{
		cs     = bl2_align_dim_to_sys( cs, elem_size );
		n_elem = cs * n;
	}
	else if ( cs == 1 )
	{
		rs     = bl2_align_dim_to_sys( rs, elem_size );
		n_elem = rs * m;
	}
	else
	{
		if ( rs < cs )
		{
			// Note this case is identical to that of rs == 1 above.
			cs     = bl2_align_dim_to_sys( cs, elem_size );
			n_elem = cs * n;
		}
		else if ( cs < rs )
		{
			// Note this case is identical to that of cs == 1 above.
			rs     = bl2_align_dim_to_sys( rs, elem_size );
			n_elem = rs * m;
		}
		else
		{
			bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
		}
	}

	// Compute the size of the total buffer to be allocated, which includes
	// padding if the leading dimension was increased for alignment purposes.
	buffer_size = n_elem * elem_size;

	// Allocate the buffer.
	p = bl2_malloc( buffer_size );

	// Set individual fields.
	bl2_obj_set_buffer( p, *obj );
	bl2_obj_set_incs( rs, cs, *obj );
}

void bl2_obj_attach_buffer( void*  p,
                            inc_t  rs,
                            inc_t  cs,
                            obj_t* obj )
{
	dim_t  m, n;

	m = bl2_obj_length( *obj );
	n = bl2_obj_width( *obj );

	// Adjust the strides, if necessary.
	bl2_adjust_strides( m, n, &rs, &cs );

	// Notice that we wait until after strides have been adjusted to
	// error-check.
	if ( bl2_error_checking_is_enabled() )
		bl2_obj_attach_buffer_check( p, rs, cs, obj );

	// Update the object.
	bl2_obj_set_buffer( p, *obj );
	bl2_obj_set_incs( rs, cs, *obj );
}

void bl2_obj_attach_internal_buffer( obj_t* obj )
{
	void* p;

	// Query the address of the object's internal scalar buffer.
	p = bl2_obj_internal_scalar_buffer( *obj );

	// Update the object.
	bl2_obj_set_buffer( p, *obj );
	bl2_obj_set_incs( 1, 1, *obj );
}

void bl2_obj_init_scalar( num_t  dt,
                          obj_t* b )
{
	// Initialize b without a buffer and then attach its internal buffer.
	bl2_obj_create_without_buffer( dt, 1, 1, b );
	bl2_obj_attach_internal_buffer( b );
}

void bl2_obj_init_scalar_copy_of( num_t  dt,
                                  conj_t conj,
                                  obj_t* a,
                                  obj_t* b )
{
	obj_t a_local;

	// Make a local copy of scalar a so we can apply the conj parameter.
	bl2_obj_alias_to( *a, a_local );
	bl2_obj_apply_conj( conj, a_local );

	// Initialize b without a buffer and then attach its internal buffer.
	bl2_obj_create_without_buffer( dt, 1, 1, b );
	bl2_obj_attach_internal_buffer( b );

	// Copy the scalar value in a to object b, conjugating if needed.
	bl2_copysc( &a_local, b );
}

void bl2_obj_create_scalar( num_t  dt,
                            obj_t* obj )
{
	bl2_obj_create_without_buffer( dt, 1, 1, obj );

	bl2_obj_alloc_buffer( 1, 1, obj );
}

void bl2_obj_create_scalar_with_attached_buffer( num_t  dt,
                                                 void*  p,
                                                 obj_t* obj )
{
	bl2_obj_create_without_buffer( dt, 1, 1, obj );

	bl2_obj_attach_buffer( p, 1, 1, obj );
}

void bl2_obj_free( obj_t* obj )
{
	if ( bl2_error_checking_is_enabled() )
		bl2_obj_free_check( obj );

	// Don't dereference obj if it is NULL.
	if ( obj != NULL )
	{
		// Idiot safety: Don't try to free the buffer field if it currently
		// refers to the internal scalar buffer.
		if ( bl2_obj_buffer( *obj ) != bl2_obj_internal_scalar_buffer( *obj ) )
			bl2_free( bl2_obj_buffer( *obj ) );
	}
}

void bl2_obj_create_const( double value, obj_t* obj )
{
	int*      temp_i;
	float*    temp_s;
	double*   temp_d;
	scomplex* temp_c;
	dcomplex* temp_z;

	if ( bl2_error_checking_is_enabled() )
		bl2_obj_create_const_check( value, obj );

	bl2_obj_create( BLIS_CONSTANT, 1, 1, 1, 1, obj );

	temp_i       = BLIS_CONST_I_PTR( *obj );
	temp_s       = BLIS_CONST_S_PTR( *obj );
	temp_d       = BLIS_CONST_D_PTR( *obj );
	temp_c       = BLIS_CONST_C_PTR( *obj );
	temp_z       = BLIS_CONST_Z_PTR( *obj );

	*temp_i      = ( int   ) value;
	*temp_s      = ( float ) value;
	*temp_d      =           value;
	temp_c->real = ( float ) value;
	temp_c->imag = ( float ) 0.0;
	temp_z->real =           value;
	temp_z->imag =           0.0;
}

void bl2_obj_create_const_copy_of( obj_t* a, obj_t* b )
{
	int*      temp_i;
	float*    temp_s;
	double*   temp_d;
	scomplex* temp_c;
	dcomplex* temp_z;
	dcomplex  value;

	if ( bl2_error_checking_is_enabled() )
		bl2_obj_create_const_copy_of_check( a, b );

	bl2_obj_create( BLIS_CONSTANT, 1, 1, 1, 1, b );

	temp_i       = BLIS_CONST_I_PTR( *b );
	temp_s       = BLIS_CONST_S_PTR( *b );
	temp_d       = BLIS_CONST_D_PTR( *b );
	temp_c       = BLIS_CONST_C_PTR( *b );
	temp_z       = BLIS_CONST_Z_PTR( *b );

	value.real = 0.0;
	value.imag = 0.0;

	if ( bl2_obj_is_float( *a ) )
	{
		value.real = *(( float*  )( bl2_obj_buffer_at_off( *a ) ));
		value.imag = 0.0;
	}
	else if ( bl2_obj_is_double( *a ) )
	{
		value.real = *(( double* )( bl2_obj_buffer_at_off( *a ) ));
		value.imag = 0.0;
	}
	else if ( bl2_obj_is_scomplex( *a ) )
	{
		value.real =  (( scomplex* )( bl2_obj_buffer_at_off( *a ) ))->real;
		value.imag =  (( scomplex* )( bl2_obj_buffer_at_off( *a ) ))->imag;
	}
	else if ( bl2_obj_is_dcomplex( *a ) )
	{
		value.real =  (( dcomplex* )( bl2_obj_buffer_at_off( *a ) ))->real;
		value.imag =  (( dcomplex* )( bl2_obj_buffer_at_off( *a ) ))->imag;
	}
	else
	{
		bl2_abort();
	}

	*temp_i      = ( int   ) value.real;
	*temp_s      = ( float ) value.real;
	*temp_d      =           value.real;
	temp_c->real = ( float ) value.real;
	temp_c->imag = ( float ) value.imag;
	temp_z->real =           value.real;
	temp_z->imag =           value.imag;
}

void bl2_adjust_strides( dim_t  m,
                         dim_t  n,
                         dim_t* rs,
                         dim_t* cs )
{
	// Here, we check the strides that were input from the user and modify
	// them if needed.

	// Interpret rs = cs = 0 as request for column storage.
	if ( *rs == 0 && *cs == 0 )
	{
		// We use column-major storage, except when m == 1, because we don't
		// want both strides to be unit.
		if ( m == 1 && n > 1 )
		{
			*rs = n;
			*cs = 1;
		}
		else
		{
			*rs = 1;
			*cs = m;
		}
	}
	else if ( *rs == 1 && *cs == 1 )
	{
		// If both strides are unit, this is probably a "lazy" request for a
		// single vector (but could also be a request for a 1xn matrix in
		// column-major order or an mx1 matrix in row-major order). In BLIS,
		// we have decided to "reserve" the case where rs = cs = 1 for
		// scalars only.
		if ( m > 1 && n == 1 )
		{
			// Set the column stride to indicate that this is a column vector
			// stored in column-major order. This is done for legacy reasons,
			// because we at one time we had to satisify the error checking
			// in the underlying BLAS library, which expects the leading 
			// dimension to be set to at least m, even if it will never be
			// used for indexing since it is a vector and thus only has one
			// column of data.
			*cs = m;
		}
		else if ( m == 1 && n > 1 )
		{
			// Set the row stride to indicate that this is a row vector stored
			// in row-major order.
			*rs = n;
		}

		// Nothing needs to be done for the scalar case where m == n == 1.
	}
}

static siz_t dt_sizes[6] =
{
	sizeof( float ),
	sizeof( scomplex ),
	sizeof( double ),
	sizeof( dcomplex ),
	sizeof( int ),
	BLIS_CONSTANT_SIZE
};

siz_t bl2_datatype_size( num_t dt )
{
	if ( bl2_error_checking_is_enabled() )
		bl2_datatype_size_check( dt );

	return dt_sizes[dt];
}

dim_t bl2_align_dim_to_mult( dim_t dim, dim_t dim_mult )
{
	dim = ( ( dim + dim_mult - 1 ) /
	        dim_mult ) *
	        dim_mult;

	return dim;
}

dim_t bl2_align_dim_to_sys( dim_t dim, dim_t elem_size )
{
	dim = ( ( dim * elem_size + BLIS_MEMORY_ALIGNMENT_BOUNDARY - 1 ) /
	        BLIS_MEMORY_ALIGNMENT_BOUNDARY ) *
	        BLIS_MEMORY_ALIGNMENT_BOUNDARY /
	        elem_size;

	return dim;
}

dim_t bl2_align_dim( dim_t dim, dim_t dim_mult, dim_t elem_size )
{
	dim = bl2_align_dim_to_mult( dim, dim_mult );
	dim = bl2_align_dim_to_sys( dim, elem_size );

	return dim;
}

static num_t type_union[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] =
{
            // s             c              d              z
	/* s */ { BLIS_FLOAT,    BLIS_SCOMPLEX, BLIS_DOUBLE,   BLIS_DCOMPLEX },
	/* c */ { BLIS_SCOMPLEX, BLIS_SCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX },
	/* d */ { BLIS_DOUBLE,   BLIS_DCOMPLEX, BLIS_DOUBLE,   BLIS_DCOMPLEX },
	/* z */ { BLIS_DCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX }
};

num_t bl2_datatype_union( num_t dt1, num_t dt2 )
{
	if ( bl2_error_checking_is_enabled() )
		bl2_datatype_union_check( dt1, dt2 );

	return type_union[dt1][dt2];
}

void bl2_obj_print( char* label, obj_t* obj )
{
	FILE*  file     = stdout;
	mem_t* pack_mem = bl2_obj_pack_mem( *obj );
	mem_t* cast_mem = bl2_obj_cast_mem( *obj );

	if ( bl2_error_checking_is_enabled() )
		bl2_obj_print_check( label, obj );

	fprintf( file, "\n" );
	fprintf( file, "%s\n", label );
	fprintf( file, "\n" );

	fprintf( file, " m x n           %lu x %lu\n", bl2_obj_length( *obj ),
	                                               bl2_obj_width( *obj ) );
	fprintf( file, "\n" );

	fprintf( file, " offm, offn      %lu, %lu\n", bl2_obj_row_offset( *obj ),
	                                              bl2_obj_col_offset( *obj ) );
	fprintf( file, " diagoff         %ld\n", bl2_obj_diag_offset( *obj ) );
	fprintf( file, "\n" );

	fprintf( file, " buf             %p\n",  bl2_obj_buffer( *obj ) );
	fprintf( file, " elem size       %lu\n", bl2_obj_elem_size( *obj ) );
	fprintf( file, " rs, cs          %lu, %lu\n", bl2_obj_row_stride( *obj ),
	                                              bl2_obj_col_stride( *obj ) );
	fprintf( file, " cast_mem          \n" );
	fprintf( file, " - buf           %p\n",  bl2_mem_buffer( cast_mem ) );
	fprintf( file, " - m             %lu\n", bl2_mem_length( cast_mem ) );
	fprintf( file, " - n             %lu\n", bl2_mem_width( cast_mem ) );
	fprintf( file, " pack_mem          \n" );
	fprintf( file, " - buf           %p\n",  bl2_mem_buffer( pack_mem ) );
	fprintf( file, " - m             %lu\n", bl2_mem_length( pack_mem ) );
	fprintf( file, " - n             %lu\n", bl2_mem_width( pack_mem ) );
	fprintf( file, " ps              %lu\n", bl2_obj_panel_stride( *obj ) );
	fprintf( file, "\n" );

	fprintf( file, " info            %lX\n", (*obj).info );
	fprintf( file, " - is complex    %u\n",  bl2_obj_is_complex( *obj ) );
	fprintf( file, " - is d. prec    %u\n",  bl2_obj_is_double_precision( *obj ) );
	fprintf( file, " - has trans     %u\n",  bl2_obj_has_trans( *obj ) );
	fprintf( file, " - has conj      %u\n",  bl2_obj_has_conj( *obj ) );
	fprintf( file, " - struc type    %lu\n", bl2_obj_struc( *obj ) );
	fprintf( file, " - uplo type     %lu\n", bl2_obj_uplo( *obj ) );
	fprintf( file, "   - is upper    %u\n",  bl2_obj_is_upper( *obj ) );
	fprintf( file, "   - is lower    %u\n",  bl2_obj_is_lower( *obj ) );
	fprintf( file, "   - is dense    %u\n",  bl2_obj_is_dense( *obj ) );
	fprintf( file, " - datatype      %lu\n", bl2_obj_datatype( *obj ) );
	fprintf( file, " - target dt     %lu\n", bl2_obj_target_datatype( *obj ) );
	fprintf( file, " - exec dt       %lu\n", bl2_obj_execution_datatype( *obj ) );
	fprintf( file, "\n" );
}

