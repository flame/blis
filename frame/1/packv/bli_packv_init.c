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

void bli_packv_init( obj_t*   a,
                     obj_t*   p,
                     packv_t* cntl )
{
	// The purpose of packm_init() is to initialize an object P so that
	// a source object A can be packed into P via one of the packv
	// implementations. This initialization includes acquiring a suitable
	// block of memory from the memory allocator, if such a block of memory
	// has not already been allocated previously.

	pack_t   pack_schema;
	blksz_t* mult_m;
	obj_t    c;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packv_check( a, p, cntl );

	// First check if we are to skip this operation because the control tree
	// is NULL, and if so, simply alias the object to its packed counterpart.
	if ( cntl_is_noop( cntl ) )
	{
		bli_obj_alias_to( *a, *p );
		return;
	}

	// At this point, we can be assured that cntl is not NULL. Let us now
	// check to see if the object has already been packed to the desired
	// schema (as encoded in the control tree). If so, we can alias and
	// return, as above.
	// Note that in most cases, bli_obj_pack_schema() will return
	// BLIS_NOT_PACKED and thus packing will be called for (but in some
	// cases packing has already taken place). Also, not all combinations
	// of current pack status and desired pack schema are valid.
	if ( bli_obj_pack_schema( *a ) == cntl_pack_schema( cntl ) )
	{
		bli_obj_alias_to( *a, *p );
		return;
	}

	// Now, if we are not skipping the pack operation, then the only question
	// left is whether we are to typecast vector a before packing.
	if ( bli_obj_datatype( *a ) != bli_obj_target_datatype( *a ) )
		bli_abort();
/*
	{
		// Initialize an object c for the intermediate typecast vector.
		bli_packv_init_cast( a,
		                     p,
		                     &c );

		// Copy/typecast vector a to vector c.
		bli_copyv( a,
		           &c );
	}
	else
*/
	{
		// If no cast is needed, then aliasing object c to the original
		// vector serves as a minor optimization. This causes the packv
		// implementation to pack directly from vector a.
		bli_obj_alias_to( *a, c );
	}


	// Extract various fields from the control tree and pass them in
	// explicitly into _init_pack(). This allows external code generators
	// the option of bypassing usage of control trees altogether.
	pack_schema = cntl_pack_schema( cntl );
	mult_m      = cntl_mult_dim( cntl );

	// Initialize object p for the final packed vector.
	bli_packv_init_pack( pack_schema,
	                     mult_m,
	                     &c,
	                     p );

	// Now p is ready to be packed.
}


void bli_packv_init_pack( pack_t   pack_schema,
                          blksz_t* mult_m,
                          obj_t*   c,
                          obj_t*   p )
{
	num_t  datatype     = bli_obj_datatype( *c );
	dim_t  dim_c        = bli_obj_vector_dim( *c );
	dim_t  mult_m_dim   = bli_blksz_for_type( datatype, mult_m );

	mem_t* mem_p;
	dim_t  m_p_pad;
	siz_t  size_p;
	inc_t  rs_p, cs_p;
	void*  buf;


	// We begin by copying the basic fields of c.
	bli_obj_alias_to( *c, *p );

	// Update the dimensions.
	bli_obj_set_dims( dim_c, 1, *p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, *p );

	// Set the pack schema in the p object to the value in the control tree
	// node.
	bli_obj_set_pack_schema( pack_schema, *p );

	// Extract the address of the mem_t object within p that will track
	// properties of the packed buffer.
	mem_p = bli_obj_pack_mem( *p );

	// Compute the dimensions padded by the dimension multiples.
	m_p_pad = bli_align_dim_to_mult( bli_obj_vector_dim( *p ), mult_m_dim );

	// Compute the size of the packed buffer.
	size_p = m_p_pad * 1 * bli_obj_elem_size( *p );

	if ( bli_mem_is_unalloc( mem_p ) )
	{
		// If the mem_t object of p has not yet been allocated, then acquire
		// a memory block suitable for a vector.
		bli_mem_acquire_v( size_p,
		                   mem_p );
	}
	else
	{
 		// If the mem_t object has already been allocated, then release and
		// re-acquire the memory so there is sufficient space.
		if ( bli_mem_size( mem_p ) < size_p )
		{
			bli_mem_release( mem_p ); 

			bli_mem_acquire_v( size_p,
			                   mem_p );
		}
	}

	// Save the padded (packed) dimensions into the packed object.
	bli_obj_set_padded_dims( m_p_pad, 1, *p );

	// Grab the buffer address from the mem_t object and copy it to the
	// main object buffer field. (Sometimes this buffer address will be
	// copied when the value is already up-to-date, because it persists
	// in the main object buffer field across loop iterations.)
	buf = bli_mem_buffer( mem_p );
	bli_obj_set_buffer( buf, *p );


	// Set the row and column strides of p based on the pack schema.
	if ( pack_schema == BLIS_PACKED_VECTOR )
	{
		// Set the strides to reflect a column-stored vector. Note that the
		// column stride may never be used, and is only useful to determine
		// how much space beyond the vector would need to be zero-padded, if
		// zero-padding was needed.
		rs_p = 1;
		cs_p = bli_obj_padded_length( *p );

		bli_obj_set_incs( rs_p, cs_p, *p );
	}
}


/*
void bli_packv_init_cast( obj_t*  a,
                          obj_t*  p,
                          obj_t*  c )
{
	// The idea here is that we want to create an object c that is identical
	// to object a, except that:
	//  (1) the storage datatype of c is equal to the target datatype of a,
	//      with the element size of c adjusted accordingly,
	//  (2) object c is marked as being stored in a standard, contiguous
	//      format (ie: a column vector),
	//  (3) the view offset of c is reset to (0,0), and
	//  (4) object c's main buffer is set to a new memory region acquired
	//      from the memory manager, or extracted from p if a mem entry is
	//      already available. (After acquring a mem entry from the memory
	//      manager, it is cached within p for quick access later on.)

	num_t dt_targ_a    = bli_obj_target_datatype( *a );
	dim_t dim_a        = bli_obj_vector_dim( *a );
	siz_t elem_size_c  = bli_datatype_size( dt_targ_a );

	// We begin by copying the basic fields of a.
	bli_obj_alias_to( *a, *c );

	// Update datatype and element size fields.
	bli_obj_set_datatype( dt_targ_a, *c );
	bli_obj_set_elem_size( elem_size_c, *c );

	// Update the dimensions.
	bli_obj_set_dims( dim_a, 1, *c );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, *c );

	// Check the mem_t entry of p associated with the cast buffer. If it is
	// NULL, then acquire memory sufficient to hold the object data and cache
	// it to p. (Otherwise, if it is non-NULL, then memory has already been
	// acquired from the memory manager and cached.) We then set the main
	// buffer of c to the cached address of the cast memory.
	bli_obj_set_buffer_with_cached_cast_mem( *p, *c );

	// Update the strides. We set the increments to reflect a column storage.
	// Note that the column stride should never be used.
	bli_obj_set_incs( 1, dim_a, *c );
}
*/

