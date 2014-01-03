/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

void bli_packm_init( obj_t*   a,
                     obj_t*   p,
                     packm_t* cntl )
{
	// The purpose of packm_init() is to initialize an object P so that
	// a source object A can be packed into P via one of the packm
	// implementations. This initialization includes acquiring a suitable
	// block of memory from the memory allocator, if such a block of memory
	// has not already been allocated previously.

	bool_t    needs_densify;
	invdiag_t invert_diag;
	pack_t    pack_schema;
	packord_t pack_ord_if_up;
	packord_t pack_ord_if_lo;
	packbuf_t pack_buf_type;
	blksz_t*  mr;
	blksz_t*  nr;
	obj_t     c;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packm_init_check( a, p, cntl );

	// First check if we are to skip this operation because the control tree
	// is NULL, and if so, simply alias the object to its packed counterpart.
	if ( cntl_is_noop( cntl ) )
	{
		bli_obj_alias_to( *a, *p );
		return;
	}

	// Let us now check to see if the object has already been packed. First
	// we check if it has been packed to an unspecified (row or column)
	// format, in which case we can alias the object and return.
	// NOTE: The reason we don't need to even look at the control tree in
	// this case is as follows: an object's pack status is only set to
	// BLIS_PACKED_UNSPEC for situations when the actual format used is
	// not important, as long as its packed into contiguous rows or
	// contiguous columns. A good example of this is packing for matrix
	// operands in the level-2 operations.
	if ( bli_obj_pack_status( *a ) == BLIS_PACKED_UNSPEC )
	{
		bli_obj_alias_to( *a, *p );
		return;
	}

	// At this point, we can be assured that cntl is not NULL. Now we check
	// if the object has already been packed to the desired schema (as en-
	// coded in the control tree). If so, we can alias and return, as above.
	// NOTE: In most cases, an object's pack status will be BLIS_NOT_PACKED
	// and thus packing will be called for (but in some cases packing has
	// already taken place, or does not need to take place, and so that will
	// be indicated by the pack status). Also, not all combinations of
	// current pack status and desired pack schema are valid.
	if ( bli_obj_pack_status( *a ) == cntl_pack_schema( cntl ) )
	{
		bli_obj_alias_to( *a, *p );
		return;
	}

	// Now, if we are not skipping the pack operation, then the only question
	// left is whether we are to typecast matrix a before packing.
	if ( bli_obj_datatype( *a ) != bli_obj_target_datatype( *a ) )
		bli_abort();
/*
	{
		// Initialize an object c for the intermediate typecast matrix.
		bli_packm_init_cast( a,
		                     p,
		                     &c );

		// Copy/typecast matrix a to matrix c.
		bli_copym( a,
		           &c );
	}
	else
*/
	{
		// If no cast is needed, then aliasing object c to the original
		// matrix serves as a minor optimization. This causes the packm
		// implementation to pack directly from matrix a.
		bli_obj_alias_to( *a, c );
	}

	
	// Extract various fields from the control tree and pass them in
	// explicitly into _init_pack(). This allows external code generators
	// the option of bypassing usage of control trees altogether.
	needs_densify  = cntl_does_densify( cntl );
	pack_schema    = cntl_pack_schema( cntl );
	pack_buf_type  = cntl_pack_buf_type( cntl );
	mr             = cntl_mr( cntl );
	nr             = cntl_nr( cntl );

	if ( cntl_does_invert_diag( cntl ) ) invert_diag = BLIS_INVERT_DIAG;
	else                                 invert_diag = BLIS_NO_INVERT_DIAG;

	if ( cntl_rev_iter_if_upper( cntl ) ) pack_ord_if_up = BLIS_PACK_REV_IF_UPPER;
	else                                  pack_ord_if_up = BLIS_PACK_FWD_IF_UPPER;

	if ( cntl_rev_iter_if_lower( cntl ) ) pack_ord_if_lo = BLIS_PACK_REV_IF_LOWER;
	else                                  pack_ord_if_lo = BLIS_PACK_FWD_IF_LOWER;

	// Initialize object p for the final packed matrix.
	bli_packm_init_pack( needs_densify,
	                     invert_diag,
	                     pack_schema,
	                     pack_ord_if_up,
	                     pack_ord_if_lo,
	                     pack_buf_type,
	                     mr,
	                     nr,
	                     &c,
	                     p );

	// Now p is ready to be packed.
}


void bli_packm_init_pack( bool_t    densify,
                          invdiag_t invert_diag,
                          pack_t    pack_schema,
                          packord_t pack_ord_if_up,
                          packord_t pack_ord_if_lo,
                          packbuf_t pack_buf_type,
                          blksz_t*  mr,
                          blksz_t*  nr,
                          obj_t*    c,
                          obj_t*    p )
{
	num_t   datatype     = bli_obj_datatype( *c );
	trans_t transc       = bli_obj_onlytrans_status( *c );
	dim_t   m_c          = bli_obj_length( *c );
	dim_t   n_c          = bli_obj_width( *c );
	dim_t   mr_def_dim   = bli_blksz_for_type( datatype, mr );
	dim_t   mr_ext_dim   = bli_blksz_ext_for_type( datatype, mr );
	dim_t   nr_def_dim   = bli_blksz_for_type( datatype, nr );
	dim_t   nr_ext_dim   = bli_blksz_ext_for_type( datatype, nr );

	dim_t   mr_pack_dim  = mr_def_dim + mr_ext_dim;
	dim_t   nr_pack_dim  = nr_def_dim + nr_ext_dim;

	mem_t*  mem_p;
	dim_t   m_p_pad, n_p_pad;
	siz_t   size_p;
	siz_t   elem_size_p;
	inc_t   rs_p, cs_p;
	void*   buf;


	// We begin by copying the basic fields of c. We do NOT copy the
	// pack_mem entry from c because the entry in p may be cached from
	// a previous iteration, and thus we don't want to overwrite it.
	bli_obj_alias_for_packing( *c, *p );

	// Update the dimension fields to explicitly reflect a transposition,
	// if needed.
	// Then, clear the conjugation and transposition fields from the object
	// since matrix packing in BLIS is deemed to take care of all conjugation
	// and transposition necessary.
	// Then, we adjust the properties of p when c needs a transposition.
	// We negate the diagonal offset, and if c is upper- or lower-stored,
	// we either toggle the uplo of p.
	// Finally, if we are going to densify c, we mark p as dense.
	bli_obj_set_dims_with_trans( transc, m_c, n_c, *p );
	bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, *p );
	if ( bli_does_trans( transc ) )
	{
		bli_obj_negate_diag_offset( *p );
		if ( bli_obj_is_upper_or_lower( *c ) )
			bli_obj_toggle_uplo( *p );
	}
	if ( densify ) bli_obj_set_uplo( BLIS_DENSE, *p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, *p );

	// Set the invert diagonal field.
	bli_obj_set_invert_diag( invert_diag, *p );

	// Set the pack status of p to the pack schema prescribed in the control
	// tree node.
	bli_obj_set_pack_schema( pack_schema, *p );

	// Set the packing order bits.
	bli_obj_set_pack_order_if_upper( pack_ord_if_up, *p );
	bli_obj_set_pack_order_if_lower( pack_ord_if_lo, *p );

	// Extract the address of the mem_t object within p that will track
	// properties of the packed buffer.
	mem_p = bli_obj_pack_mem( *p );

	// Compute the dimensions padded by the dimension multiples. These
	// dimensions will be the dimensions of the packed matrices, including
	// zero-padding, and will be used by the macro- and micro-kernels.
	// We compute them by starting with the effective dimensions of c (now
	// in p) and aligning them to the dimension multiples (typically equal
	// to register blocksizes). This does waste a little bit of space for
	// level-2 operations, but that's okay with us.
	m_p_pad = bli_align_dim_to_mult( bli_obj_length( *p ), mr_def_dim );
	n_p_pad = bli_align_dim_to_mult( bli_obj_width( *p ),  nr_def_dim );

	// Save the padded dimensions into the packed object. It is important
	// to save these dimensions since they represent the actual dimensions
	// of the zero-padded matrix.
	bli_obj_set_padded_dims( m_p_pad, n_p_pad, *p );

	// Now we prepare to compute strides, align them, and compute the
	// total number of bytes needed for the packed buffer. After that,
	// we will acquire an appropriate block of memory from the memory
	// allocator.

	// Extract the element size for the packed object.
	elem_size_p = bli_obj_elem_size( *p );

	// Set the row and column strides of p based on the pack schema.
	if      ( pack_schema == BLIS_PACKED_ROWS )
	{
		// For regular row storage, the padded width of our matrix
		// should be used for the row stride, with the column stride set
		// to one. By using the WIDTH of the mem_t region, we allow for
		// zero-padding (if necessary/desired) along the right edge of
		// the matrix.
		rs_p = n_p_pad;
		cs_p = 1;

		// Align the leading dimension according to the heap stride
		// alignment size so that the second, third, etc rows begin at
		// aligned addresses.
		rs_p = bli_align_dim_to_size( rs_p, elem_size_p,
		                              BLIS_HEAP_STRIDE_ALIGN_SIZE );

		// Store the strides in p.
		bli_obj_set_incs( rs_p, cs_p, *p );

		// Compute the size of the packed buffer.
		size_p = m_p_pad * rs_p * elem_size_p;
	}
	else if ( pack_schema == BLIS_PACKED_COLUMNS )
	{
		// For regular column storage, the padded length of our matrix
		// should be used for the column stride, with the row stride set
		// to one. By using the LENGTH of the mem_t region, we allow for
		// zero-padding (if necessary/desired) along the bottom edge of
		// the matrix.
		cs_p = m_p_pad;
		rs_p = 1;

		// Align the leading dimension according to the heap stride
		// alignment size so that the second, third, etc columns begin at
		// aligned addresses.
		cs_p = bli_align_dim_to_size( cs_p, elem_size_p,
		                              BLIS_HEAP_STRIDE_ALIGN_SIZE );

		// Store the strides in p.
		bli_obj_set_incs( rs_p, cs_p, *p );

		// Compute the size of the packed buffer.
		size_p = cs_p * n_p_pad * elem_size_p;
	}
	else if ( pack_schema == BLIS_PACKED_ROW_PANELS )
	{
		dim_t m_panel;
		dim_t ps_p;

		// The maximum panel length (for each datatype) should be equal to
		// the register blocksize in the m dimension.
		m_panel = mr_def_dim;

		// The "column stride" of a row panel packed object is interpreted as
		// the column stride WITHIN a panel. Thus, this is equal to the panel
		// dimension plus an extension (which may be zero, meaning there is
		// no extension).
		cs_p = mr_pack_dim;

		// The "row stride" of a row panel packed object is interpreted
		// as the row stride WITHIN a panel. Thus, it is unit.
		rs_p = 1;

		// The "panel stride" of a panel packed object is interpreted as the
		// distance between the (0,0) element of panel k and the (0,0)
		// element of panel k+1. We use the padded width computed above to
		// allow for zero-padding (if necessary/desired) along the far end
		// of each panel (ie: the right edge of the matrix). Zero-padding
		// can also occur along the long edge of the last panel if the m
		// dimension of the matrix is not a whole multiple of MR.
		ps_p = cs_p * n_p_pad;

		// Align the panel dimension according to the contiguous memory
		// stride alignment size so that the second, third, etc panels begin
		// at aligned addresses.
		ps_p = bli_align_dim_to_size( ps_p, elem_size_p,
		                              BLIS_CONTIG_STRIDE_ALIGN_SIZE );

		// Store the strides and panel dimension in p.
		bli_obj_set_incs( rs_p, cs_p, *p );
		bli_obj_set_panel_dim( m_panel, *p );
		bli_obj_set_panel_stride( ps_p, *p );

		// Compute the size of the packed buffer.
		size_p = ps_p * (m_p_pad / m_panel) * elem_size_p;
	}
	else if ( pack_schema == BLIS_PACKED_COL_PANELS )
	{
		dim_t n_panel;
		dim_t ps_p;

		// The maximum panel width (for each datatype) should be equal to
		// the register blocksize in the n dimension.
		n_panel = nr_def_dim;

		// The "row stride" of a column panel packed object is interpreted as
		// the row stride WITHIN a panel. Thus, this is equal to the panel
		// dimension plus an extension (which may be zero, meaning there is
		// no extension).
		rs_p = nr_pack_dim;

		// The "column stride" of a column panel packed object is interpreted
		// as the column stride WITHIN a panel. Thus, it is unit.
		cs_p = 1;

		// The "panel stride" of a panel packed object is interpreted as the
		// distance between the (0,0) element of panel k and the (0,0)
		// element of panel k+1. We use the padded length computed above to
		// allow for zero-padding (if necessary/desired) along the far end
		// of each panel (ie: the bottom edge of the matrix). Zero-padding
		// can also occur along the long edge of the last panel if the n
		// dimension of the matrix is not a whole multiple of NR.
		ps_p = m_p_pad * rs_p;

		// Align the panel dimension according to the contiguous memory
		// stride alignment size so that the second, third, etc panels begin
		// at aligned addresses.
		ps_p = bli_align_dim_to_size( ps_p, elem_size_p,
		                              BLIS_CONTIG_STRIDE_ALIGN_SIZE );

		// Store the strides and panel dimension in p.
		bli_obj_set_incs( rs_p, cs_p, *p );
		bli_obj_set_panel_dim( n_panel, *p );
		bli_obj_set_panel_stride( ps_p, *p );

		// Compute the size of the packed buffer.
		size_p = ps_p * (n_p_pad / n_panel) * elem_size_p;
	}
	else
	{
		// NOTE: When implementing block storage, we only need to implement
		// the following two cases:
		// - row-stored blocks in row-major order
		// - column-stored blocks in column-major order
		// The other two combinations coincide with that of packed row-panel
		// and packed column- panel storage.

		size_p = 0;
	}


	if ( bli_mem_is_unalloc( mem_p ) )
	{
		// If the mem_t object of p has not yet been allocated, then acquire
		// a memory block of type pack_buf_type.
		bli_mem_acquire_m( size_p,
		                   pack_buf_type,
		                   mem_p );
	}
	else
	{
		// If the mem_t object is currently allocated and smaller than is
		// needed, then it must have been allocated for a different type
		// of object (a different pack_buf_type value), so we must first
		// release it and then re-acquire it using the new size and new
		// pack_buf_type value.
		if ( bli_mem_size( mem_p ) < size_p )
		{
			bli_mem_release( mem_p );
			bli_mem_acquire_m( size_p,
			                   pack_buf_type,
			                   mem_p );
		}
	}

	// Grab the buffer address from the mem_t object and copy it to the
	// main object buffer field. (Sometimes this buffer address will be
	// copied when the value is already up-to-date, because it persists
	// in the main object buffer field across loop iterations.)
	buf = bli_mem_buffer( mem_p );
	bli_obj_set_buffer( buf, *p );

}


/*
void bli_packm_init_cast( obj_t*  a,
                          obj_t*  p,
                          obj_t*  c )
{
	// The idea here is that we want to create an object c that is identical
	// to object a, except that:
	//  (1) the storage datatype of c is equal to the target datatype of a,
	//      with the element size of c adjusted accordingly,
	//  (2) the view offset of c is reset to (0,0),
	//  (3) object c's main buffer is set to a new memory region acquired
	//      from the memory manager, or extracted from p if a mem entry is
	//      already available, (After acquring a mem entry from the memory
	//      manager, it is cached within p for quick access later on.)
	//  (4) object c is marked as being stored in a standard, contiguous
	//      format (ie: a column-major order).
	// Any transposition encoded within object a will not be handled here,
	// but rather will be handled in the packm implementation. That way,
	// the only thing castm needs to do is cast.

	num_t dt_targ_a    = bli_obj_target_datatype( *a );
	dim_t m_a          = bli_obj_length( *a );
	siz_t elem_size_c  = bli_datatype_size( dt_targ_a );
	inc_t rs_c, cs_c;

	// We begin by copying the basic fields of a.
	bli_obj_alias_to( *a, *c );

	// Update datatype and element size fields.
	bli_obj_set_datatype( dt_targ_a, *c );
	bli_obj_set_elem_size( elem_size_c, *c );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, *c );

    // Check the mem_t entry of p associated with the cast buffer. If it is
    // NULL, then acquire memory sufficient to hold the object data and cache
    // it to p. (Otherwise, if it is non-NULL, then memory has already been
    // acquired from the memory manager and cached.) We then set the main
    // buffer of c to the cached address of the cast memory.
    bli_obj_set_buffer_with_cached_cast_mem( *p, *c );

	// Update the strides. We set the increments to reflect column-major order
	// storage. We start the leading dimension out as m(a) and increment it if
	// necessary so that the beginning of each column is aligned.
	cs_c = bli_align_dim_to_size( m_a, elem_size_c,
	                                   BLIS_HEAP_STRIDE_ALIGN_SIZE );
	rs_c = 1;
	bli_obj_set_incs( rs_c, cs_c, *c );
}
*/

