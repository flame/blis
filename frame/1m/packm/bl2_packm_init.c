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

void bl2_packm_init( obj_t*   a,
                     obj_t*   p,
                     packm_t* cntl )
{
	// The packm operation consists of an optional typecasting pre-process.
	// Here are the following possible ways packm can execute:
	//  1. cast and pack: When typecasting and packing are both
	//     precribed, typecast a to temporary matrix c and then pack
	//     c to p.
	//  2. pack only: Typecasting is skipped when it is not needed;
	//     simply pack a directly to p.
	//  3. cast only: Not yet supported / not used.
	//  4. no-op: The control tree sometimes directs us to skip the
	//     pack operation entirely. Alias p to a and return.
	bool_t    needs_densify;
	invdiag_t invert_diag;
	pack_t    pack_schema;
	packord_t pack_ord_if_up;
	packord_t pack_ord_if_lo;
	blksz_t*  mult_m;
	blksz_t*  mult_n;
	obj_t     c;

	// Check parameters.
	if ( bl2_error_checking_is_enabled() )
		bl2_packm_check( &BLIS_ONE, a, p, cntl );

	// First check if we are to skip this operation because the control tree
	// is NULL, and if so, simply alias the object to its packed counterpart.
	if ( cntl_is_noop( cntl ) )
	{
		bl2_obj_alias_to( *a, *p );
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
	if ( bl2_obj_pack_status( *a ) == BLIS_PACKED_UNSPEC )
	{
		bl2_obj_alias_to( *a, *p );
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
	if ( bl2_obj_pack_status( *a ) == cntl_pack_schema( cntl ) )
	{
		bl2_obj_alias_to( *a, *p );
		return;
	}

	// Now, if we are not skipping the pack operation, then the only question
	// left is whether we are to typecast matrix a before packing.
	if ( bl2_obj_datatype( *a ) != bl2_obj_target_datatype( *a ) )
		bl2_abort();
/*
	{
		// Initialize an object c for the intermediate typecast matrix.
		bl2_packm_init_cast( a,
		                     p,
		                     &c );

		// Copy/typecast matrix a to matrix c.
		bl2_copym( a,
		           &c );
	}
	else
*/
	{
		// If no cast is needed, then aliasing object c to the original
		// matrix serves as a minor optimization. This causes the packm
		// implementation to pack directly from matrix a.
		bl2_obj_alias_to( *a, c );
	}

	
	// Extract various fields from the control tree and pass them in
	// explicitly into _init_pack(). This allows external code generators
	// the option of bypassing usage of control trees altogether.
	needs_densify  = cntl_does_densify( cntl );
	pack_schema    = cntl_pack_schema( cntl );
	mult_m         = cntl_mult_m( cntl );
	mult_n         = cntl_mult_n( cntl );

	if ( cntl_does_invert_diag( cntl ) ) invert_diag = BLIS_INVERT_DIAG;
	else                                 invert_diag = BLIS_NO_INVERT_DIAG;

	if ( cntl_rev_iter_if_upper( cntl ) ) pack_ord_if_up = BLIS_PACK_REV_IF_UPPER;
	else                                  pack_ord_if_up = BLIS_PACK_FWD_IF_UPPER;

	if ( cntl_rev_iter_if_lower( cntl ) ) pack_ord_if_lo = BLIS_PACK_REV_IF_LOWER;
	else                                  pack_ord_if_lo = BLIS_PACK_FWD_IF_LOWER;

	// Initialize object p for the final packed matrix.
	bl2_packm_init_pack( needs_densify,
	                     invert_diag,
	                     pack_schema,
	                     pack_ord_if_up,
	                     pack_ord_if_lo,
	                     mult_m,
	                     mult_n,
	                     &c,
	                     p );

	// Now p is ready to be packed.
}


void bl2_packm_init_pack( bool_t    densify,
                          invdiag_t invert_diag,
                          pack_t    pack_schema,
                          packord_t pack_ord_if_up,
                          packord_t pack_ord_if_lo,
                          blksz_t*  mult_m,
                          blksz_t*  mult_n,
                          obj_t*    c,
                          obj_t*    p )
{
	// In this function, we initialize an object p to represent the packed
	// copy of the intermediate object c. At this point, the datatype of
	// object c should be equal to the target datatype of the original
	// object, either because:
	//  (1) c is set up to contain the typecast of the original object, or
	//  (2) c is aliased to the original object, which would only happen
	//      when the original object's datatype and target datatype are
	//      equal.
	// So here, we want to create an object p that is identical to c, except
	// that:
	//  (1) the dimensions of p are explicitly transposed, if c needs
	//      transposition,
	//  (2) if c needs transposition, we adjust the diagonal offset of p
	//      and we also either set the uplo of p to dense (if we are going
	//      to densify), or to its toggled value.
	//  (3) the view offset of p is reset to (0,0),
	//  (4) object p contains a pack schema field that reflects its desired
	//      packing,
	//  (5) object p's main buffer is set to a new memory region acquired
	//      from the memory manager, or extracted from p if a mem entry is
	//      already available, (After acquring a mem entry from the memory
	//      manager, it is cached within p for quick access later on.)
	//  (6) object p gets new stride information based on the pack schema
	//      embedded in the control tree node.

	num_t   datatype     = bl2_obj_datatype( *c );
	trans_t transc       = bl2_obj_trans_status( *c );
	dim_t   m_c          = bl2_obj_length( *c );
	dim_t   n_c          = bl2_obj_width( *c );
	dim_t   mult_m_dim   = bl2_blksz_for_type( datatype, mult_m );
	dim_t   mult_n_dim   = bl2_blksz_for_type( datatype, mult_n );
	inc_t   rs_p, cs_p;

	// We begin by copying the basic fields of c.
	bl2_obj_alias_to( *c, *p );

	// Update the dimension fields to explicitly reflect a transposition,
	// if needed.
	// Then, clear the conjugation and transposition fields from the object
	// since matrix packing in BLIS is deemed to take care of all conjugation
	// and transposition necessary.
	// Then, we adjust the properties of p when c needs a transposition.
	// We negate the diagonal offset, and if c is upper- or lower-stored,
	// we either toggle the uplo of p.
	// Finally, if we are going to densify c, we mark p as dense.
	bl2_obj_set_dims_with_trans( transc, m_c, n_c, *p );
	bl2_obj_set_conjtrans( BLIS_NO_TRANSPOSE, *p );
	if ( bl2_does_trans( transc ) )
	{
		bl2_obj_negate_diag_offset( *p );
		if ( bl2_obj_is_upper_or_lower( *c ) )
			bl2_obj_toggle_uplo( *p );
	}
	if ( densify ) bl2_obj_set_uplo( BLIS_DENSE, *p );

	// Reset the view offsets to (0,0).
	bl2_obj_set_offs( 0, 0, *p );

	// Set the invert diagonal field.
	bl2_obj_set_invert_diag( invert_diag, *p );

	// Set the pack status of p to the pack schema prescribed in the control
	// tree node.
	bl2_obj_set_pack_schema( pack_schema, *p );

	// Set the packing order bits.
	bl2_obj_set_pack_order_if_upper( pack_ord_if_up, *p );
	bl2_obj_set_pack_order_if_lower( pack_ord_if_lo, *p );

	// Check the mem_t entry of p associated with the pack buffer. If it is
	// NULL, then acquire memory sufficient to hold the object data and cache
	// it to p. (Otherwise, if it is non-NULL, then memory has already been
	// acquired from the memory manager and cached.) We then set the main
	// buffer of p to the cached address of the pack memory.
	bl2_obj_set_buffer_with_cached_packm_mem( *p, *p, mult_m_dim, mult_n_dim );

	// Set the row and column strides of p based on the pack schema.
	if      ( pack_schema == BLIS_PACKED_ROWS )
	{
		mem_t* mem;

		// Access the mem_t entry cached in p.
		mem = bl2_obj_pack_mem( *p );

		// For regular row storage, the n dimension used when acquiring the
		// pack memory should be used for our row stride, with the column
		// stride set to one. By using the WIDTH of the mem_t region, we
		// allow for zero-padding (if necessary/desired) along the right
		// edge of the matrix.
		rs_p = bl2_mem_width( mem );
		cs_p = 1;

		bl2_obj_set_incs( rs_p, cs_p, *p );
	}
	else if ( pack_schema == BLIS_PACKED_COLUMNS )
	{
		mem_t* mem;

		// Access the mem_t entry cached in p.
		mem = bl2_obj_pack_mem( *p );

		// For regular column storage, the m dimension used when acquiring the
		// pack memory should be used for our column stride, with the row
		// stride set to one. By using the LENGTH of the mem_t region, we
		// allow for zero-padding (if necessary/desired) along the bottom
		// edge of the matrix.
		cs_p = bl2_mem_length( mem );
		rs_p = 1;

		bl2_obj_set_incs( rs_p, cs_p, *p );
	}
	else if ( pack_schema == BLIS_PACKED_ROW_PANELS )
	{
		mem_t*   mem     = bl2_obj_pack_mem( *p );
		dim_t    m_panel;
		dim_t    ps_p;

		// The maximum panel length (for each datatype) should be equal to
		// the m dimension multiple field of the control tree node.
		m_panel = mult_m_dim;

		// The "column stride" of a row panel packed object is interpreted as
		// the column stride WITHIN a panel. Thus, this is equal to the panel
		// length.
		cs_p = m_panel;

		// The "row stride" of a row panel packed object is interpreted
		// as the row stride WITHIN a panel. Thus, it is unit.
		rs_p = 1;

		// The "panel stride" of a panel packed object is interpreted as the
		// distance between the (0,0) element of panel k and the (0,0)
		// element of panel k+1. We use the WIDTH of the mem_t region to
		// determine the panel "width"; this will allow for zero-padding
		// (if necessary/desired) along the far end of each panel (ie: the
		// right edge of the matrix).
		ps_p = cs_p * bl2_mem_width( mem );

		// Store the strides in p.
		bl2_obj_set_incs( rs_p, cs_p, *p );
		bl2_obj_set_panel_stride( ps_p, *p );
	}
	else if ( pack_schema == BLIS_PACKED_COL_PANELS )
	{
		mem_t*   mem     = bl2_obj_pack_mem( *p );
		dim_t    n_panel;
		dim_t    ps_p;

		// The maximum panel width (for each datatype) should be equal to
		// the n dimension multiple field of the control tree node.
		n_panel = mult_n_dim;

		// The "row stride" of a column panel packed object is interpreted as
		// the row stride WITHIN a panel. Thus, it is equal to the panel
		// width.
		rs_p = n_panel;

		// The "column stride" of a column panel packed object is interpreted
		// as the column stride WITHIN a panel. Thus, it is unit.
		cs_p = 1;

		// The "panel stride" of a panel packed object is interpreted as the
		// distance between the (0,0) element of panel k and the (0,0)
		// element of panel k+1. We use the LENGTH of the mem_t region to
		// determine the panel "length"; this will allow for zero-padding
		// (if necessary/desired) along the far end of each panel (ie: the
		// bottom edge of the matrix).
		ps_p = bl2_mem_length( mem ) * rs_p;

		// Store the strides in p.
		bl2_obj_set_incs( rs_p, cs_p, *p );
		bl2_obj_set_panel_stride( ps_p, *p );
	}
	else
	{
		// If the pack schema is something else, we assume stride information
		// of p is set later on, by the implementation.
	}
}


/*
void bl2_packm_init_cast( obj_t*  a,
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

	num_t dt_targ_a    = bl2_obj_target_datatype( *a );
	dim_t m_a          = bl2_obj_length( *a );
	siz_t elem_size_c  = bl2_datatype_size( dt_targ_a );
	inc_t rs_c, cs_c;

	// We begin by copying the basic fields of a.
	bl2_obj_alias_to( *a, *c );

	// Update datatype and element size fields.
	bl2_obj_set_datatype( dt_targ_a, *c );
	bl2_obj_set_elem_size( elem_size_c, *c );

	// Reset the view offsets to (0,0).
	bl2_obj_set_offs( 0, 0, *c );

    // Check the mem_t entry of p associated with the cast buffer. If it is
    // NULL, then acquire memory sufficient to hold the object data and cache
    // it to p. (Otherwise, if it is non-NULL, then memory has already been
    // acquired from the memory manager and cached.) We then set the main
    // buffer of c to the cached address of the cast memory.
    bl2_obj_set_buffer_with_cached_cast_mem( *p, *c );

	// Update the strides. We set the increments to reflect column-major order
	// storage. We start the leading dimension out as m(a) and increment it if
	// necessary so that the beginning of each column is aligned.
	cs_c = bl2_align_dim_to_sys( m_a, elem_size_c );
	rs_c = 1;
	bl2_obj_set_incs( rs_c, cs_c, *c );
}
*/

