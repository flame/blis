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


// -- Matrix partitioning ------------------------------------------------------


void bl2_packm_acquire_mpart_t2b( subpart_t requested_part,
                                  dim_t     i,
                                  dim_t     b,
                                  obj_t*    obj,
                                  obj_t*    sub_obj )
{
	dim_t m, n;

	if ( requested_part != BLIS_SUBPART1 )
	{
		bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	// Partitioning through packed column-stored rows not yet supported
	if ( bl2_obj_is_col_stored( *obj ) )
	{
		bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	// Query the dimensions of the parent object.
	m = bl2_obj_length( *obj );
	n = bl2_obj_width( *obj );

	// Foolproofing: do not let b exceed what's left of the m dimension at
	// row offset i.
	if ( b > m - i ) b = m - i;

	// Begin by copying the info, elem size, buffer, row stride, and column
	// stride fields of the parent object. Note that this omits copying view
	// information because the new partition will have its own dimensions
	// and offsets.
	bl2_obj_init_subpart_from( *obj, *sub_obj );

	// Modify offsets and dimensions of requested partition.
	bl2_obj_set_dims( b, n, *sub_obj );

	// Tweak the width of the pack_mem region of the subpartition to trick
	// the underlying implementation into only zero-padding for the narrow
	// submatrix of interest. Usually, the value we want is b (for non-edge
	// cases), but at the edges, we want the remainder of the mem_t region
	// in the m dimension. Edge cases are defined as occurring when i + b is
	// exactly equal to the length of the parent object. In these cases, we
	// arrive at the new pack_mem region width by simply subtracting off i.
	{
		mem_t* pack_mem  = bl2_obj_pack_mem( *sub_obj );
		dim_t  m_max     = bl2_mem_length( pack_mem );
		dim_t  m_mem;

		if ( i + b == m ) m_mem = m_max - i;
		else              m_mem = b;

		bl2_mem_set_length( m_mem, pack_mem );
	}

	// Translate the desired offsets to a panel offset and adjust the
	// buffer pointer of the subpartition object.
	{
		char* buf_p        = bl2_obj_buffer( *sub_obj );
		siz_t elem_size    = bl2_obj_elem_size( *sub_obj );
		inc_t cs_p         = bl2_obj_col_stride( *sub_obj );
		dim_t off_to_elem  = i * cs_p;

		buf_p = buf_p + elem_size * off_to_elem;

		bl2_obj_set_buffer( ( void* )buf_p, *sub_obj );
	}
}



void bl2_packm_acquire_mpart_l2r( subpart_t requested_part,
                                  dim_t     j,
                                  dim_t     b,
                                  obj_t*    obj,
                                  obj_t*    sub_obj )
{
	dim_t m, n;

	// Check parameters.
	//if ( bl2_error_checking_is_enabled() )
	//	bl2_packm_acquire_mpart_l2r_check( requested_part, j, b, obj, sub_obj );

	if ( requested_part != BLIS_SUBPART1 )
	{
		bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	// Partitioning through packed column-stored rows not yet supported
	if ( bl2_obj_is_col_stored( *obj ) )
	{
		bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	// Query the dimensions of the parent object.
	m = bl2_obj_length( *obj );
	n = bl2_obj_width( *obj );

	// Foolproofing: do not let b exceed what's left of the n dimension at
	// column offset j.
	if ( b > n - j ) b = n - j;

	// Begin by copying the info, elem size, buffer, row stride, and column
	// stride fields of the parent object. Note that this omits copying view
	// information because the new partition will have its own dimensions
	// and offsets.
	bl2_obj_init_subpart_from( *obj, *sub_obj );

	// Modify offsets and dimensions of requested partition.
	bl2_obj_set_dims( m, b, *sub_obj );

/* DON'T NEED THIS NOW THAT COPYING IS DONE IN _INIT_SUBPART_FROM().
	// Copy the pack_mem and cast_mem entries.
	{
		mem_t* pack_mem = bl2_obj_pack_mem( *obj );
		mem_t* cast_mem = bl2_obj_cast_mem( *obj );

		bl2_obj_set_pack_mem( pack_mem, *sub_obj );
		bl2_obj_set_cast_mem( cast_mem, *sub_obj );
	}

	// Copy the panel stride from the original object.
	{
		inc_t ps = bl2_obj_panel_stride( *obj );

		bl2_obj_set_panel_stride( ps, *sub_obj );
	}
*/

	// Tweak the width of the pack_mem region of the subpartition to trick
	// the underlying implementation into only zero-padding for the narrow
	// submatrix of interest. Usually, the value we want is b (for non-edge
	// cases), but at the edges, we want the remainder of the mem_t region
	// in the n dimension. Edge cases are defined as occurring when j + b is
	// exactly equal to the width of the parent object. In these cases, we
	// arrive at the new pack_mem region width by simply subtracting off j.
	{
		mem_t* pack_mem  = bl2_obj_pack_mem( *sub_obj );
		dim_t  n_max     = bl2_mem_width( pack_mem );
		dim_t  n_mem;

		if ( j + b == n ) n_mem = n_max - j;
		else              n_mem = b;

		bl2_mem_set_width( n_mem, pack_mem );
	}

	// Translate the desired offsets to a panel offset and adjust the
	// buffer pointer of the subpartition object.
	{
		char* buf_p        = bl2_obj_buffer( *sub_obj );
		siz_t elem_size    = bl2_obj_elem_size( *sub_obj );
		dim_t off_to_panel = bl2_packm_offset_to_panel_for( j, sub_obj );

		buf_p = buf_p + elem_size * off_to_panel;

		bl2_obj_set_buffer( ( void* )buf_p, *sub_obj );
	}
}



void bl2_packm_acquire_mpart_tl2br( subpart_t requested_part,
                                    dim_t     ij,
                                    dim_t     b,
                                    obj_t*    obj,
                                    obj_t*    sub_obj )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}



dim_t bl2_packm_offset_to_panel_for( dim_t offmn, obj_t* p )
{
	dim_t panel_off;

	if      ( bl2_obj_pack_status( *p ) == BLIS_PACKED_ROWS )
	{
		// For the "packed rows" schema, a single row is effectively one
		// row panel, and so we use the row offset as the panel offset.
		// Then we multiply this offset by the effective panel stride
		// (ie: the row stride) to arrive at the desired offset.
		panel_off = offmn * bl2_obj_row_stride( *p );
	}
	else if ( bl2_obj_pack_status( *p ) == BLIS_PACKED_COLUMNS )
	{
		// For the "packed columns" schema, a single column is effectively one
		// column panel, and so we use the column offset as the panel offset.
		// Then we multiply this offset by the effective panel stride
		// (ie: the column stride) to arrive at the desired offset.
		panel_off = offmn * bl2_obj_col_stride( *p );
	}
	else if ( bl2_obj_pack_status( *p ) == BLIS_PACKED_ROW_PANELS )
	{
		// For the "packed row panels" schema, the column stride is equal to
		// the panel dimension (length). So we can divide it into offmn
		// (interpreted as a row offset) to arrive at a panel offset. Then
		// we multiply this offset by the panel stride to arrive at the total
		// offset to the panel (in units of elements).
		panel_off = offmn / bl2_obj_col_stride( *p );
		panel_off = panel_off * bl2_obj_panel_stride( *p );

		// Sanity check.
		if ( offmn % bl2_obj_col_stride( *p ) > 0 ) bl2_abort();
	}
	else if ( bl2_obj_pack_status( *p ) == BLIS_PACKED_COL_PANELS )
	{
		// For the "packed column panels" schema, the row stride is equal to
		// the panel dimension (width). So we can divide it into offmn
		// (interpreted as a column offset) to arrive at a panel offset. Then
		// we multiply this offset by the panel stride to arrive at the total
		// offset to the panel (in units of elements).
		panel_off = offmn / bl2_obj_row_stride( *p );
		panel_off = panel_off * bl2_obj_panel_stride( *p );

		// Sanity check.
		if ( offmn % bl2_obj_row_stride( *p ) > 0 ) bl2_abort();
	}
	else
	{
		panel_off = 0;
		bl2_abort();
	}

	return panel_off;
}
