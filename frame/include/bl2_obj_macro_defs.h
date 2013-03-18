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

#ifndef BLIS_OBJ_MACRO_DEFS_H
#define BLIS_OBJ_MACRO_DEFS_H


// -- Object query/modification macros --

// Info query

#define bl2_obj_domain( obj ) \
\
	(   (obj).info & BLIS_DOMAIN_BIT )

#define bl2_obj_is_real( obj ) \
\
	( ( (obj).info & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_REAL )

#define bl2_obj_is_complex( obj ) \
\
	( ( (obj).info & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_COMPLEX )

#define bl2_obj_precision( obj ) \
\
	(   (obj).info & BLIS_PRECISION_BIT )

#define bl2_obj_is_double_precision( obj ) \
\
	( ( (obj).info & BLIS_PRECISION_BIT ) == BLIS_BITVAL_DOUBLE_PREC )

#define bl2_obj_datatype( obj ) \
\
	(   (obj).info & BLIS_DATATYPE_BITS )

#define bl2_obj_datatype_proj_to_real( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) & ~BLIS_BITVAL_COMPLEX )

#define bl2_obj_datatype_proj_to_complex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) &  BLIS_BITVAL_COMPLEX )

#define bl2_obj_target_datatype( obj ) \
\
	( ( (obj).info & BLIS_TARGET_DT_BITS ) >> BLIS_TARGET_DT_SHIFT )

#define bl2_obj_execution_datatype( obj ) \
\
	( ( (obj).info & BLIS_EXECUTION_DT_BITS ) >> BLIS_EXECUTION_DT_SHIFT )

#define bl2_obj_is_float( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_FLOAT_TYPE )

#define bl2_obj_is_double( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_DOUBLE_TYPE )

#define bl2_obj_is_scomplex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_SCOMPLEX_TYPE )

#define bl2_obj_is_dcomplex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_DCOMPLEX_TYPE )

#define bl2_obj_is_int( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_INT_TYPE )

#define bl2_obj_is_const( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_CONST_TYPE )

#define bl2_obj_conjtrans_status( obj ) \
\
	(   (obj).info & BLIS_CONJTRANS_BITS )

#define bl2_obj_trans_status( obj ) \
\
	(   (obj).info & BLIS_TRANS_BIT )

#define bl2_obj_has_trans( obj ) \
\
	( ( (obj).info  & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS ) \

#define bl2_obj_has_notrans( obj ) \
\
	( ( (obj).info  & BLIS_TRANS_BIT ) == BLIS_BITVAL_NO_TRANS ) \

#define bl2_obj_conj_status( obj ) \
\
	(   (obj).info & BLIS_CONJ_BIT )

#define bl2_obj_has_conj( obj ) \
\
	( ( (obj).info  & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ ) \

#define bl2_obj_has_noconj( obj ) \
\
	( ( (obj).info  & BLIS_CONJ_BIT ) == BLIS_BITVAL_NO_CONJ ) \

#define bl2_obj_uplo( obj ) \
\
	(   (obj).info & BLIS_UPLO_BITS ) \

#define bl2_obj_is_upper( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_UPPER )

#define bl2_obj_is_lower( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_LOWER )

#define bl2_obj_is_upper_after_trans( obj ) \
\
	( bl2_obj_has_trans( (obj) ) ? bl2_obj_is_lower( (obj) ) \
	                             : bl2_obj_is_upper( (obj) ) )

#define bl2_obj_is_lower_after_trans( obj ) \
\
	( bl2_obj_has_trans( (obj) ) ? bl2_obj_is_upper( (obj) ) \
	                             : bl2_obj_is_lower( (obj) ) )

#define bl2_obj_is_upper_or_lower( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_UPPER || \
	  ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_LOWER )

#define bl2_obj_is_dense( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_DENSE )

#define bl2_obj_is_zeros( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_ZEROS )

#define bl2_obj_diag( obj ) \
\
	(   (obj).info & BLIS_UNIT_DIAG_BIT )

#define bl2_obj_has_nonunit_diag( obj ) \
\
	( ( (obj).info & BLIS_UNIT_DIAG_BIT ) == BLIS_BITVAL_NONUNIT_DIAG )

#define bl2_obj_has_unit_diag( obj ) \
\
	( ( (obj).info & BLIS_UNIT_DIAG_BIT ) == BLIS_BITVAL_UNIT_DIAG )

#define bl2_obj_has_inverted_diag( obj ) \
\
	( ( (obj).info & BLIS_INVERT_DIAG_BIT ) == BLIS_BITVAL_INVERT_DIAG )

#define bl2_obj_is_pack_rev_if_upper( obj ) \
\
	( ( (obj).info & BLIS_PACK_REV_IF_UPPER_BIT ) == BLIS_BITVAL_PACK_REV_IF_UPPER )

#define bl2_obj_is_pack_rev_if_lower( obj ) \
\
	( ( (obj).info & BLIS_PACK_REV_IF_LOWER_BIT ) == BLIS_BITVAL_PACK_REV_IF_LOWER )

#define bl2_obj_pack_status( obj ) \
\
	(   (obj).info & BLIS_PACK_BITS )

#define bl2_obj_pack_buffer_type( obj ) \
\
	(   (obj).info & BLIS_PACK_BUFFER_BITS )

#define bl2_obj_struc( obj ) \
\
	(   (obj).info & BLIS_STRUC_BITS )

#define bl2_obj_is_general( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_GENERAL )

#define bl2_obj_is_hermitian( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_HERMITIAN )

#define bl2_obj_is_symmetric( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_SYMMETRIC )

#define bl2_obj_is_triangular( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_TRIANGULAR )


// Info modification

#define bl2_obj_set_conjtrans( conjtrans, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_CONJTRANS_BITS ) | (conjtrans); \
}

#define bl2_obj_set_trans( trans, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_TRANS_BIT ) | (trans); \
}

#define bl2_obj_set_conj( conj, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_CONJ_BIT ) | (conj); \
}

#define bl2_obj_set_uplo( uplo, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_UPLO_BITS ) | (uplo); \
}

#define bl2_obj_set_diag( diag, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_UNIT_DIAG_BIT ) | (diag); \
}

#define bl2_obj_set_invert_diag( inv_diag, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_INVERT_DIAG_BIT ) | (inv_diag); \
}

#define bl2_obj_set_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_DATATYPE_BITS ) | (dt); \
}

#define bl2_obj_set_target_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_TARGET_DT_BITS ) | ( dt << BLIS_TARGET_DT_SHIFT ); \
}

#define bl2_obj_set_execution_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_EXECUTION_DT_BITS ) | ( dt << BLIS_EXECUTION_DT_SHIFT ); \
}

#define bl2_obj_set_pack_schema( pack, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_BITS ) | (pack); \
}

#define bl2_obj_set_pack_order_if_upper( packordifup, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_REV_IF_UPPER_BIT ) | (packordifup); \
}

#define bl2_obj_set_pack_order_if_lower( packordiflo, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_REV_IF_LOWER_BIT ) | (packordiflo); \
}

#define bl2_obj_set_pack_buffer_type( packbuf, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_BUFFER_BITS ) | (packbuf); \
}

#define bl2_obj_set_struc( struc, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_STRUC_BITS ) | (struc); \
}

#define bl2_obj_toggle_trans( obj )\
{ \
	(obj).info = ( (obj).info ^ BLIS_TRANS_BIT ); \
}

#define bl2_obj_toggle_conj( obj )\
{ \
	(obj).info = ( (obj).info ^ BLIS_CONJ_BIT ); \
}

#define bl2_obj_toggle_uplo( obj ) \
{ \
	(obj).info = ( (obj).info ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT; \
}

#define bl2_obj_toggle_region_ref( obj ) \
{ \
	if      ( bl2_obj_is_upper( obj ) ) bl2_obj_inc_diag_off( -1, obj ); \
	else if ( bl2_obj_is_lower( obj ) ) bl2_obj_inc_diag_off(  1, obj ); \
\
	bl2_obj_toggle_uplo( obj ); \
}

#define bl2_obj_toggle_uplo_if_trans( trans, obj ) \
{ \
	if ( bl2_does_trans( trans ) && ( bl2_obj_is_upper_or_lower( obj ) ) ) \
	{ \
		bl2_obj_toggle_uplo( obj ); \
		bl2_obj_negate_diag_offset( obj ); \
	} \
}

#define bl2_obj_apply_trans( trans, obj )\
{ \
	(obj).info = ( (obj).info ^ (trans) ); \
}

#define bl2_obj_apply_conj( conj, obj )\
{ \
	(obj).info = ( (obj).info ^ (conj) ); \
}


// Root matrix query

#define bl2_obj_root( obj ) \
\
	((obj).root)

#define bl2_obj_root_is_general( obj ) \
\
	bl2_obj_is_general( *bl2_obj_root( obj ) ) \

#define bl2_obj_root_is_hermitian( obj ) \
\
	bl2_obj_is_hermitian( *bl2_obj_root( obj ) ) \

#define bl2_obj_root_is_symmetric( obj ) \
\
	bl2_obj_is_symmetric( *bl2_obj_root( obj ) ) \

#define bl2_obj_root_is_triangular( obj ) \
\
	bl2_obj_is_triangular( *bl2_obj_root( obj ) ) \

#define bl2_obj_root_is_upper( obj ) \
\
	bl2_obj_is_upper( *bl2_obj_root( obj ) ) \

#define bl2_obj_root_is_lower( obj ) \
\
	bl2_obj_is_lower( *bl2_obj_root( obj ) ) \


// Root matrix modification

#define bl2_obj_set_as_root( obj )\
{ \
	(obj).root = &(obj); \
}


// Dimension query

#define bl2_obj_length( obj ) \
\
	((obj).m)

#define bl2_obj_width( obj ) \
\
	((obj).n)

#define bl2_obj_min_dim( obj ) \
\
	( bl2_min( bl2_obj_length( obj ), \
	           bl2_obj_width( obj ) ) )

#define bl2_obj_max_dim( obj ) \
\
	( bl2_max( bl2_obj_length( obj ), \
	           bl2_obj_width( obj ) ) )

#define bl2_obj_length_after_trans( obj ) \
\
	( bl2_obj_has_trans( (obj) ) ? bl2_obj_width(  (obj) ) \
	                             : bl2_obj_length( (obj) ) )

#define bl2_obj_width_after_trans( obj ) \
\
	( bl2_obj_has_trans( (obj) ) ? bl2_obj_length( (obj) ) \
	                             : bl2_obj_width(  (obj) ) )

#define bl2_obj_get_dims_after_trans( obj, dim_m, dim_n ) \
{ \
	if ( bl2_obj_has_notrans( trans ) ) \
	{ \
		dim_m = bl2_obj_length( obj ); \
		dim_n = bl2_obj_width( obj ); \
	} \
	else \
	{ \
		dim_m = bl2_obj_width( obj ); \
		dim_n = bl2_obj_length( obj ); \
	} \
}

/*
bl2_obj_length_stored( obj )
{
	if ( lower )
	{
		if ( diagoff < 0 ) m_stored = m + diagoff;
		else               m_stored = m
	}
	else if ( upper )
	{
		if ( diagoff < 0 ) m_stored = min( m, n - diagoff );
		else               m_stored = min( m, n - diagoff );
	}
}
bl2_obj_width_stored( obj )
{
	if ( lower )
	{
		if ( diagoff < 0 ) n_stored = min( n, m + diagoff );
		else               n_stored = min( n, m + diagoff );
	}
	else if ( upper )
	{
		if ( diagoff < 0 ) n_stored = n;
		else               n_stored = n - diagoff;
	}
}
*/
// Note: The purpose of these macros is to obtain the length and width
// of the smallest submatrices of an object that could still encompass
// the stored data above (if obj is upper) or below (if obj is lower)
// the diagonal.
#define bl2_obj_length_stored( obj ) \
\
	( bl2_obj_is_upper( obj ) \
		? bl2_min( bl2_obj_length( obj ), \
		           bl2_obj_width( obj )  - bl2_obj_diag_offset( obj ) ) \
		: bl2_min( bl2_obj_length( obj ), \
		           bl2_obj_length( obj ) + bl2_obj_diag_offset( obj ) ) \
	)

#define bl2_obj_width_stored( obj ) \
\
	( bl2_obj_is_lower( obj ) \
		? bl2_min( bl2_obj_width( obj ), \
		           bl2_obj_length( obj ) + bl2_obj_diag_offset( obj ) ) \
		: bl2_min( bl2_obj_width( obj ), \
		           bl2_obj_width( obj )  - bl2_obj_diag_offset( obj ) ) \
	)

#define bl2_obj_length_stored_after_trans( obj ) \
\
	( bl2_obj_has_trans( obj ) ? bl2_obj_width_stored( obj ) \
	                           : bl2_obj_length_stored( obj ) )

#define bl2_obj_width_stored_after_trans( obj ) \
\
	( bl2_obj_has_trans( obj ) ? bl2_obj_length_stored( obj ) \
	                           : bl2_obj_width_stored( obj ) )

#define bl2_obj_vector_dim( x ) \
\
	( bl2_obj_length( x ) == 1 ? bl2_obj_width( x ) \
	                           : bl2_obj_length( x ) )

#define bl2_obj_vector_inc( x ) \
\
	( bl2_obj_length( x ) == 1 ? bl2_obj_col_stride( x ) \
	                           : bl2_obj_row_stride( x ) )

#define bl2_obj_is_vector( x ) \
\
	( bl2_obj_length( x ) == 1 || \
	  bl2_obj_width( x )  == 1 )

#define bl2_obj_is_row_vector( x ) \
\
	( bl2_obj_length( x ) == 1 )

#define bl2_obj_is_col_vector( x ) \
\
	( bl2_obj_width( x ) == 1 )

#define bl2_obj_has_zero_dim( obj ) \
\
	( bl2_obj_length( obj ) == 0 || \
	  bl2_obj_width( obj )  == 0 )


// Dimension modification

#define bl2_obj_set_dims( dim_m, dim_n, obj ) \
{ \
	(obj).m = dim_m; \
	(obj).n = dim_n; \
}

#define bl2_obj_set_dims_with_trans( trans, dim_m, dim_n, obj ) \
{ \
	if ( bl2_does_notrans( trans ) ) \
	{ \
		(obj).m = dim_m; \
		(obj).n = dim_n; \
	} \
	else \
	{ \
		(obj).m = dim_n; \
		(obj).n = dim_m; \
	} \
}


// Stride/increment query

#define bl2_obj_row_stride( obj ) \
\
	((obj).rs)

#define bl2_obj_col_stride( obj ) \
\
	((obj).cs)

#define bl2_obj_is_row_stored( obj ) \
\
	( (obj).cs == 1 )

#define bl2_obj_is_col_stored( obj ) \
\
	( (obj).rs == 1 )

#define bl2_obj_is_gen_stored( obj ) \
\
	( (obj).rs != 1 && (obj).cs != 1 )

#define bl2_obj_is_row_tilted( obj ) \
\
	( (obj).cs < (obj).rs )

#define bl2_obj_is_col_tilted( obj ) \
\
	( (obj).rs < (obj).cs )


// Stride/increment modification

#define bl2_obj_set_incs( row_stride, col_stride, obj ) \
{ \
	(obj).rs = row_stride; \
	(obj).cs = col_stride; \
}


// Offset query

#define bl2_obj_row_offset( obj ) \
\
	( (obj).offm )

#define bl2_obj_col_offset( obj ) \
\
	( (obj).offn )


// Offset modification

#define bl2_obj_set_offs( offset_m, offset_n, obj ) \
{ \
	(obj).offm = offset_m; \
	(obj).offn = offset_n; \
}

#define bl2_obj_inc_offs( offset_m, offset_n, obj ) \
{ \
	(obj).offm += offset_m; \
	(obj).offn += offset_n; \
}

#define bl2_obj_dec_offs( offset_m, offset_n, obj ) \
{ \
	(obj).offm -= offset_m; \
	(obj).offn -= offset_n; \
}


// Diagonal offset query

#define bl2_obj_diag_offset( obj ) \
\
	((obj).diag_off)

#define bl2_obj_diag_offset_after_trans( obj ) \
\
	( bl2_obj_has_trans( obj ) ? -bl2_obj_diag_offset( obj ) \
	                           :  bl2_obj_diag_offset( obj ) )

#define bl2_obj_has_main_diag( obj ) \
\
	( bl2_obj_diag_offset( obj ) == 0 )

#define bl2_obj_is_strictly_above_diag( obj ) \
\
	( ( doff_t )bl2_obj_length( obj ) <= -bl2_obj_diag_offset( obj ) )

#define bl2_obj_is_strictly_below_diag( obj ) \
\
	( ( doff_t )bl2_obj_width( obj )  <=  bl2_obj_diag_offset( obj ) )

#define bl2_obj_is_outside_diag( obj ) \
\
	( bl2_obj_is_strictly_above_diag( obj ) || \
	  bl2_obj_is_strictly_below_diag( obj ) )

#define bl2_obj_intersects_diag( obj ) \
\
	( !bl2_obj_is_strictly_above_diag( obj ) && \
	  !bl2_obj_is_strictly_below_diag( obj ) )

#define bl2_obj_is_unstored_subpart( obj ) \
\
	( ( bl2_obj_root_is_lower( obj ) && bl2_obj_is_strictly_above_diag( obj ) ) || \
	  ( bl2_obj_root_is_upper( obj ) && bl2_obj_is_strictly_below_diag( obj ) ) )


// Diagonal offset modification

#define bl2_obj_set_diag_offset( offset, obj ) \
{ \
	(obj).diag_off  = ( doff_t )(offset); \
}

#define bl2_obj_negate_diag_offset( obj ) \
{ \
	(obj).diag_off  = -(obj).diag_off; \
}

#define bl2_obj_inc_diag_off( offset, obj ) \
{ \
	(obj).diag_off += ( doff_t )(offset); \
}


// Buffer address query

#define bl2_obj_buffer( obj ) \
\
	(obj).buffer

// Buffer address modification

#define bl2_obj_set_buffer( buf, obj ) \
{ \
	(obj).buffer = buf; \
}


// Bufferless scalar field query

#define bl2_obj_internal_scalar_buffer( obj ) \
\
	&((obj).scalar)


// Element size query

#define bl2_obj_elem_size( obj ) \
\
	(obj).elem_size \

// Element size modification

#define bl2_obj_set_elem_size( size, obj ) \
{ \
	(obj).elem_size = size; \
}


// Pack mem_t entry query

#define bl2_obj_pack_mem( obj ) \
\
	( &((obj).pack_mem) )

// Pack mem_t entry modification

#define bl2_obj_set_pack_mem( mem_p, obj ) \
{ \
	(obj).pack_mem = *mem_p; \
}


// Packed dimensions query

#define bl2_obj_packed_length( obj ) \
\
	( (obj).m_packed )

#define bl2_obj_packed_width( obj ) \
\
	( (obj).n_packed )

// Packed dimensions modification

#define bl2_obj_set_packed_length( m0, obj ) \
{ \
    (obj).m_packed = m0; \
}

#define bl2_obj_set_packed_width( n0, obj ) \
{ \
    (obj).n_packed = n0; \
}

#define bl2_obj_set_packed_dims( m0, n0, obj ) \
{ \
	bl2_obj_set_packed_length( m0, obj ); \
	bl2_obj_set_packed_width( n0, obj ); \
}


// Packed panel stride query

#define bl2_obj_panel_stride( obj ) \
\
	((obj).ps)

// Packed panel stride modification

#define bl2_obj_set_panel_stride( panel_stride, obj ) \
{ \
	(obj).ps = panel_stride; \
}


/*
// Cast mem entry query

#define bl2_obj_cast_mem( obj ) \
\
	( &((obj).cast_mem) )

// Cast mem entry modification

#define bl2_obj_set_cast_mem( mem_p, obj ) \
{ \
	(obj).cast_mem = *mem_p; \
}
*/
 

// -- Miscellaneous object macros --

// Make an alias (shallow copy)

#define bl2_obj_alias_to( a, b ) \
{ \
	bl2_obj_init_as_copy_of( a, b ); \
}

// Check if two objects are aliases of one another

#define bl2_obj_is_alias_of( a, b ) \
\
	( (b).buffer == (a).buffer )

// Create an alias with a trans value applied.
// (Note: trans may include a conj component.)

#define bl2_obj_alias_with_trans( trans, a, b ) \
{ \
	bl2_obj_alias_to( a, b ); \
	bl2_obj_apply_trans( trans, b ); \
}

// Create an alias with a conj value applied.

#define bl2_obj_alias_with_conj( conj, a, b ) \
{ \
	bl2_obj_alias_to( a, b ); \
	bl2_obj_apply_conj( conj, b ); \
}


// Initialize object with default properties (info field)

#define bl2_obj_set_defaults( obj ) \
{ \
	(obj).info = 0x0; \
	(obj).info = (obj).info | BLIS_BITVAL_DENSE | BLIS_BITVAL_GENERAL; \
}


// Initialize object for packing purposes

#define bl2_obj_init_pack( obj_p ) \
{ \
	mem_t* pack_mem = bl2_obj_pack_mem( *obj_p ); \
	/*mem_t* cast_mem = bl2_obj_cast_mem( *obj_p );*/ \
\
	bl2_mem_set_buffer( NULL, pack_mem ); \
	/*bl2_mem_set_buffer( NULL, cast_mem );*/ \
}


// Check if an object is a packed object
// (ie: was a pack buffer acquired for this object; TRUE here does not mean
// the actual packing is complete, such as with incremental packing.)

#define bl2_obj_is_packed( obj ) \
\
	( bl2_obj_buffer( obj ) == bl2_mem_buffer( bl2_obj_pack_mem( obj ) ) && \
	  bl2_obj_buffer( obj ) != NULL \
		? TRUE \
		: FALSE ) \


// Release object's pack (and cast) memory entries back to memory manager

#define bl2_obj_release_pack( obj_p ) \
{ \
	mem_t* pack_mem = bl2_obj_pack_mem( *(obj_p) ); \
	if ( bl2_mem_is_alloc( pack_mem ) ) \
		bl2_mem_release( pack_mem ); \
\
/*
	mem_t* cast_mem = bl2_obj_cast_mem( *(obj_p) ); \
	if ( bl2_mem_is_alloc( cast_mem ) ) \
		bl2_mem_release( cast_mem ); \
*/ \
}



// Submatrix acquisition

#define bl2_obj_buffer_at_off( obj ) \
\
	( void* )( ( ( char* ) (obj).buffer ) + (obj).elem_size * ( (obj).offn * (obj).cs + \
                                                                (obj).offm * (obj).rs ) \
	         )

#define bl2_obj_scalar_buffer( dt, obj ) \
\
	( void* )( bl2_obj_is_const( obj ) ? ( ( char* ) (obj).buffer ) + ( (dt) * BLIS_CONSTANT_SLOT_SIZE ) \
                                       :             (obj).buffer                                        \
	         )

// Swap object pointers

#define bl2_obj_swap_pointers( a, b ) \
{ \
	obj_t* t; \
	t = b; b = a; a = t; \
}

// If a transposition is needed, induce one: swap dimensions, increments
// and offsets, and then clear the trans bit.

#define bl2_obj_induce_trans( obj ) \
{ \
	{ \
		dim_t  m        = bl2_obj_length( obj ); \
		dim_t  n        = bl2_obj_width( obj ); \
		inc_t  rs       = bl2_obj_row_stride( obj ); \
		inc_t  cs       = bl2_obj_col_stride( obj ); \
		dim_t  offm     = bl2_obj_row_offset( obj ); \
		dim_t  offn     = bl2_obj_col_offset( obj ); \
		doff_t diag_off = bl2_obj_diag_offset( obj ); \
\
		bl2_obj_set_dims( n, m, obj ); \
		bl2_obj_set_incs( cs, rs, obj ); \
		bl2_obj_set_offs( offn, offm, obj ); \
		bl2_obj_set_diag_offset( -diag_off, obj ); \
\
		if ( bl2_obj_is_upper_or_lower( obj ) ) \
			bl2_obj_toggle_uplo( obj ); \
\
		/* Note that this macro DOES NOT touch the transposition bit! If
		   the calling code is using this macro to handle an object whose
		   transposition bit is set prior to computation, that code needs
		   to manually clear or toggle the bit, via bl2_obj_set_trans() or
		   bl2_obj_toggle_trans(), respectively. */ \
	} \
}

// Sometimes we need to "reflect" a partition because the data we want is
// actually stored on the other side of the diagonal. The nuts and bolts of
// this macro look a lot like an induced transposition, except that the row
// and column strides are left unchanged (which, of course, drastically
// changes the effect of the macro).

#define bl2_obj_reflect_about_diag( obj ) \
{ \
	{ \
		dim_t  m        = bl2_obj_length( obj ); \
		dim_t  n        = bl2_obj_width( obj ); \
		dim_t  offm     = bl2_obj_row_offset( obj ); \
		dim_t  offn     = bl2_obj_col_offset( obj ); \
		doff_t diag_off = bl2_obj_diag_offset( obj ); \
\
		bl2_obj_set_dims( n, m, obj ); \
		bl2_obj_set_offs( offn, offm, obj ); \
		bl2_obj_set_diag_offset( -diag_off, obj ); \
\
		bl2_obj_toggle_trans( obj ); \
	} \
}

#endif
