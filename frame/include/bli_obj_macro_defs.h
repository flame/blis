/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

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

#ifndef BLIS_OBJ_MACRO_DEFS_H
#define BLIS_OBJ_MACRO_DEFS_H


// -- Object query/modification macros --

// Info query

#define bli_obj_is_float( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_FLOAT_TYPE )

#define bli_obj_is_double( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_DOUBLE_TYPE )

#define bli_obj_is_scomplex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_SCOMPLEX_TYPE )

#define bli_obj_is_dcomplex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_DCOMPLEX_TYPE )

#define bli_obj_is_int( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_INT_TYPE )

#define bli_obj_is_const( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) == BLIS_BITVAL_CONST_TYPE )

#define bli_obj_domain( obj ) \
\
	(   (obj).info & BLIS_DOMAIN_BIT )

#define bli_obj_is_real( obj ) \
\
	( ( (obj).info & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_REAL )

#define bli_obj_is_complex( obj ) \
\
	( ( (obj).info & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_COMPLEX )

#define bli_obj_precision( obj ) \
\
	(   (obj).info & BLIS_PRECISION_BIT )

#define bli_obj_is_double_precision( obj ) \
\
	( ( (obj).info & BLIS_PRECISION_BIT ) == BLIS_BITVAL_DOUBLE_PREC )

#define bli_obj_datatype( obj ) \
\
	(   (obj).info & BLIS_DATATYPE_BITS )

#define bli_obj_datatype_proj_to_real( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) & ~BLIS_BITVAL_COMPLEX )

#define bli_obj_datatype_proj_to_complex( obj ) \
\
	( ( (obj).info & BLIS_DATATYPE_BITS ) &  BLIS_BITVAL_COMPLEX )

#define bli_obj_target_datatype( obj ) \
\
	( ( (obj).info & BLIS_TARGET_DT_BITS ) >> BLIS_TARGET_DT_SHIFT )

#define bli_obj_execution_datatype( obj ) \
\
	( ( (obj).info & BLIS_EXECUTION_DT_BITS ) >> BLIS_EXECUTION_DT_SHIFT )

#define bli_obj_conjtrans_status( obj ) \
\
	(   (obj).info & BLIS_CONJTRANS_BITS )

#define bli_obj_onlytrans_status( obj ) \
\
	(   (obj).info & BLIS_TRANS_BIT )

#define bli_obj_has_trans( obj ) \
\
	( ( (obj).info  & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS ) \

#define bli_obj_has_notrans( obj ) \
\
	( ( (obj).info  & BLIS_TRANS_BIT ) == BLIS_BITVAL_NO_TRANS ) \

#define bli_obj_conj_status( obj ) \
\
	(   (obj).info & BLIS_CONJ_BIT )

#define bli_obj_has_conj( obj ) \
\
	( ( (obj).info  & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ ) \

#define bli_obj_has_noconj( obj ) \
\
	( ( (obj).info  & BLIS_CONJ_BIT ) == BLIS_BITVAL_NO_CONJ ) \

#define bli_obj_uplo( obj ) \
\
	(   (obj).info & BLIS_UPLO_BITS ) \

#define bli_obj_is_upper( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_UPPER )

#define bli_obj_is_lower( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_LOWER )

#define bli_obj_is_upper_after_trans( obj ) \
\
	( bli_obj_has_trans( (obj) ) ? bli_obj_is_lower( (obj) ) \
	                             : bli_obj_is_upper( (obj) ) )

#define bli_obj_is_lower_after_trans( obj ) \
\
	( bli_obj_has_trans( (obj) ) ? bli_obj_is_upper( (obj) ) \
	                             : bli_obj_is_lower( (obj) ) )

#define bli_obj_is_upper_or_lower( obj ) \
\
	( bli_obj_is_upper( obj ) || \
	  bli_obj_is_lower( obj ) )

#define bli_obj_is_dense( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_DENSE )

#define bli_obj_is_zeros( obj ) \
\
	( ( (obj).info & BLIS_UPLO_BITS ) == BLIS_BITVAL_ZEROS )

#define bli_obj_diag( obj ) \
\
	(   (obj).info & BLIS_UNIT_DIAG_BIT )

#define bli_obj_has_nonunit_diag( obj ) \
\
	( ( (obj).info & BLIS_UNIT_DIAG_BIT ) == BLIS_BITVAL_NONUNIT_DIAG )

#define bli_obj_has_unit_diag( obj ) \
\
	( ( (obj).info & BLIS_UNIT_DIAG_BIT ) == BLIS_BITVAL_UNIT_DIAG )

#define bli_obj_has_inverted_diag( obj ) \
\
	( ( (obj).info & BLIS_INVERT_DIAG_BIT ) == BLIS_BITVAL_INVERT_DIAG )

#define bli_obj_is_pack_rev_if_upper( obj ) \
\
	( ( (obj).info & BLIS_PACK_REV_IF_UPPER_BIT ) == BLIS_BITVAL_PACK_REV_IF_UPPER )

#define bli_obj_is_pack_rev_if_lower( obj ) \
\
	( ( (obj).info & BLIS_PACK_REV_IF_LOWER_BIT ) == BLIS_BITVAL_PACK_REV_IF_LOWER )

#define bli_obj_pack_schema( obj ) \
\
	(   (obj).info & BLIS_PACK_SCHEMA_BITS )

#define bli_obj_is_packed( obj ) \
\
	( ( (obj).info & BLIS_PACK_BIT  ) )

#define bli_obj_is_row_packed( obj ) \
\
	( ( (obj).info & BLIS_PACK_RC_BIT  ) == ( BLIS_BITVAL_PACKED_UNSPEC ^ \
	                                          BLIS_BITVAL_PACKED_ROWS    ) )

#define bli_obj_is_col_packed( obj ) \
\
	( ( (obj).info & BLIS_PACK_RC_BIT  ) == ( BLIS_BITVAL_PACKED_UNSPEC ^ \
	                                          BLIS_BITVAL_PACKED_COLUMNS ) )

#define bli_obj_is_panel_packed( obj ) \
\
	( ( (obj).info & BLIS_PACK_PANEL_BIT ) )

#define bli_obj_pack_buffer_type( obj ) \
\
	(   (obj).info & BLIS_PACK_BUFFER_BITS )

#define bli_obj_struc( obj ) \
\
	(   (obj).info & BLIS_STRUC_BITS )

#define bli_obj_is_general( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_GENERAL )

#define bli_obj_is_hermitian( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_HERMITIAN )

#define bli_obj_is_symmetric( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_SYMMETRIC )

#define bli_obj_is_triangular( obj ) \
\
	( ( (obj).info & BLIS_STRUC_BITS ) == BLIS_BITVAL_TRIANGULAR )

#define bli_obj_is_herm_or_symm( obj ) \
\
	( bli_obj_is_hermitian( obj ) || \
	  bli_obj_is_symmetric( obj ) )



// Info modification

#define bli_obj_set_conjtrans( conjtrans, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_CONJTRANS_BITS ) | (conjtrans); \
}

#define bli_obj_set_onlytrans( trans, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_TRANS_BIT ) | (trans); \
}

#define bli_obj_set_conj( conjval, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_CONJ_BIT ) | (conjval); \
}

#define bli_obj_set_uplo( uplo, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_UPLO_BITS ) | (uplo); \
}

#define bli_obj_set_diag( diag, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_UNIT_DIAG_BIT ) | (diag); \
}

#define bli_obj_set_invert_diag( inv_diag, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_INVERT_DIAG_BIT ) | (inv_diag); \
}

#define bli_obj_set_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_DATATYPE_BITS ) | (dt); \
}

#define bli_obj_set_target_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_TARGET_DT_BITS ) | ( dt << BLIS_TARGET_DT_SHIFT ); \
}

#define bli_obj_set_execution_datatype( dt, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_EXECUTION_DT_BITS ) | ( dt << BLIS_EXECUTION_DT_SHIFT ); \
}

#define bli_obj_set_pack_schema( pack, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_SCHEMA_BITS ) | (pack); \
}

#define bli_obj_set_pack_order_if_upper( packordifup, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_REV_IF_UPPER_BIT ) | (packordifup); \
}

#define bli_obj_set_pack_order_if_lower( packordiflo, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_REV_IF_LOWER_BIT ) | (packordiflo); \
}

#define bli_obj_set_pack_buffer_type( packbuf, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_PACK_BUFFER_BITS ) | (packbuf); \
}

#define bli_obj_set_struc( struc, obj ) \
{ \
	(obj).info = ( (obj).info & ~BLIS_STRUC_BITS ) | (struc); \
}

#define bli_obj_toggle_trans( obj )\
{ \
	(obj).info = ( (obj).info ^ BLIS_TRANS_BIT ); \
}

#define bli_obj_toggle_conj( obj )\
{ \
	(obj).info = ( (obj).info ^ BLIS_CONJ_BIT ); \
}

#define bli_obj_toggle_uplo( obj ) \
{ \
	(obj).info = ( (obj).info ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT; \
}

#define bli_obj_toggle_region_ref( obj ) \
{ \
	if      ( bli_obj_is_upper( obj ) ) bli_obj_inc_diag_off( -1, obj ); \
	else if ( bli_obj_is_lower( obj ) ) bli_obj_inc_diag_off(  1, obj ); \
\
	bli_obj_toggle_uplo( obj ); \
}

#define bli_obj_toggle_uplo_if_trans( trans, obj ) \
{ \
	if ( bli_does_trans( trans ) && ( bli_obj_is_upper_or_lower( obj ) ) ) \
	{ \
		bli_obj_toggle_uplo( obj ); \
		bli_obj_negate_diag_offset( obj ); \
	} \
}

#define bli_obj_apply_trans( trans, obj ) \
{ \
	(obj).info = ( (obj).info ^ (trans) ); \
}

#define bli_obj_apply_conj( conjval, obj ) \
{ \
	(obj).info = ( (obj).info ^ (conjval) ); \
}


// Root matrix query

#define bli_obj_root( obj ) \
\
	((obj).root)

#define bli_obj_root_uplo( obj ) \
\
	bli_obj_uplo( *bli_obj_root( obj ) )

#define bli_obj_root_is_general( obj ) \
\
	bli_obj_is_general( *bli_obj_root( obj ) )

#define bli_obj_root_is_hermitian( obj ) \
\
	bli_obj_is_hermitian( *bli_obj_root( obj ) )

#define bli_obj_root_is_symmetric( obj ) \
\
	bli_obj_is_symmetric( *bli_obj_root( obj ) )

#define bli_obj_root_is_triangular( obj ) \
\
	bli_obj_is_triangular( *bli_obj_root( obj ) )

#define bli_obj_root_is_herm_or_symm( obj ) \
\
	( bli_obj_is_hermitian( *bli_obj_root( obj ) ) || \
	  bli_obj_is_symmetric( *bli_obj_root( obj ) ) )

#define bli_obj_root_is_upper( obj ) \
\
	bli_obj_is_upper( *bli_obj_root( obj ) )

#define bli_obj_root_is_lower( obj ) \
\
	bli_obj_is_lower( *bli_obj_root( obj ) )


// Root matrix modification

#define bli_obj_set_as_root( obj )\
{ \
	(obj).root = &(obj); \
}


// Dimension query

#define bli_obj_length( obj ) \
\
	( (obj).dim[BLIS_M] )

#define bli_obj_width( obj ) \
\
	( (obj).dim[BLIS_N] )

#define bli_obj_dim( mdim, obj ) \
\
	( (obj).dim[mdim] )

#define bli_obj_min_dim( obj ) \
\
	( bli_min( bli_obj_length( obj ), \
	           bli_obj_width( obj ) ) )

#define bli_obj_max_dim( obj ) \
\
	( bli_max( bli_obj_length( obj ), \
	           bli_obj_width( obj ) ) )

#define bli_obj_length_after_trans( obj ) \
\
	( bli_obj_has_trans( (obj) ) ? bli_obj_width(  (obj) ) \
	                             : bli_obj_length( (obj) ) )

#define bli_obj_width_after_trans( obj ) \
\
	( bli_obj_has_trans( (obj) ) ? bli_obj_length( (obj) ) \
	                             : bli_obj_width(  (obj) ) )

#define bli_obj_get_dims_after_trans( obj, dim_m, dim_n ) \
{ \
	if ( bli_obj_has_notrans( trans ) ) \
	{ \
		dim_m = bli_obj_length( obj ); \
		dim_n = bli_obj_width( obj ); \
	} \
	else \
	{ \
		dim_m = bli_obj_width( obj ); \
		dim_n = bli_obj_length( obj ); \
	} \
}

/*
bli_obj_length_stored( obj )
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
bli_obj_width_stored( obj )
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
#define bli_obj_length_stored( obj ) \
\
	( bli_obj_is_upper( obj ) \
		? bli_min( bli_obj_length( obj ), \
		           bli_obj_width( obj )  - bli_obj_diag_offset( obj ) ) \
		: bli_min( bli_obj_length( obj ), \
		           bli_obj_length( obj ) + bli_obj_diag_offset( obj ) ) \
	)

#define bli_obj_width_stored( obj ) \
\
	( bli_obj_is_lower( obj ) \
		? bli_min( bli_obj_width( obj ), \
		           bli_obj_length( obj ) + bli_obj_diag_offset( obj ) ) \
		: bli_min( bli_obj_width( obj ), \
		           bli_obj_width( obj )  - bli_obj_diag_offset( obj ) ) \
	)

#define bli_obj_length_stored_after_trans( obj ) \
\
	( bli_obj_has_trans( obj ) ? bli_obj_width_stored( obj ) \
	                           : bli_obj_length_stored( obj ) )

#define bli_obj_width_stored_after_trans( obj ) \
\
	( bli_obj_has_trans( obj ) ? bli_obj_length_stored( obj ) \
	                           : bli_obj_width_stored( obj ) )

#define bli_obj_vector_dim( x ) \
\
	( bli_obj_length( x ) == 1 ? bli_obj_width( x ) \
	                           : bli_obj_length( x ) )

#define bli_obj_vector_inc( x ) \
\
	( bli_obj_is_1x1( x ) ? 1 : \
	( bli_obj_length( x ) == 1 ? bli_obj_col_stride( x ) \
	                           : bli_obj_row_stride( x ) ) \
	)

#define bli_obj_is_vector( x ) \
\
	( bli_obj_length( x ) == 1 || \
	  bli_obj_width( x )  == 1 )

#define bli_obj_is_row_vector( x ) \
\
	( bli_obj_length( x ) == 1 )

#define bli_obj_is_col_vector( x ) \
\
	( bli_obj_width( x ) == 1 )

#define bli_obj_has_zero_dim( obj ) \
\
	( bli_obj_length( obj ) == 0 || \
	  bli_obj_width( obj )  == 0 )

#define bli_obj_is_1x1( x ) \
\
	( bli_obj_length( x ) == 1 && \
	  bli_obj_width( x )  == 1 )


// Dimension modification

#define bli_obj_set_length( dim_m, obj ) \
{ \
	(obj).dim[BLIS_M] = dim_m; \
}

#define bli_obj_set_width( dim_n, obj ) \
{ \
	(obj).dim[BLIS_N] = dim_n; \
}

#define bli_obj_set_dim( mdim, dim_val, obj ) \
{ \
	(obj).dim[mdim] = dim_val; \
}

#define bli_obj_set_dims( dim_m, dim_n, obj ) \
{ \
	bli_obj_set_length( dim_m, obj ); \
	bli_obj_set_width( dim_n, obj ); \
}

#define bli_obj_set_dims_with_trans( trans, dim_m, dim_n, obj ) \
{ \
	if ( bli_does_notrans( trans ) ) \
	{ \
		bli_obj_set_length( dim_m, obj ); \
		bli_obj_set_width( dim_n, obj ); \
	} \
	else \
	{ \
		bli_obj_set_length( dim_n, obj ); \
		bli_obj_set_width( dim_m, obj ); \
	} \
}


// Stride/increment query

#define bli_obj_row_stride( obj ) \
\
	( (obj).rs )

#define bli_obj_col_stride( obj ) \
\
	( (obj).cs )

#define bli_obj_imag_stride( obj ) \
\
	( (obj).is )

#define bli_obj_row_stride_mag( obj ) \
\
	( bli_abs( bli_obj_row_stride( obj ) ) )

#define bli_obj_col_stride_mag( obj ) \
\
	( bli_abs( bli_obj_col_stride( obj ) ) )

#define bli_obj_imag_stride_mag( obj ) \
\
	( bli_abs( bli_obj_imag_stride( obj ) ) )

//
// NOTE: The following two macros differ from their non-obj counterparts
// in that they do not identify m x 1 and 1 x n objects as row-stored and
// column-stored, respectively, which is needed when considering packed
// objects. But this is okay, since none of the invocations of these
// "obj" macros are used on packed matrices.
//
#define bli_obj_is_row_stored( obj ) \
\
	( bli_obj_col_stride_mag( obj ) == 1 )

#define bli_obj_is_col_stored( obj ) \
\
	( bli_obj_row_stride_mag( obj ) == 1 )

#define bli_obj_is_gen_stored( obj ) \
\
	( bli_obj_row_stride_mag( obj ) != 1 && \
	  bli_obj_col_stride_mag( obj ) != 1 )

#define bli_obj_is_row_tilted( obj ) \
\
	( bli_obj_col_stride_mag( obj ) < bli_obj_row_stride_mag( obj ) )

#define bli_obj_is_col_tilted( obj ) \
\
	( bli_obj_row_stride_mag( obj ) < bli_obj_col_stride_mag( obj ) )


// Stride/increment modification

#define bli_obj_set_strides( row_stride, col_stride, obj ) \
{ \
	(obj).rs = row_stride; \
	(obj).cs = col_stride; \
}

#define bli_obj_set_imag_stride( imag_stride, obj ) \
{ \
	(obj).is = imag_stride; \
}


// Offset query

#define bli_obj_row_off( obj ) \
\
	( (obj).off[BLIS_M] )

#define bli_obj_col_off( obj ) \
\
	( (obj).off[BLIS_N] )

#define bli_obj_off( mdim, obj ) \
\
	( (obj).off[mdim] )


// Offset modification

#define bli_obj_set_off( mdim, offset, obj ) \
{ \
	(obj).off[mdim] = offset; \
}

#define bli_obj_set_offs( offset_m, offset_n, obj ) \
{ \
	bli_obj_set_off( BLIS_M, offset_m, obj ); \
	bli_obj_set_off( BLIS_N, offset_n, obj ); \
}

#define bli_obj_inc_off( mdim, offset, obj ) \
{ \
	(obj).off[mdim] += offset; \
}

#define bli_obj_inc_offm( offset, obj ) \
{ \
	bli_obj_inc_off( BLIS_M, offset, obj ); \
}

#define bli_obj_inc_offn( offset, obj ) \
{ \
	bli_obj_inc_off( BLIS_N, offset, obj ); \
}

#define bli_obj_inc_offs( offset_m, offset_n, obj ) \
{ \
	bli_obj_inc_off( BLIS_M, offset_m, obj ); \
	bli_obj_inc_off( BLIS_N, offset_n, obj ); \
}



// Diagonal offset query

#define bli_obj_diag_offset( obj ) \
\
	( (obj).diag_off )

#define bli_obj_diag_offset_after_trans( obj ) \
\
	( bli_obj_has_trans( obj ) ? -bli_obj_diag_offset( obj ) \
	                           :  bli_obj_diag_offset( obj ) )

#define bli_obj_has_main_diag( obj ) \
\
	( bli_obj_diag_offset( obj ) == 0 )

#define bli_obj_is_strictly_above_diag( obj ) \
\
	( ( doff_t )bli_obj_length( obj ) <= -bli_obj_diag_offset( obj ) )

#define bli_obj_is_strictly_below_diag( obj ) \
\
	( ( doff_t )bli_obj_width( obj )  <=  bli_obj_diag_offset( obj ) )

#define bli_obj_is_outside_diag( obj ) \
\
	( bli_obj_is_strictly_above_diag( obj ) || \
	  bli_obj_is_strictly_below_diag( obj ) )

#define bli_obj_intersects_diag( obj ) \
\
	( !bli_obj_is_strictly_above_diag( obj ) && \
	  !bli_obj_is_strictly_below_diag( obj ) )

#define bli_obj_is_unstored_subpart( obj ) \
\
	( ( bli_obj_root_is_lower( obj ) && bli_obj_is_strictly_above_diag( obj ) ) || \
	  ( bli_obj_root_is_upper( obj ) && bli_obj_is_strictly_below_diag( obj ) ) )


// Diagonal offset modification

#define bli_obj_set_diag_offset( offset, obj ) \
{ \
	(obj).diag_off  = ( doff_t )(offset); \
}

#define bli_obj_negate_diag_offset( obj ) \
{ \
	(obj).diag_off  = -(obj).diag_off; \
}

#define bli_obj_inc_diag_off( offset, obj ) \
{ \
	(obj).diag_off += ( doff_t )(offset); \
}


// Buffer address query

#define bli_obj_buffer( obj ) \
\
	( (obj).buffer )

// Buffer address modification

#define bli_obj_set_buffer( buf, obj ) \
{ \
	(obj).buffer = buf; \
}


// Bufferless scalar field query

#define bli_obj_internal_scalar_buffer( obj ) \
\
	&( (obj).scalar )

// Bufferless scalar field modification

#define bli_obj_set_internal_scalar( val, obj ) \
{ \
	(obj).scalar = val; \
}

#define bli_obj_copy_internal_scalar( a, b ) \
{ \
	(b).scalar = (a).scalar; \
}

// Element size query

#define bli_obj_elem_size( obj ) \
\
	( (obj).elem_size )

// Element size modification

#define bli_obj_set_elem_size( size, obj ) \
{ \
	(obj).elem_size = size; \
}


// Pack mem_t entry query

#define bli_obj_pack_mem( obj ) \
\
	( &((obj).pack_mem) )

// Pack mem_t entry modification

#define bli_obj_set_pack_mem( mem_p, obj ) \
{ \
	(obj).pack_mem = *mem_p; \
}


// Packed matrix info query

#define bli_obj_padded_length( obj ) \
\
	( (obj).m_padded )

#define bli_obj_padded_width( obj ) \
\
	( (obj).n_padded )

// Packed matrix info modification

#define bli_obj_set_padded_length( m0, obj ) \
{ \
    (obj).m_padded = m0; \
}

#define bli_obj_set_padded_width( n0, obj ) \
{ \
    (obj).n_padded = n0; \
}

#define bli_obj_set_padded_dims( m0, n0, obj ) \
{ \
	bli_obj_set_padded_length( m0, obj ); \
	bli_obj_set_padded_width( n0, obj ); \
}


// Packed panel info query

#define bli_obj_panel_length( obj ) \
\
	( (obj).m_panel )

#define bli_obj_panel_width( obj ) \
\
	( (obj).n_panel )

#define bli_obj_panel_dim( obj ) \
\
	( (obj).pd )

#define bli_obj_panel_stride( obj ) \
\
	( (obj).ps )

// Packed panel info modification

#define bli_obj_set_panel_length( m0, obj ) \
{ \
	(obj).m_panel = m0; \
}

#define bli_obj_set_panel_width( n0, obj ) \
{ \
	(obj).n_panel = n0; \
}

#define bli_obj_set_panel_dim( panel_dim, obj ) \
{ \
	(obj).pd = panel_dim; \
}

#define bli_obj_set_panel_stride( panel_stride, obj ) \
{ \
	(obj).ps = panel_stride; \
}

 

// -- Miscellaneous object macros --

// Make a special alias (shallow copy) that does not overwrite pack_mem
// entry.

#define bli_obj_alias_for_packing( a, b ) \
{ \
	bli_obj_init_basic_shallow_copy_of( a, b ); \
}

// Make a full alias (shallow copy), including pack_mem and friends

#define bli_obj_alias_to( a, b ) \
{ \
	bli_obj_init_full_shallow_copy_of( a, b ); \
}

// Check if two objects are aliases of one another

#define bli_obj_is_alias_of( a, b ) \
\
	( (b).buffer == (a).buffer )

// Create an alias with a trans value applied.
// (Note: trans may include a conj component.)

#define bli_obj_alias_with_trans( trans, a, b ) \
{ \
	bli_obj_alias_to( a, b ); \
	bli_obj_apply_trans( trans, b ); \
}

// Create an alias with a conj value applied.

#define bli_obj_alias_with_conj( conja, a, b ) \
{ \
	bli_obj_alias_to( a, b ); \
	bli_obj_apply_conj( conja, b ); \
}


// Initialize object with default properties (info field)

#define bli_obj_set_defaults( obj ) \
{ \
	(obj).info = 0x0; \
	(obj).info = (obj).info | BLIS_BITVAL_DENSE | BLIS_BITVAL_GENERAL; \
}


// Initialize object for packing purposes

#define bli_obj_init_pack( obj_p ) \
{ \
	mem_t* pack_mem_ = bli_obj_pack_mem( *obj_p ); \
\
	bli_mem_set_buffer( NULL, pack_mem_ ); \
}


// Release object's pack mem_t entries back to memory manager

#define bli_obj_release_pack( obj_p ) \
{ \
	mem_t* pack_mem_ = bli_obj_pack_mem( *(obj_p) ); \
\
	if ( bli_mem_is_alloc( pack_mem_ ) ) \
		bli_membrk_release( pack_mem_ ); \
}



// Submatrix/scalar buffer acquisition

#define BLIS_CONSTANT_SLOT_SIZE  BLIS_MAX_TYPE_SIZE
#define BLIS_CONSTANT_SIZE       ( 5 * BLIS_CONSTANT_SLOT_SIZE )

#define bli_obj_buffer_for_const( dt, obj ) \
\
	( void* )( \
	           ( ( char* )( bli_obj_buffer( obj ) ) ) + \
                 ( dim_t )( dt * BLIS_CONSTANT_SLOT_SIZE ) \
	         )

#define bli_obj_buffer_at_off( obj ) \
\
	( void* )( \
	           ( ( char* )( bli_obj_buffer   ( obj ) ) + \
                 ( dim_t )( bli_obj_elem_size( obj ) ) * \
                            ( bli_obj_col_off( obj ) * bli_obj_col_stride( obj ) + \
                              bli_obj_row_off( obj ) * bli_obj_row_stride( obj ) \
                            ) \
               ) \
	         )

#define bli_obj_buffer_for_1x1( dt, obj ) \
\
	( void* )( bli_obj_is_const( obj ) ? ( bli_obj_buffer_for_const( dt, obj ) ) \
	                                   : ( bli_obj_buffer_at_off( obj ) ) \
	         )


// Swap objects

#define bli_obj_swap( a, b ) \
{ \
	obj_t t_; \
	t_ = b; b = a; a = t_; \
}


// Swap object pointers

#define bli_obj_swap_pointers( a, b ) \
{ \
	obj_t* t_; \
	t_ = b; b = a; a = t_; \
}


// If a transposition is needed, induce one: swap dimensions, increments
// and offsets, and then clear the trans bit.

#define bli_obj_induce_trans( obj ) \
{ \
	{ \
		dim_t  m_        = bli_obj_length( obj ); \
		dim_t  n_        = bli_obj_width( obj ); \
		inc_t  rs_       = bli_obj_row_stride( obj ); \
		inc_t  cs_       = bli_obj_col_stride( obj ); \
		dim_t  offm_     = bli_obj_row_off( obj ); \
		dim_t  offn_     = bli_obj_col_off( obj ); \
		doff_t diag_off_ = bli_obj_diag_offset( obj ); \
\
		bli_obj_set_dims( n_, m_, obj ); \
		bli_obj_set_strides( cs_, rs_, obj ); \
		bli_obj_set_offs( offn_, offm_, obj ); \
		bli_obj_set_diag_offset( -diag_off_, obj ); \
\
		if ( bli_obj_is_upper_or_lower( obj ) ) \
			bli_obj_toggle_uplo( obj ); \
\
		/* Note that this macro DOES NOT touch the transposition bit! If
		   the calling code is using this macro to handle an object whose
		   transposition bit is set prior to computation, that code needs
		   to manually clear or toggle the bit, via
		   bli_obj_set_onlytrans() or bli_obj_toggle_trans(),
		   respectively. */ \
	} \
}

// Sometimes we need to "reflect" a partition because the data we want is
// actually stored on the other side of the diagonal. The nuts and bolts of
// this macro look a lot like an induced transposition, except that the row
// and column strides are left unchanged (which, of course, drastically
// changes the effect of the macro).

#define bli_obj_reflect_about_diag( obj ) \
{ \
	{ \
		dim_t  m_        = bli_obj_length( obj ); \
		dim_t  n_        = bli_obj_width( obj ); \
		dim_t  offm_     = bli_obj_row_off( obj ); \
		dim_t  offn_     = bli_obj_col_off( obj ); \
		doff_t diag_off_ = bli_obj_diag_offset( obj ); \
\
		bli_obj_set_dims( n_, m_, obj ); \
		bli_obj_set_offs( offn_, offm_, obj ); \
		bli_obj_set_diag_offset( -diag_off_, obj ); \
\
		bli_obj_toggle_trans( obj ); \
	} \
}

#endif
