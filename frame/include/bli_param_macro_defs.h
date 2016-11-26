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

#ifndef BLIS_PARAM_MACRO_DEFS_H
#define BLIS_PARAM_MACRO_DEFS_H


// -- Parameter query macros --

// buffer

#define bli_is_aligned_to( p, size ) \
\
	( ( uintptr_t )(p) % ( uintptr_t )(size) == 0 )

#define bli_is_unaligned_to( p, size ) \
\
	( ( uintptr_t )(p) % ( uintptr_t )(size) != 0 )

#define bli_offset_past_alignment( p, size ) \
\
	( ( uintptr_t )(p) % ( uintptr_t )(size) )


// datatype

#define bli_is_float( dt ) \
\
	( dt == BLIS_FLOAT )

#define bli_is_double( dt ) \
\
	( dt == BLIS_DOUBLE )

#define bli_is_scomplex( dt ) \
\
	( dt == BLIS_SCOMPLEX )

#define bli_is_dcomplex( dt ) \
\
	( dt == BLIS_DCOMPLEX )

#define bli_is_constant( dt ) \
\
	( dt == BLIS_CONSTANT )

#define bli_is_int( dt ) \
\
	( dt == BLIS_INT )

#define bli_is_real( dt ) \
\
    ( bli_is_float( dt ) || \
	  bli_is_double( dt ) )

#define bli_is_complex( dt ) \
\
    ( bli_is_scomplex( dt ) || \
	  bli_is_dcomplex( dt ) )

#define bli_is_single_prec( dt ) \
\
    ( bli_is_float( dt ) || \
	  bli_is_scomplex( dt ) )

#define bli_is_double_prec( dt ) \
\
    ( bli_is_double( dt ) || \
	  bli_is_dcomplex( dt ) )

#define bli_datatype_proj_to_real( dt ) \
\
	( dt & ~BLIS_BITVAL_COMPLEX )

#define bli_datatype_proj_to_complex( dt ) \
\
	( dt &  BLIS_BITVAL_COMPLEX )

#define bli_domain_of_dt( dt ) \
\
	( dt & BLIS_DOMAIN_BIT )


// side

#define bli_is_left( side ) \
\
    ( side == BLIS_LEFT )

#define bli_is_right( side ) \
\
    ( side == BLIS_RIGHT )

#define bli_side_toggled( side ) \
\
	( bli_is_left( side ) ? BLIS_RIGHT : BLIS_LEFT )

#define bli_toggle_side( side ) \
{ \
	side = bli_side_toggled( side ); \
}


// uplo

#define bli_is_lower( uplo ) \
\
    ( uplo == BLIS_LOWER )

#define bli_is_upper( uplo ) \
\
    ( uplo == BLIS_UPPER )

#define bli_is_upper_or_lower( uplo ) \
\
    ( bli_is_upper( uplo ) || \
	  bli_is_lower( uplo ) )

#define bli_is_dense( uplo ) \
\
    ( uplo == BLIS_DENSE )

#define bli_is_zeros( uplo ) \
\
    ( uplo == BLIS_ZEROS )

#define bli_uplo_toggled( uplo ) \
\
	( bli_is_upper_or_lower( uplo ) ? \
	  ( ( uplo ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT ) : uplo \
	)

#define bli_toggle_uplo( uplo ) \
{ \
	uplo = bli_uplo_toggled( uplo ); \
}

#define bli_set_uplo_with_trans( trans, uplo, uplo_trans ) \
{ \
	if ( bli_does_notrans( trans ) ) { uplo_trans = uplo;                     } \
	else                             { uplo_trans = bli_uplo_toggled( uplo ); } \
}


// structure

#define bli_is_general( struc ) \
\
	( struc == BLIS_GENERAL )

#define bli_is_hermitian( struc ) \
\
	( struc == BLIS_HERMITIAN )

#define bli_is_symmetric( struc ) \
\
	( struc == BLIS_SYMMETRIC )

#define bli_is_triangular( struc ) \
\
	( struc == BLIS_TRIANGULAR )

#define bli_is_herm_or_symm( struc ) \
\
	( bli_is_hermitian( struc ) || \
	  bli_is_symmetric( struc ) )



// conj

#define bli_is_noconj( conjval ) \
\
    ( conjval == BLIS_NO_CONJUGATE )

#define bli_is_conj( conjval ) \
\
    ( conjval == BLIS_CONJUGATE )

#define bli_conj_toggled( conjval ) \
\
	( conjval ^ BLIS_CONJ_BIT )

#define bli_apply_conj( conjapp, conjval )\
\
	( conjval ^ (conjapp) )

#define bli_toggle_conj( conjval ) \
{ \
	conjval = bli_conj_toggled( conjval ); \
}


// trans

#define bli_is_notrans( trans ) \
\
    ( trans == BLIS_NO_TRANSPOSE )

#define bli_is_trans( trans ) \
\
    ( trans == BLIS_TRANSPOSE )

#define bli_is_conjnotrans( trans ) \
\
    ( trans == BLIS_CONJ_NO_TRANSPOSE )

#define bli_is_conjtrans( trans ) \
\
    ( trans == BLIS_CONJ_TRANSPOSE )

#define bli_does_notrans( trans ) \
\
	( (~(trans) & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS )

#define bli_does_trans( trans ) \
\
	( (  trans  & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS )

#define bli_does_noconj( trans ) \
\
	( (~(trans) & BLIS_CONJ_BIT  ) == BLIS_BITVAL_CONJ )

#define bli_does_conj( trans ) \
\
	( (  trans  & BLIS_CONJ_BIT  ) == BLIS_BITVAL_CONJ )

#define bli_extract_trans( trans ) \
\
	(    trans  & BLIS_TRANS_BIT  )

#define bli_extract_conj( trans ) \
\
	(    trans  & BLIS_CONJ_BIT  )

#define bli_trans_toggled( trans ) \
\
	( trans ^ BLIS_TRANS_BIT )

#define bli_toggle_trans( trans ) \
{ \
	trans = bli_trans_toggled( trans ); \
}


// diag

#define bli_is_nonunit_diag( diag ) \
\
    ( diag == BLIS_NONUNIT_DIAG )

#define bli_is_unit_diag( diag ) \
\
    ( diag == BLIS_UNIT_DIAG )


// dimension-related

#define bli_zero_dim1( n ) \
\
	( (n) == 0 )

#define bli_zero_dim2( m, n ) \
\
	( (m) == 0 || (n) == 0 )

#define bli_zero_dim3( m, n, k ) \
\
	( (m) == 0 || (n) == 0 || (k) == 0 )

#define bli_nonzero_dim( n ) \
\
	( (n) > 0 )

#define bli_vector_dim( m, n ) \
\
	( (m) == 1 ? (n) : (m) )

#define bli_is_vector( m, n ) \
\
	( (m) == 1 || (n) == 1 )

#define bli_is_row_vector( m, n ) \
\
	( (m) == 1 )

#define bli_is_col_vector( m, n ) \
\
	( (n) == 1 )

#define bli_set_dim_with_side( side, m, n, dim ) \
{ \
	if ( bli_is_left( side ) ) { dim = m; } \
	else                       { dim = n; } \
}

#define bli_set_dims_with_trans( trans, m, n, mtrans, ntrans ) \
{ \
	if ( bli_does_notrans( trans ) ) { mtrans = m; ntrans = n; } \
	else                             { mtrans = n; ntrans = m; } \
}

#define bli_set_dims_incs_with_trans( trans, m, n, rs, cs, mt, nt, rst, cst ) \
{ \
	if ( bli_does_notrans( trans ) ) { mt = m; nt = n; rst = rs; cst = cs; } \
	else                             { mt = n; nt = m; rst = cs; cst = rs; } \
}


// blocksize-related

#define bli_determine_blocksize_dim_f( i, dim, b_alg ) \
\
	( bli_min( b_alg, dim - i ) )

#define bli_determine_blocksize_dim_b( i, dim, b_alg ) \
\
	( i == 0 && dim % b_alg != 0 ? dim % b_alg \
	                             : b_alg )


// stride-related

#define bli_vector_inc( trans, m, n, rs, cs ) \
\
	( bli_does_notrans( trans ) ? ( m == 1 ? (cs) : (rs) ) \
                                : ( m == 1 ? (rs) : (cs) ) )

#define bli_is_row_stored( rs, cs ) \
\
	( bli_abs( cs ) == 1 )

#define bli_is_col_stored( rs, cs ) \
\
	( bli_abs( rs ) == 1 )

#define bli_is_row_stored_f( m, n, rs, cs ) \
\
	( cs == 1 && ( rs > 1 || n == 1 ) )

#define bli_is_col_stored_f( m, n, rs, cs ) \
\
	( rs == 1 && ( cs > 1 || m == 1 ) )

#define bli_is_gen_stored( rs, cs ) \
\
	( bli_abs( rs ) != 1 && \
	  bli_abs( cs ) != 1 )

#define bli_is_row_tilted( m, n, rs, cs ) \
\
	( bli_abs( cs ) == bli_abs( rs ) ? n < m \
	                                 : bli_abs( cs ) < bli_abs( rs ) )

#define bli_is_col_tilted( m, n, rs, cs ) \
\
	( bli_abs( rs ) == bli_abs( cs ) ? m < n \
	                                 : bli_abs( rs ) < bli_abs( cs ) )

#define bli_has_nonunit_inc1( inc1 ) \
\
	( inc1 != 1 )

#define bli_has_nonunit_inc2( inc1, inc2 ) \
\
	( inc1 != 1 || inc2 != 1 )

#define bli_has_nonunit_inc3( inc1, inc2, inc3 ) \
\
	( inc1 != 1 || inc2 != 1 || inc3 != 1 )


// diag offset-related

#define bli_negate_diag_offset( diagoff ) \
{ \
	diagoff = -diagoff; \
}

#define bli_shift_diag_offset_to_grow_uplo( uplo, diagoff ) \
{ \
	if      ( bli_is_upper( uplo ) ) diagoff -= 1; \
	else if ( bli_is_lower( uplo ) ) diagoff += 1; \
}

#define bli_shift_diag_offset_to_shrink_uplo( uplo, diagoff ) \
{ \
	if      ( bli_is_upper( uplo ) ) diagoff += 1; \
	else if ( bli_is_lower( uplo ) ) diagoff -= 1; \
}

#define bli_diag_offset_with_trans( trans, diagoff ) \
\
	( bli_does_trans( trans ) ? -diagoff : diagoff )

#define bli_is_strictly_above_diag( diagoff, trans, m, n ) \
\
	( bli_does_trans( trans ) ? ( ( doff_t )n <= -diagoff ) \
	                          : ( ( doff_t )m <= -diagoff ) )

#define bli_is_strictly_below_diag( diagoff, trans, m, n ) \
\
	( bli_does_trans( trans ) ? ( ( doff_t )m <=  diagoff ) \
	                          : ( ( doff_t )n <=  diagoff ) )

#define bli_is_outside_diag( diagoff, trans, m, n ) \
\
	( bli_is_strictly_above_diag( diagoff, trans, m, n ) || \
	  bli_is_strictly_below_diag( diagoff, trans, m, n ) )

#define bli_is_stored_subpart( diagoff, trans, uplo, m, n ) \
\
	( ( bli_is_upper( uplo ) && bli_is_strictly_above_diag( diagoff, trans, m, n ) ) || \
	  ( bli_is_lower( uplo ) && bli_is_strictly_below_diag( diagoff, trans, m, n ) ) )

#define bli_is_unstored_subpart( diagoff, trans, uplo, m, n ) \
\
	( ( bli_is_upper( uplo ) && bli_is_strictly_below_diag( diagoff, trans, m, n ) ) || \
	  ( bli_is_lower( uplo ) && bli_is_strictly_above_diag( diagoff, trans, m, n ) ) )

#define bli_is_strictly_above_diag_n( diagoff, m, n ) \
\
	( ( doff_t )m <= -diagoff ) \

#define bli_is_strictly_below_diag_n( diagoff, m, n ) \
\
	( ( doff_t )n <=  diagoff ) \

#define bli_intersects_diag_n( diagoff, m, n ) \
\
	( !bli_is_strictly_above_diag_n( diagoff, m, n ) && \
	  !bli_is_strictly_below_diag_n( diagoff, m, n ) )

#define bli_is_stored_subpart_n( diagoff, uplo, m, n ) \
\
	( ( bli_is_upper( uplo ) && bli_is_strictly_above_diag_n( diagoff, m, n ) ) || \
	  ( bli_is_lower( uplo ) && bli_is_strictly_below_diag_n( diagoff, m, n ) ) )

#define bli_is_unstored_subpart_n( diagoff, uplo, m, n ) \
\
	( ( bli_is_upper( uplo ) && bli_is_strictly_below_diag_n( diagoff, m, n ) ) || \
	  ( bli_is_lower( uplo ) && bli_is_strictly_above_diag_n( diagoff, m, n ) ) )


// pruning-related

#define bli_prune_unstored_region_top_l( diagoff, m, n, offm_inc ) \
{ \
	offm_inc = 0; \
\
	/* If the diagonal intersects the left side of the matrix,
	   ignore the area above that intersection. */ \
	if ( diagoff < 0 ) \
	{ \
		m        = m    + diagoff; \
		offm_inc =      - diagoff; \
		diagoff  = 0; \
	} \
}

#define bli_prune_unstored_region_right_l( diagoff, m, n, offn_inc ) \
{ \
	offn_inc = 0; \
\
	/* If the diagonal intersects the bottom side of the matrix,
	   ignore the area to the right of that intersection. */ \
	if ( n > diagoff + m ) \
	{ \
		n = diagoff + m; \
	} \
}

#define bli_prune_unstored_region_left_u( diagoff, m, n, offn_inc ) \
{ \
	offn_inc = 0; \
\
	/* If the diagonal intersects the top side of the matrix,
	   ignore the area to the left of that intersection. */ \
	if ( diagoff > 0 ) \
	{ \
		n        = n    - diagoff; \
		offn_inc =      + diagoff; \
		diagoff  = 0; \
	} \
}

#define bli_prune_unstored_region_bottom_u( diagoff, m, n, offm_inc ) \
{ \
	offm_inc = 0; \
\
	/* If the diagonal intersects the right side of the matrix,
	   ignore the area below that intersection. */ \
	if ( m > -diagoff + n ) \
	{ \
		m = -diagoff + n; \
	} \
}


// thread range-related

#define bli_rotate180_trapezoid( diagoff, uplo ) \
{ \
	diagoff = n - diagoff - m; \
	bli_toggle_uplo( uplo ); \
}

#define bli_reverse_index_direction( start, end, n ) \
{ \
	dim_t start2_ = n - start; \
	dim_t end2_   = n - end; \
	start = end2_; \
	end   = start2_; \
}

#define bli_reflect_about_diag( diagoff, uplo, m, n ) \
{ \
	bli_swap_dims( m, n ); \
	bli_negate_diag_offset( diagoff ); \
	bli_toggle_uplo( uplo ); \
}


// mdim_t-related

#define bli_is_m_dim( mdim ) \
\
	( mdim == BLIS_M )

#define bli_is_n_dim( mdim ) \
\
	( mdim == BLIS_N )

#define bli_dim_toggled( mdim ) \
\
	( mdim == BLIS_M ? BLIS_N : BLIS_M )

#define bli_toggle_dim( mdim ) \
{ \
	mdim = bli_dim_toggled( mdim ); \
}



// index-related

#define bli_is_edge_f( i1, iter, left ) \
\
	( i1 == iter - 1 && left != 0 )

#define bli_is_not_edge_f( i1, iter, left ) \
\
	( i1 != iter - 1 || left == 0 )

#define bli_is_edge_b( i1, iter, left ) \
\
	( i1 == 0 && left != 0 )

#define bli_is_not_edge_b( i1, iter, left ) \
\
	( i1 != 0 || left == 0 )

#define bli_is_last_iter( i1, niter, tid, nth ) \
\
	( i1 == niter - 1 - ( ( niter - tid - 1 ) % nth ) ) \
	/*( i1 == niter - 1 )*/


// packbuf_t-related

#define bli_packbuf_index( buf_type ) \
\
	( ( (buf_type) & BLIS_PACK_BUFFER_BITS ) >> BLIS_PACK_BUFFER_SHIFT )


// pack_t-related

#define bli_is_packed( schema ) \
\
	( ( schema & BLIS_PACK_BIT  ) )

#define bli_is_row_packed( schema ) \
\
	( ( schema & BLIS_PACK_RC_BIT  ) == ( BLIS_BITVAL_PACKED_UNSPEC ^ \
	                                      BLIS_BITVAL_PACKED_ROWS    ) )

#define bli_is_col_packed( schema ) \
\
	( ( schema & BLIS_PACK_RC_BIT  ) == ( BLIS_BITVAL_PACKED_UNSPEC ^ \
	                                      BLIS_BITVAL_PACKED_COLUMNS ) )

#define bli_is_panel_packed( schema ) \
\
	( ( schema & BLIS_PACK_PANEL_BIT ) )

#define bli_is_4mi_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_4MI )

#define bli_is_3mi_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_3MI )

#define bli_is_3ms_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_3MS )

#define bli_is_ro_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_RO )

#define bli_is_io_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_IO )

#define bli_is_rpi_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_RPI )

#define bli_is_rih_packed( schema ) \
\
	( bli_is_ro_packed( schema ) || \
	  bli_is_io_packed( schema ) || \
	  bli_is_rpi_packed( schema ) )

#define bli_is_1r_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_1R )

#define bli_is_1e_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_1E )

#define bli_is_1m_packed( schema ) \
\
	( bli_is_1r_packed( schema ) || \
	  bli_is_1e_packed( schema ) )

#define bli_is_nat_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) == 0 )

#define bli_is_ind_packed( schema ) \
\
	( ( schema & BLIS_PACK_FORMAT_BITS ) != 0 )

#define bli_pack_schema_index( schema ) \
\
	( ( (schema) & BLIS_PACK_FORMAT_BITS ) >> BLIS_PACK_FORMAT_SHIFT )



// pointer-related

// p1 = p0 + (num/dem)
#define bli_ptr_add( p1, p0, num, dem ) \
{ \
	p1 = ( typeof( p1 ) ) \
	     ( ( char* )(p0) + ( (   (num) * sizeof( *(p0) ) \
	                         ) / (dem) \
	                       ) \
	     ); \
}

#define bli_is_null( p ) \
\
	( p == NULL )

#define bli_is_nonnull( p ) \
\
	( p != NULL )


// return datatype for char

#define bli_stype ( BLIS_FLOAT    )
#define bli_dtype ( BLIS_DOUBLE   )
#define bli_ctype ( BLIS_SCOMPLEX )
#define bli_ztype ( BLIS_DCOMPLEX )


// return datatype "union" for char pair

#define bli_sstypeunion() ( BLIS_FLOAT    )
#define bli_sdtypeunion() ( BLIS_DOUBLE   )
#define bli_sctypeunion() ( BLIS_SCOMPLEX )
#define bli_sztypeunion() ( BLIS_DCOMPLEX )

#define bli_dstypeunion() ( BLIS_DOUBLE   )
#define bli_ddtypeunion() ( BLIS_DOUBLE   )
#define bli_dctypeunion() ( BLIS_DCOMPLEX )
#define bli_dztypeunion() ( BLIS_DCOMPLEX )

#define bli_cstypeunion() ( BLIS_SCOMPLEX )
#define bli_cdtypeunion() ( BLIS_DCOMPLEX )
#define bli_cctypeunion() ( BLIS_SCOMPLEX )
#define bli_cztypeunion() ( BLIS_DCOMPLEX )

#define bli_zstypeunion() ( BLIS_DCOMPLEX )
#define bli_zdtypeunion() ( BLIS_DCOMPLEX )
#define bli_zctypeunion() ( BLIS_DCOMPLEX )
#define bli_zztypeunion() ( BLIS_DCOMPLEX )


// return default format specifier for char

#define bli_sformatspec() "%9.2e"
#define bli_dformatspec() "%9.2e"
#define bli_cformatspec() "%9.2e + %9.2e "
#define bli_zformatspec() "%9.2e + %9.2e "
#define bli_iformatspec() "%6d"


// set scalar datatype and buffer

#define bli_set_scalar_dt_buffer( obj_scalar, dt_aux, dt_scalar, buf_scalar ) \
{ \
	if ( bli_obj_is_const( *(obj_scalar) ) ) \
	{ \
		dt_scalar  = dt_aux; \
		buf_scalar = bli_obj_buffer_for_1x1( dt_scalar, *(obj_scalar) ); \
	} \
	else \
	{ \
		dt_scalar  = bli_obj_datatype( *(obj_scalar) ); \
		buf_scalar = bli_obj_buffer_at_off( *(obj_scalar) ); \
	} \
}

// set constant datatype and buffer

#define bli_set_const_dt_buffer( obj_scalar, dt_aux, dt_scalar, buf_scalar ) \
{ \
	{ \
		dt_scalar  = dt_aux; \
		buf_scalar = bli_obj_buffer_for_1x1( dt_scalar, *(obj_scalar) ); \
	} \
}


// Set dimensions, increments, effective uplo/diagoff, etc for ONE matrix
// argument.

#define bli_set_dims_incs_uplo_1m( diagoffa, diaga, \
                                   uploa,    m,          n,      rs_a, cs_a, \
                                   uplo_eff, n_elem_max, n_iter, inca, lda, \
                                   ij0, n_shift ) \
{ \
	/* This is to prevent the compiler from warning about uninitialized
	   variables. */ \
	ij0     = 0; \
	n_shift = 0; \
\
	/* If matrix A is entirely "unstored", that is, if either:
	   - A is lower-stored and entirely above the diagonal, or
	   - A is upper-stored and entirely below the diagonal
	   then we mark the storage as implicitly zero. */ \
	if ( bli_is_unstored_subpart( diagoffa, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
	{ \
		uplo_eff = BLIS_ZEROS; \
	} \
	else \
	{ \
		doff_t diagoffa_use_ = diagoffa; \
		doff_t diagoff_eff_; \
		dim_t  n_iter_max_; \
\
		if ( bli_is_unit_diag( diaga ) ) \
			bli_shift_diag_offset_to_shrink_uplo( uploa, diagoffa_use_ ); \
\
		/* If matrix A is entirely "stored", that is, if either:
		   - A is upper-stored and entirely above the diagonal, or
		   - A is lower-stored and entirely below the diagonal
		   then we mark the storage as dense. */ \
		if ( bli_is_stored_subpart( diagoffa_use_, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
			uploa = BLIS_DENSE; \
\
		n_iter_max_  = n; \
		n_elem_max  = m; \
		inca        = rs_a; \
		lda         = cs_a; \
		uplo_eff    = uploa; \
		diagoff_eff_ = diagoffa_use_; \
\
		if ( bli_is_row_tilted( n_elem_max, n_iter_max_, inca, lda ) ) \
		{ \
			bli_swap_dims( n_iter_max_, n_elem_max ); \
			bli_swap_incs( inca, lda ); \
			bli_toggle_uplo( uplo_eff ); \
			bli_negate_diag_offset( diagoff_eff_ ); \
		} \
\
		if ( bli_is_dense( uplo_eff ) ) \
		{ \
			n_iter = n_iter_max_; \
		} \
		else if ( bli_is_upper( uplo_eff ) ) \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
				ij0        = 0; \
				n_shift    = -diagoff_eff_; \
				n_elem_max = bli_min( n_elem_max, n_shift + bli_min( m, n ) ); \
				n_iter     = n_iter_max_; \
			} \
			else \
			{ \
				ij0        = diagoff_eff_; \
				n_shift    = 0; \
				n_iter     = n_iter_max_ - diagoff_eff_; \
			} \
		} \
		else /* if ( bli_is_lower( uplo_eff ) ) */ \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
				ij0        = -diagoff_eff_; \
				n_shift    = 0; \
				n_elem_max = n_elem_max + diagoff_eff_; \
				n_iter     = bli_min( n_elem_max, bli_min( m, n ) ); \
			} \
			else \
			{ \
				ij0        = 0; \
				n_shift    = diagoff_eff_; \
				n_iter     = bli_min( n_iter_max_, n_shift + bli_min( m, n ) ); \
			} \
		} \
	} \
}

// Set dimensions, increments, effective uplo/diagoff, etc for ONE matrix
// argument (without column-wise stride optimization).

#define bli_set_dims_incs_uplo_1m_noswap( \
          diagoffa, diaga, \
          uploa,    m,          n,      rs_a, cs_a, \
          uplo_eff, n_elem_max, n_iter, inca, lda, \
          ij0, n_shift \
        ) \
{ \
	/* This is to prevent the compiler from warning about uninitialized
	   variables. */ \
	ij0     = 0; \
	n_shift = 0; \
\
	/* If matrix A is entirely "unstored", that is, if either:
	   - A is lower-stored and entirely above the diagonal, or
	   - A is upper-stored and entirely below the diagonal
	   then we mark the storage as implicitly zero. */ \
	if ( bli_is_unstored_subpart( diagoffa, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
	{ \
		uplo_eff = BLIS_ZEROS; \
	} \
	else \
	{ \
		doff_t diagoffa_use_ = diagoffa; \
		doff_t diagoff_eff_; \
		dim_t  n_iter_max_; \
\
		if ( bli_is_unit_diag( diaga ) ) \
			bli_shift_diag_offset_to_shrink_uplo( uploa, diagoffa_use_ ); \
\
		/* If matrix A is entirely "stored", that is, if either:
		   - A is upper-stored and entirely above the diagonal, or
		   - A is lower-stored and entirely below the diagonal
		   then we mark the storage as dense. */ \
		if ( bli_is_stored_subpart( diagoffa_use_, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
			uploa = BLIS_DENSE; \
\
		n_iter_max_  = n; \
		n_elem_max  = m; \
		inca        = rs_a; \
		lda         = cs_a; \
		uplo_eff    = uploa; \
		diagoff_eff_ = diagoffa_use_; \
\
		if ( bli_is_dense( uplo_eff ) ) \
		{ \
			n_iter = n_iter_max_; \
		} \
		else if ( bli_is_upper( uplo_eff ) ) \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
				ij0        = 0; \
				n_shift    = -diagoff_eff_; \
				n_elem_max = bli_min( n_elem_max, n_shift + bli_min( m, n ) ); \
				n_iter     = n_iter_max_; \
			} \
			else \
			{ \
				ij0        = diagoff_eff_; \
				n_shift    = 0; \
				n_iter     = n_iter_max_ - diagoff_eff_; \
			} \
		} \
		else /* if ( bli_is_lower( uplo_eff ) ) */ \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
				ij0        = -diagoff_eff_; \
				n_shift    = 0; \
				n_elem_max = n_elem_max + diagoff_eff_; \
				n_iter     = bli_min( n_elem_max, bli_min( m, n ) ); \
			} \
			else \
			{ \
				ij0        = 0; \
				n_shift    = diagoff_eff_; \
				n_iter     = bli_min( n_iter_max_, n_shift + bli_min( m, n ) ); \
			} \
		} \
	} \
}

// Set dimensions, increments, effective uplo/diagoff, etc for TWO matrix
// arguments.

#define bli_set_dims_incs_uplo_2m( \
          diagoffa, diaga, transa, \
          uploa,    m,          n,      rs_a, cs_a, rs_b, cs_b, \
          uplo_eff, n_elem_max, n_iter, inca, lda,  incb, ldb, \
          ij0, n_shift \
        ) \
{ \
	/* This is to prevent the compiler from warning about uninitialized
	   variables. */ \
	ij0     = 0; \
	n_shift = 0; \
\
	/* If matrix A is entirely "unstored", that is, if either:
	   - A is lower-stored and entirely above the diagonal, or
	   - A is upper-stored and entirely below the diagonal
	   then we mark the storage as implicitly zero. */ \
	if ( bli_is_unstored_subpart( diagoffa, transa, uploa, m, n ) ) \
	{ \
		uplo_eff = BLIS_ZEROS; \
	} \
	else \
	{ \
		doff_t diagoffa_use_ = diagoffa; \
		doff_t diagoff_eff_; \
		dim_t  n_iter_max_; \
\
		if ( bli_is_unit_diag( diaga ) ) \
			bli_shift_diag_offset_to_shrink_uplo( uploa, diagoffa_use_ ); \
\
		/* If matrix A is entirely "stored", that is, if either:
		   - A is upper-stored and entirely above the diagonal, or
		   - A is lower-stored and entirely below the diagonal
		   then we mark the storage as dense. */ \
		if ( bli_is_stored_subpart( diagoffa_use_, transa, uploa, m, n ) ) \
			uploa = BLIS_DENSE; \
\
		n_iter_max_  = n; \
		n_elem_max  = m; \
		inca        = rs_a; \
		lda         = cs_a; \
		incb        = rs_b; \
		ldb         = cs_b; \
		uplo_eff    = uploa; \
		diagoff_eff_ = diagoffa_use_; \
\
		if ( bli_does_trans( transa ) ) \
		{ \
			bli_swap_incs( inca, lda ); \
			bli_toggle_uplo( uplo_eff ); \
			bli_negate_diag_offset( diagoff_eff_ ); \
		} \
\
		if ( bli_is_row_tilted( n_elem_max, n_iter_max_, incb, ldb ) && \
		     bli_is_row_tilted( n_elem_max, n_iter_max_, inca, lda ) ) \
		{ \
			bli_swap_dims( n_iter_max_, n_elem_max ); \
			bli_swap_incs( inca, lda ); \
			bli_swap_incs( incb, ldb ); \
			bli_toggle_uplo( uplo_eff ); \
			bli_negate_diag_offset( diagoff_eff_ ); \
		} \
\
		if ( bli_is_dense( uplo_eff ) ) \
		{ \
			n_iter = n_iter_max_; \
		} \
		else if ( bli_is_upper( uplo_eff ) ) \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
/*printf( "uplo_eff = upper, diagoff_eff_ < 0\n" );*/ \
				ij0        = 0; \
				n_shift    = -diagoff_eff_; \
				n_elem_max = bli_min( n_elem_max, n_shift + bli_min( m, n ) ); \
				n_iter     = n_iter_max_; \
			} \
			else \
			{ \
/*printf( "uplo_eff = upper, diagoff_eff_ >= 0\n" );*/ \
				ij0        = diagoff_eff_; \
				n_shift    = 0; \
				n_iter     = n_iter_max_ - diagoff_eff_; \
			} \
		} \
		else /* if ( bli_is_lower( uplo_eff ) ) */ \
		{ \
			if ( diagoff_eff_ < 0 ) \
			{ \
/*printf( "uplo_eff = lower, diagoff_eff_ < 0\n" );*/ \
				ij0        = -diagoff_eff_; \
				n_shift    = 0; \
				n_elem_max = n_elem_max + diagoff_eff_; \
				n_iter     = bli_min( n_elem_max, bli_min( m, n ) ); \
			} \
			else \
			{ \
/*printf( "uplo_eff = lower, diagoff_eff_ >= 0\n" );*/ \
				ij0        = 0; \
				n_shift    = diagoff_eff_; \
				n_iter     = bli_min( n_iter_max_, n_shift + bli_min( m, n ) ); \
			} \
		} \
	} \
}

// Set dimensions, increments, etc for ONE matrix argument when operating
// on the diagonal.

#define bli_set_dims_incs_1d( diagoffx, \
                              m, n, rs_x, cs_x, \
                              offx, n_elem, incx ) \
{ \
	if ( diagoffx < 0 ) \
	{ \
		n_elem = bli_min( m - ( dim_t )(-diagoffx), n ); \
		offx   = ( dim_t )(-diagoffx) * rs_x; \
	} \
	else \
	{ \
		n_elem = bli_min( n - ( dim_t )( diagoffx), m ); \
		offx   = ( dim_t )( diagoffx) * cs_x; \
	} \
\
	incx = rs_x + cs_x; \
}

// Set dimensions, increments, etc for TWO matrix arguments when operating
// on diagonals.

#define bli_set_dims_incs_2d( diagoffx, transx, \
                              m, n, rs_x, cs_x, rs_y, cs_y, \
                              offx, offy, n_elem, incx, incy ) \
{ \
	doff_t diagoffy_ = bli_diag_offset_with_trans( transx, diagoffx ); \
\
	if ( diagoffx < 0 ) offx = -diagoffx * rs_x; \
	else                offx =  diagoffx * cs_x; \
\
	if ( diagoffy_ < 0 ) \
	{ \
		n_elem = bli_min( m - ( dim_t )(-diagoffy_), n ); \
		offy   = -diagoffy_ * rs_y; \
	} \
	else \
	{ \
		n_elem = bli_min( n - ( dim_t )( diagoffy_), m ); \
		offy   = diagoffy_ * cs_y; \
	} \
\
	incx = rs_x + cs_x; \
	incy = rs_y + cs_y; \
}

// -- Function caller/chooser macros --

#define bli_call_ft_2( dt, fname, o0, o1 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1); \
}
#define bli_call_ft_3( dt, fname, o0, o1, o2 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2); \
}
#define bli_call_ft_3i( dt, fname, o0, o1, o2 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2); \
	else if ( bli_is_int( dt )      ) PASTEMAC(i,fname)(o0,o1,o2); \
}
#define bli_call_ft_4( dt, fname, o0, o1, o2, o3 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3); \
}
#define bli_call_ft_5( dt, fname, o0, o1, o2, o3, o4 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4); \
}
#define bli_call_ft_6( dt, fname, o0, o1, o2, o3, o4, o5 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5); \
}
#define bli_call_ft_7( dt, fname, o0, o1, o2, o3, o4, o5, o6 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6); \
}
#define bli_call_ft_8( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7); \
}
#define bli_call_ft_9( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8); \
}
#define bli_call_ft_10( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9); \
}
#define bli_call_ft_11( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10); \
}
#define bli_call_ft_12( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11); \
}
#define bli_call_ft_13( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12); \
}
#define bli_call_ft_14( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13); \
}
#define bli_call_ft_15( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14); \
}
#define bli_call_ft_20( dt, fname, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19 ) \
{ \
	if      ( bli_is_float( dt )    ) PASTEMAC(s,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19); \
	else if ( bli_is_double( dt )   ) PASTEMAC(d,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19); \
	else if ( bli_is_scomplex( dt ) ) PASTEMAC(c,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19); \
	else if ( bli_is_dcomplex( dt ) ) PASTEMAC(z,fname)(o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,o19); \
}



#endif
