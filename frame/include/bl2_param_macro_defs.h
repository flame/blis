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

#ifndef BLIS_PARAM_MACRO_DEFS_H
#define BLIS_PARAM_MACRO_DEFS_H


// -- Parameter query macros --

// buffer

#define bl2_is_aligned_to( p, size ) \
\
	( ( siz_t )(p) % (size) == 0 )

#define bl2_is_unaligned_to( p, size ) \
\
	( ( siz_t )(p) % (size) != 0 )


// datatype

#define bl2_is_float( dt ) \
\
	( dt == BLIS_FLOAT )

#define bl2_is_double( dt ) \
\
	( dt == BLIS_DOUBLE )

#define bl2_is_scomplex( dt ) \
\
	( dt == BLIS_SCOMPLEX )

#define bl2_is_dcomplex( dt ) \
\
	( dt == BLIS_DCOMPLEX )

#define bl2_is_real( dt ) \
\
    ( ( dt & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_REAL )

#define bl2_is_complex( dt ) \
\
    ( ( dt & BLIS_DOMAIN_BIT ) == BLIS_BITVAL_COMPLEX )

#define bl2_is_single_prec( dt ) \
\
    ( ( dt & BLIS_PRECISION_BIT ) == BLIS_BITVAL_SINGLE_PREC )

#define bl2_is_double_prec( dt ) \
\
    ( ( dt & BLIS_PRECISION_BIT ) == BLIS_BITVAL_DOUBLE_PREC )

#define bl2_datatype_proj_to_real( dt ) \
\
	( dt & ~BLIS_BITVAL_COMPLEX )

#define bl2_datatype_proj_to_complex( dt ) \
\
	( dt &  BLIS_BITVAL_COMPLEX )


// side

#define bl2_is_left( side ) \
\
    ( side == BLIS_LEFT )

#define bl2_is_right( side ) \
\
    ( side == BLIS_RIGHT )

#define bl2_side_toggled( side ) \
\
	( bl2_is_left( side ) ? BLIS_RIGHT : BLIS_LEFT )

#define bl2_toggle_side( side ) \
{ \
	side = bl2_side_toggled( side ); \
}


// uplo

#define bl2_is_lower( uplo ) \
\
    ( uplo == BLIS_LOWER )

#define bl2_is_upper( uplo ) \
\
    ( uplo == BLIS_UPPER )

#define bl2_is_upper_or_lower( uplo ) \
\
    ( bl2_is_upper( uplo ) || bl2_is_lower( uplo ) )

#define bl2_is_dense( uplo ) \
\
    ( uplo == BLIS_DENSE )

#define bl2_is_zeros( uplo ) \
\
    ( uplo == BLIS_ZEROS )

#define bl2_uplo_toggled( uplo ) \
\
	( bl2_is_upper_or_lower( uplo ) ? \
	  ( ( uplo ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT ) : uplo \
	)

#define bl2_toggle_uplo( uplo ) \
{ \
	uplo = bl2_uplo_toggled( uplo ); \
}

#define bl2_set_uplo_with_trans( trans, uplo, uplo_trans ) \
{ \
	if ( bl2_does_notrans( trans ) ) { uplo_trans = uplo;                     } \
	else                             { uplo_trans = bl2_uplo_toggled( uplo ); } \
}


// structure

#define bl2_is_general( struc ) \
\
	( struc == BLIS_GENERAL )

#define bl2_is_hermitian( struc ) \
\
	( struc == BLIS_HERMITIAN )

#define bl2_is_symmetric( struc ) \
\
	( struc == BLIS_SYMMETRIC )

#define bl2_is_triangular( struc ) \
\
	( struc == BLIS_TRIANGULAR )


// conj

#define bl2_is_noconj( conj ) \
\
    ( conj == BLIS_NO_CONJUGATE )

#define bl2_is_conj( conj ) \
\
    ( conj == BLIS_CONJUGATE )

#define bl2_conj_toggled( conj ) \
\
	( conj ^ BLIS_CONJ_BIT )

#define bl2_apply_conj( conjapp, conj )\
\
	( conj ^ (conjapp) )

#define bl2_toggle_conj( conj ) \
{ \
	conj = bl2_conj_toggled( conj ); \
}


// trans

#define bl2_is_notrans( trans ) \
\
    ( trans == BLIS_NO_TRANSPOSE )

#define bl2_is_trans( trans ) \
\
    ( trans == BLIS_TRANSPOSE )

#define bl2_is_conjnotrans( trans ) \
\
    ( trans == BLIS_CONJ_NO_TRANSPOSE )

#define bl2_is_conjtrans( trans ) \
\
    ( trans == BLIS_CONJ_TRANSPOSE )

#define bl2_does_notrans( trans ) \
\
	( (~(trans) & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS )

#define bl2_does_trans( trans ) \
\
	( (  trans  & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS )

#define bl2_does_noconj( trans ) \
\
	( (~(trans) & BLIS_CONJ_BIT  ) == BLIS_BITVAL_CONJ )

#define bl2_does_conj( trans ) \
\
	( (  trans  & BLIS_CONJ_BIT  ) == BLIS_BITVAL_CONJ )

#define bl2_extract_trans( trans ) \
\
	(    trans  & BLIS_TRANS_BIT  )

#define bl2_extract_conj( trans ) \
\
	(    trans  & BLIS_CONJ_BIT  )

#define bl2_trans_toggled( trans ) \
\
	( trans ^ BLIS_TRANS_BIT )

#define bl2_toggle_trans( trans ) \
{ \
	trans = bl2_trans_toggled( trans ); \
}


// diag

#define bl2_is_nonunit_diag( diag ) \
\
    ( diag == BLIS_NONUNIT_DIAG )

#define bl2_is_unit_diag( diag ) \
\
    ( diag == BLIS_UNIT_DIAG )


// dimension-related

#define bl2_zero_dim1( n ) \
\
	( (n) == 0 )

#define bl2_zero_dim2( m, n ) \
\
	( (m) == 0 || (n) == 0 )

#define bl2_zero_dim3( m, n, k ) \
\
	( (m) == 0 || (n) == 0 || (k) == 0 )

#define bl2_nonzero_dim( n ) \
\
	( (n) > 0 )

#define bl2_vector_dim( m, n ) \
\
	( (m) == 1 ? (n) : (m) )

#define bl2_is_vector( m, n ) \
\
	( (m) == 1 || (n) == 1 )

#define bl2_is_row_vector( m, n ) \
\
	( (m) == 1 )

#define bl2_is_col_vector( m, n ) \
\
	( (n) == 1 )

#define bl2_set_dim_with_side( side, m, n, dim ) \
{ \
	if ( bl2_is_left( side ) ) { dim = m; } \
	else                       { dim = n; } \
}

#define bl2_set_dims_with_trans( trans, m, n, mtrans, ntrans ) \
{ \
	if ( bl2_does_notrans( trans ) ) { mtrans = m; ntrans = n; } \
	else                             { mtrans = n; ntrans = m; } \
}

#define bl2_set_dims_incs_with_trans( trans, m, n, rs, cs, mt, nt, rst, cst ) \
{ \
	if ( bl2_does_notrans( trans ) ) { mt = m; nt = n; rst = rs; cst = cs; } \
	else                             { mt = n; nt = m; rst = cs; cst = rs; } \
}

// stride-related

#define bl2_vector_inc( trans, m, n, rs, cs ) \
\
	( bl2_does_notrans( trans ) ? ( m == 1 ? (cs) : (rs) ) \
                                : ( m == 1 ? (rs) : (cs) ) )

#define bl2_is_row_stored( rs, cs ) \
\
	( cs == 1 )

#define bl2_is_col_stored( rs, cs ) \
\
	( rs == 1 )

#define bl2_is_gen_stored( rs, cs ) \
\
	( rs != 1 && cs != 1 )

#define bl2_is_row_tilted( rs, cs ) \
\
	( cs < rs )

#define bl2_is_col_tilted( rs, cs ) \
\
	( rs < cs )


// diag offset-related

#define bl2_negate_diag_offset( diagoff ) \
{ \
	diagoff = -diagoff; \
}

#define bl2_shift_diag_offset_to_grow_uplo( uplo, diagoff ) \
{ \
	if      ( bl2_is_upper( uplo ) ) diagoff -= 1; \
	else if ( bl2_is_lower( uplo ) ) diagoff += 1; \
}

#define bl2_shift_diag_offset_to_shrink_uplo( uplo, diagoff ) \
{ \
	if      ( bl2_is_upper( uplo ) ) diagoff += 1; \
	else if ( bl2_is_lower( uplo ) ) diagoff -= 1; \
}

#define bl2_diag_offset_with_trans( trans, diagoff ) \
\
	( bl2_does_trans( trans ) ? -diagoff : diagoff )

#define bl2_is_strictly_above_diag( diagoff, trans, m, n ) \
\
	( bl2_does_trans( trans ) ? ( ( doff_t )n <= -diagoff ) \
	                          : ( ( doff_t )m <= -diagoff ) )

#define bl2_is_strictly_below_diag( diagoff, trans, m, n ) \
\
	( bl2_does_trans( trans ) ? ( ( doff_t )m <=  diagoff ) \
	                          : ( ( doff_t )n <=  diagoff ) )

#define bl2_is_outside_diag( diagoff, trans, m, n ) \
\
	( bl2_is_strictly_above_diag( diagoff, trans, m, n ) || \
	  bl2_is_strictly_below_diag( diagoff, trans, m, n ) )

#define bl2_is_stored_subpart( diagoff, trans, uplo, m, n ) \
\
	( ( bl2_is_upper( uplo ) && bl2_is_strictly_above_diag( diagoff, trans, m, n ) ) || \
	  ( bl2_is_lower( uplo ) && bl2_is_strictly_below_diag( diagoff, trans, m, n ) ) )

#define bl2_is_unstored_subpart( diagoff, trans, uplo, m, n ) \
\
	( ( bl2_is_upper( uplo ) && bl2_is_strictly_below_diag( diagoff, trans, m, n ) ) || \
	  ( bl2_is_lower( uplo ) && bl2_is_strictly_above_diag( diagoff, trans, m, n ) ) )

#define bl2_is_strictly_above_diag_n( diagoff, m, n ) \
\
	( ( doff_t )m <= -diagoff ) \

#define bl2_is_strictly_below_diag_n( diagoff, m, n ) \
\
	( ( doff_t )n <=  diagoff ) \

#define bl2_intersects_diag_n( diagoff, m, n ) \
\
	( !bl2_is_strictly_above_diag_n( diagoff, m, n ) && \
	  !bl2_is_strictly_below_diag_n( diagoff, m, n ) )

#define bl2_is_stored_subpart_n( diagoff, uplo, m, n ) \
\
	( ( bl2_is_upper( uplo ) && bl2_is_strictly_above_diag_n( diagoff, m, n ) ) || \
	  ( bl2_is_lower( uplo ) && bl2_is_strictly_below_diag_n( diagoff, m, n ) ) )

#define bl2_is_unstored_subpart_n( diagoff, uplo, m, n ) \
\
	( ( bl2_is_upper( uplo ) && bl2_is_strictly_below_diag_n( diagoff, m, n ) ) || \
	  ( bl2_is_lower( uplo ) && bl2_is_strictly_above_diag_n( diagoff, m, n ) ) )


// index-related

#define bl2_is_edge_f( i1, iter, left ) \
\
	( i1 == iter - 1 && left != 0 )

#define bl2_is_not_edge_f( i1, iter, left ) \
\
	( i1 != iter - 1 || left == 0 )

#define bl2_is_edge_b( i1, iter, left ) \
\
	( i1 == 0 && left != 0 )

#define bl2_is_not_edge_b( i1, iter, left ) \
\
	( i1 != 0 || left == 0 )


// packbuf_t-related

#define bl2_packbuf_index( buf_type ) \
\
	( ( (buf_type) & BLIS_PACK_BUFFER_BITS ) >> BLIS_PACK_BUFFER_SHIFT )


// return value for char


// return datatype for char

#define bl2_stype ( BLIS_FLOAT    )
#define bl2_dtype ( BLIS_DOUBLE   )
#define bl2_ctype ( BLIS_SCOMPLEX )
#define bl2_ztype ( BLIS_DCOMPLEX )


// return datatype "union" for char pair

#define bl2_sstypeunion() ( BLIS_FLOAT    )
#define bl2_sdtypeunion() ( BLIS_DOUBLE   )
#define bl2_sctypeunion() ( BLIS_SCOMPLEX )
#define bl2_sztypeunion() ( BLIS_DCOMPLEX )

#define bl2_dstypeunion() ( BLIS_DOUBLE   )
#define bl2_ddtypeunion() ( BLIS_DOUBLE   )
#define bl2_dctypeunion() ( BLIS_DCOMPLEX )
#define bl2_dztypeunion() ( BLIS_DCOMPLEX )

#define bl2_cstypeunion() ( BLIS_SCOMPLEX )
#define bl2_cdtypeunion() ( BLIS_DCOMPLEX )
#define bl2_cctypeunion() ( BLIS_SCOMPLEX )
#define bl2_cztypeunion() ( BLIS_DCOMPLEX )

#define bl2_zstypeunion() ( BLIS_DCOMPLEX )
#define bl2_zdtypeunion() ( BLIS_DCOMPLEX )
#define bl2_zctypeunion() ( BLIS_DCOMPLEX )
#define bl2_zztypeunion() ( BLIS_DCOMPLEX )


// return default format specifier for char

#define bl2_sformatspec() "%9.2e"
#define bl2_dformatspec() "%9.2e"
#define bl2_cformatspec() "%9.2e + %9.2e "
#define bl2_zformatspec() "%9.2e + %9.2e "


// project dt to real if imaginary component is zero

#define bl2_proj_dt_to_real_if_imag_eq0( buf, dt ) \
{ \
	if ( bl2_is_scomplex( dt ) ) \
	{ \
		if ( bl2_cimageq0( buf ) ) dt = BLIS_FLOAT; \
	} \
	else if ( bl2_is_dcomplex( dt ) ) \
	{ \
		if ( bl2_zimageq0( buf ) ) dt = BLIS_DOUBLE; \
	} \
}

// set scalar datatype and buffer

#define bl2_set_scalar_dt_buffer( obj_scalar, dt_aux, dt_scalar, buf_scalar ) \
{ \
	if ( bl2_obj_is_const( *(obj_scalar) ) ) \
	{ \
		dt_scalar  = dt_aux; \
		buf_scalar = bl2_obj_scalar_buffer( dt_scalar, *(obj_scalar) ); \
	} \
	else \
	{ \
		dt_scalar  = bl2_obj_datatype( *(obj_scalar) ); \
		buf_scalar = bl2_obj_buffer_at_off( *(obj_scalar) ); \
	} \
\
	/* Projecting the scalar datatype to the real domain when the imaginary
	   part of is zero doesn't work when only basic datatype support is
	   enabled, because it can result in trying to use mixed datatype
	   functionality.
	bl2_proj_dt_to_real_if_imag_eq0( buf_scalar, dt_scalar ); */ \
}

// set constant datatype and buffer

#define bl2_set_const_dt_buffer( obj_scalar, dt_aux, dt_scalar, buf_scalar ) \
{ \
	{ \
		dt_scalar  = dt_aux; \
		buf_scalar = bl2_obj_scalar_buffer( dt_scalar, *(obj_scalar) ); \
	} \
\
	/* Projecting the scalar datatype to the real domain when the imaginary
	   part of is zero doesn't work when only basic datatype support is
	   enabled, because it can result in trying to use mixed datatype
	   functionality.
	bl2_proj_dt_to_real_if_imag_eq0( buf_scalar, dt_scalar ); */ \
}


// Set dimensions, increments, effective uplo/diagoff, etc for ONE matrix
// argument.

#define bl2_set_dims_incs_uplo_1m( diagoffa, diaga, \
                                   uploa,    m,          n,      rs_a, cs_a, \
                                   uplo_eff, n_elem_max, n_iter, inca, lda, \
                                   ij0, n_shift ) \
{ \
	/* If matrix A is entirely "unstored", that is, if either:
	   - A is lower-stored and entirely above the diagonal, or
	   - A is upper-stored and entirely below the diagonal
	   then we mark the storage as implicitly zero. */ \
	if ( bl2_is_unstored_subpart( diagoffa, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
	{ \
		uplo_eff = BLIS_ZEROS; \
	} \
	else \
	{ \
		doff_t diagoffa_use = diagoffa; \
		doff_t diagoff_eff; \
		dim_t  n_iter_max; \
\
		if ( bl2_is_unit_diag( diaga ) ) \
			bl2_shift_diag_offset_to_shrink_uplo( uploa, diagoffa_use ); \
\
		/* If matrix A is entirely "stored", that is, if either:
		   - A is upper-stored and entirely above the diagonal, or
		   - A is lower-stored and entirely below the diagonal
		   then we mark the storage as dense. */ \
		if ( bl2_is_stored_subpart( diagoffa_use, BLIS_NO_TRANSPOSE, uploa, m, n ) ) \
			uploa = BLIS_DENSE; \
\
		n_iter_max  = n; \
		n_elem_max  = m; \
		inca        = rs_a; \
		lda         = cs_a; \
		uplo_eff    = uploa; \
		diagoff_eff = diagoffa_use; \
\
		if ( bl2_is_row_tilted( inca, lda ) ) \
		{ \
			bl2_swap_dims( n_iter_max, n_elem_max ); \
			bl2_swap_incs( inca, lda ); \
			bl2_toggle_uplo( uplo_eff ); \
			bl2_negate_diag_offset( diagoff_eff ); \
		} \
\
		if ( bl2_is_dense( uplo_eff ) ) \
		{ \
			n_iter = n_iter_max; \
		} \
		else if ( bl2_is_upper( uplo_eff ) ) \
		{ \
			if ( diagoff_eff < 0 ) \
			{ \
				ij0        = 0; \
				n_shift    = -diagoff_eff; \
				n_elem_max = bl2_min( n_elem_max, n_shift + bl2_min( m, n ) ); \
				n_iter     = n_iter_max; \
			} \
			else \
			{ \
				ij0        = diagoff_eff; \
				n_shift    = 0; \
				n_iter     = n_iter_max - diagoff_eff; \
			} \
		} \
		else /* if ( bl2_is_lower( uplo_eff ) ) */ \
		{ \
			if ( diagoff_eff < 0 ) \
			{ \
				ij0        = -diagoff_eff; \
				n_shift    = 0; \
				n_elem_max = n_elem_max + diagoff_eff; \
				n_iter     = bl2_min( n_elem_max, bl2_min( m, n ) ); \
			} \
			else \
			{ \
				ij0        = 0; \
				n_shift    = diagoff_eff; \
				n_iter     = bl2_min( n_iter_max, n_shift + bl2_min( m, n ) ); \
			} \
		} \
	} \
}

// Set dimensions, increments, effective uplo/diagoff, etc for TWO matrix
// arguments.

#define bl2_set_dims_incs_uplo_2m( \
          diagoffa, diaga, transa, \
          uploa,    m,          n,      rs_a, cs_a, rs_b, cs_b, \
          uplo_eff, n_elem_max, n_iter, inca, lda,  incb, ldb, \
          ij0, n_shift \
        ) \
{ \
	/* If matrix A is entirely "unstored", that is, if either:
	   - A is lower-stored and entirely above the diagonal, or
	   - A is upper-stored and entirely below the diagonal
	   then we mark the storage as implicitly zero. */ \
	if ( bl2_is_unstored_subpart( diagoffa, transa, uploa, m, n ) ) \
	{ \
		uplo_eff = BLIS_ZEROS; \
	} \
	else \
	{ \
		doff_t diagoffa_use = diagoffa; \
		doff_t diagoff_eff; \
		dim_t  n_iter_max; \
\
		if ( bl2_is_unit_diag( diaga ) ) \
			bl2_shift_diag_offset_to_shrink_uplo( uploa, diagoffa_use ); \
\
		/* If matrix A is entirely "stored", that is, if either:
		   - A is upper-stored and entirely above the diagonal, or
		   - A is lower-stored and entirely below the diagonal
		   then we mark the storage as dense. */ \
		if ( bl2_is_stored_subpart( diagoffa_use, transa, uploa, m, n ) ) \
			uploa = BLIS_DENSE; \
\
		n_iter_max  = n; \
		n_elem_max  = m; \
		inca        = rs_a; \
		lda         = cs_a; \
		incb        = rs_b; \
		ldb         = cs_b; \
		uplo_eff    = uploa; \
		diagoff_eff = diagoffa_use; \
\
		if ( bl2_does_trans( transa ) ) \
		{ \
			bl2_swap_incs( inca, lda ); \
			bl2_toggle_uplo( uplo_eff ); \
			bl2_negate_diag_offset( diagoff_eff ); \
		} \
\
		if ( bl2_is_row_tilted( incb, ldb ) && \
		     bl2_is_row_tilted( inca, lda ) ) \
		{ \
			bl2_swap_dims( n_iter_max, n_elem_max ); \
			bl2_swap_incs( inca, lda ); \
			bl2_swap_incs( incb, ldb ); \
			bl2_toggle_uplo( uplo_eff ); \
			bl2_negate_diag_offset( diagoff_eff ); \
		} \
\
		if ( bl2_is_dense( uplo_eff ) ) \
		{ \
			n_iter = n_iter_max; \
		} \
		else if ( bl2_is_upper( uplo_eff ) ) \
		{ \
			if ( diagoff_eff < 0 ) \
			{ \
/*printf( "uplo_eff = upper, diagoff_eff < 0\n" );*/ \
				ij0        = 0; \
				n_shift    = -diagoff_eff; \
				n_elem_max = bl2_min( n_elem_max, n_shift + bl2_min( m, n ) ); \
				n_iter     = n_iter_max; \
			} \
			else \
			{ \
/*printf( "uplo_eff = upper, diagoff_eff >= 0\n" );*/ \
				ij0        = diagoff_eff; \
				n_shift    = 0; \
				n_iter     = n_iter_max - diagoff_eff; \
			} \
		} \
		else /* if ( bl2_is_lower( uplo_eff ) ) */ \
		{ \
			if ( diagoff_eff < 0 ) \
			{ \
/*printf( "uplo_eff = lower, diagoff_eff < 0\n" );*/ \
				ij0        = -diagoff_eff; \
				n_shift    = 0; \
				n_elem_max = n_elem_max + diagoff_eff; \
				n_iter     = bl2_min( n_elem_max, bl2_min( m, n ) ); \
			} \
			else \
			{ \
/*printf( "uplo_eff = lower, diagoff_eff >= 0\n" );*/ \
				ij0        = 0; \
				n_shift    = diagoff_eff; \
				n_iter     = bl2_min( n_iter_max, n_shift + bl2_min( m, n ) ); \
			} \
		} \
	} \
}

// Set dimensions, increments, etc for ONE matrix argument when operating
// on the diagonal.

#define bl2_set_dims_incs_1d( diagoffx, \
                              m, n, rs_x, cs_x, \
                              offx, n_elem, incx ) \
{ \
	if ( diagoffx < 0 ) \
	{ \
		n_elem = bl2_min( m - ( dim_t )(-diagoffx), n ); \
		offx   = ( dim_t )(-diagoffx) * rs_x; \
	} \
	else \
	{ \
		n_elem = bl2_min( n - ( dim_t )( diagoffx), m ); \
		offx   = ( dim_t )( diagoffx) * cs_x; \
	} \
\
	incx = rs_x + cs_x; \
}

// Set dimensions, increments, etc for TWO matrix arguments when operating
// on diagonals.

#define bl2_set_dims_incs_2d( diagoffx, transx, \
                              m, n, rs_x, cs_x, rs_y, cs_y, \
                              offx, offy, n_elem, incx, incy ) \
{ \
	doff_t diagoffy = bl2_diag_offset_with_trans( transx, diagoffx ); \
\
	if ( diagoffx < 0 ) offx = -diagoffx * rs_x; \
	else                offx =  diagoffx * cs_x; \
\
	if ( diagoffy < 0 ) \
	{ \
		n_elem = bl2_min( m - ( dim_t )(-diagoffy), n ); \
		offy   = -diagoffy * rs_y; \
	} \
	else \
	{ \
		n_elem = bl2_min( n - ( dim_t )( diagoffy), m ); \
		offy   = diagoffy * cs_y; \
	} \
\
	incx = rs_x + cs_x; \
	incy = rs_y + cs_y; \
}

#endif
