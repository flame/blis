/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

bool bli_packm_init
     (
       obj_t*  c,
       obj_t*  p,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	bli_init_once();

	// The purpose of packm_init() is to initialize an object P so that
	// a source object A can be packed into P via one of the packm
	// implementations. This initialization precedes the acquisition of a
	// suitable block of memory from the memory allocator (if such a block
	// of memory has not already been allocated previously).

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packm_init_check( c, p, cntx );

	// We begin by copying the fields of A.
	bli_obj_alias_to( c, p );

	// If the object is marked as being filled with zeros, then we can skip
	// the packm operation entirely and alias.
	if ( bli_obj_is_zeros( c ) )
		return false;

	// Extract various fields from the control tree.
	bszid_t bmult_id_m   = bli_cntl_packm_params_bmid_m( cntl );
	bszid_t bmult_id_n   = bli_cntl_packm_params_bmid_n( cntl );
	pack_t  schema       = bli_cntl_packm_params_pack_schema( cntl );
	num_t   dt_tar       = bli_obj_target_dt( c );
	num_t   dt_scalar    = bli_obj_scalar_dt( c );
	dim_t   bmult_m_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_m, cntx );
	dim_t   bmult_m_pack = bli_cntx_get_blksz_max_dt( dt_tar, bmult_id_m, cntx );
	dim_t   bmult_n_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_n, cntx );

	// Typecast the internal scalar value to the target datatype.
	// Note that if the typecasting is needed, this must happen BEFORE we
	// change the datatype of P to reflect the target_dt.
	if ( dt_scalar != dt_tar )
	{
		bli_obj_scalar_cast_to( dt_tar, p );
	}

	// Update the storage datatype of P to be the target datatype of A.
	bli_obj_set_dt( dt_tar, p );
    bli_obj_set_elem_size( bli_dt_size( dt_tar ), p );

	// Store the pack schema to the object.
	bli_obj_set_pack_schema( schema, p );

	// Clear the conjugation field from the object since matrix packing
	// in BLIS is deemed to take care of all conjugation necessary.
	bli_obj_set_conj( BLIS_NO_CONJUGATE, p );

	// If we are packing micropanels, mark P as dense.
	bli_obj_set_uplo( BLIS_DENSE, p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, p );

	// Compute the dimensions padded by the dimension multiples. These
	// dimensions will be the dimensions of the packed matrices, including
	// zero-padding, and will be used by the macro- and micro-kernels.
	// We compute them by starting with the effective dimensions of A (now
	// in P) and aligning them to the dimension multiples (typically equal
	// to register blocksizes). This does waste a little bit of space for
	// level-2 operations, but that's okay with us.
	dim_t m_p     = bli_obj_length( p );
	dim_t n_p     = bli_obj_width( p );
	dim_t m_p_pad = bli_align_dim_to_mult( m_p, bmult_m_def );
	dim_t n_p_pad = bli_align_dim_to_mult( n_p, bmult_n_def );

	// Save the padded dimensions into the packed object. It is important
	// to save these dimensions since they represent the actual dimensions
	// of the zero-padded matrix.
	bli_obj_set_padded_dims( m_p_pad, n_p_pad, p );

	// Now we prepare to compute strides, align them, and compute the
	// total number of bytes needed for the packed buffer. The caller
	// will then use that value to acquire an appropriate block of memory
	// from the memory allocator.

	// Extract the element size for the packed object.
	siz_t elem_size_p = bli_obj_elem_size( p );

	// The panel dimension (for each datatype) should be equal to the
	// default (logical) blocksize multiple in the m dimension.
	dim_t m_panel = bmult_m_def;

	// The "column stride" of a row-micropanel packed object is interpreted
	// as the column stride WITHIN a micropanel. Thus, this is equal to the
	// packing (storage) blocksize multiple, which may be equal to the
	// default (logical) blocksize multiple).
	inc_t cs_p = bmult_m_pack;

	// The "row stride" of a row-micropanel packed object is interpreted
	// as the row stride WITHIN a micropanel. Thus, it is unit.
	inc_t rs_p = 1;

	// The "panel stride" of a micropanel packed object is interpreted as
	// the distance between the (0,0) element of panel k and the (0,0)
	// element of panel k+1. We use the padded width computed above to
	// allow for zero-padding (if necessary/desired) along the far end
	// of each micropanel (ie: the right edge of the matrix). Zero-padding
	// can also occur along the long edge of the last micropanel if the m
	// dimension of the matrix is not a whole multiple of MR.
	inc_t ps_p = cs_p * n_p_pad;

	// As a general rule, we don't want micropanel strides to be odd. This
	// is primarily motivated by our desire to support interleaved 3m
	// micropanels, in which case we have to scale the panel stride
	// by 3/2. That division by 2 means the numerator (prior to being
	// scaled by 3) must be even.
	if ( bli_is_odd( ps_p ) ) ps_p += 1;

	// Set the imaginary stride (in units of fundamental elements) for
	// 3m and 4m (separated or interleaved). We use ps_p_orig since
	// that variable tracks the number of real part elements contained
	// within each micropanel of the source matrix. Therefore, this
	// is the number of real elements that must be traversed before
	// reaching the imaginary part (3mi/4mi) of the packed micropanel,
	// or the real part of the next micropanel (3ms).
	inc_t is_p;
	if      ( bli_is_3mi_packed( schema ) ) is_p = ps_p;
	else if ( bli_is_4mi_packed( schema ) ) is_p = ps_p;
	else if ( bli_is_3ms_packed( schema ) ) is_p = ps_p * ( m_p_pad / m_panel );
	else                                    is_p = 1;

	// Here, we adjust the panel stride, if necessary. Remember: ps_p is
	// always interpreted as being in units of the datatype of the object
	// which is not necessarily how the micropanels will be stored. For
	// interleaved 3m, we will increase ps_p by 50%, and for ro/io/rpi,
	// we halve ps_p. Why? Because the macro-kernel indexes in units of
	// the complex datatype. So these changes "trick" it into indexing
	// the correct amount.
	if ( bli_is_3mi_packed( schema ) )
	{
		ps_p = ( ps_p * 3 ) / 2;
	}
	else if ( bli_is_3ms_packed( schema ) ||
	          bli_is_ro_packed( schema )  ||
	          bli_is_io_packed( schema )  ||
	          bli_is_rpi_packed( schema ) )
	{
		// Despite the fact that the packed micropanels will contain
		// real elements, the panel stride that we store in the obj_t
		// (which is passed into the macro-kernel) needs to be in units
		// of complex elements, since the macro-kernel will index through
		// micropanels via complex pointer arithmetic for trmm/trsm.
		// Since the indexing "increment" will be twice as large as each
		// actual stored element, we divide the panel_stride by 2.
		ps_p = ps_p / 2;
	}

	// Store the strides and panel dimension in P.
	bli_obj_set_strides( rs_p, cs_p, p );
	bli_obj_set_imag_stride( is_p, p );
	bli_obj_set_panel_dim( m_panel, p );
	bli_obj_set_panel_stride( ps_p, p );
	bli_obj_set_panel_length( m_panel, p );
	bli_obj_set_panel_width( n_p, p );

	// Compute the size of the packed buffer.
	siz_t size_p = ps_p * ( m_p_pad / m_panel ) * elem_size_p;

	if ( size_p == 0 )
		return false;

	// Update the buffer address in p to point to the buffer associated
	// with the mem_t entry acquired from the memory broker (now cached in
	// the control tree node).
	void* buffer = bli_packm_alloc( size_p, rntm, cntl, thread );
	bli_obj_set_buffer( buffer, p );

	return true;
}

