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

blksz_t* packm_mult_ldim;
blksz_t* packm_mult_nvec;

func_t*  packm_struc_cxk_kers;
func_t*  packm_struc_cxk_4m_kers;
func_t*  packm_struc_cxk_3m_kers;
func_t*  packm_struc_cxk_rih_kers;

packm_t* packm_cntl_row;
packm_t* packm_cntl_col;

packm_t* packm_cntl_rpn;
packm_t* packm_cntl_cpn;

packm_t* packm_cntl;

void bli_packm_cntl_init()
{
	// Create function pointer object for each datatype-specific packm
	// kernel.
	packm_struc_cxk_kers
	=
	bli_func_obj_create( bli_spackm_struc_cxk, FALSE,
	                     bli_dpackm_struc_cxk, FALSE,
	                     bli_cpackm_struc_cxk, FALSE,
	                     bli_zpackm_struc_cxk, FALSE );

	packm_struc_cxk_4m_kers
	=
	bli_func_obj_create( NULL,                    FALSE,
	                     NULL,                    FALSE,
	                     bli_cpackm_struc_cxk_4m, FALSE,
	                     bli_zpackm_struc_cxk_4m, FALSE );

	packm_struc_cxk_3m_kers
	=
	bli_func_obj_create( NULL,                    FALSE,
	                     NULL,                    FALSE,
	                     bli_cpackm_struc_cxk_3m, FALSE,
	                     bli_zpackm_struc_cxk_3m, FALSE );

	packm_struc_cxk_rih_kers
	=
	bli_func_obj_create( NULL,                     FALSE,
	                     NULL,                     FALSE,
	                     bli_cpackm_struc_cxk_rih, FALSE,
	                     bli_zpackm_struc_cxk_rih, FALSE );


	// Create blocksize objects for m and n register blocking. We will attach
	// these to the packm control node so they can be used to (a) allocate a
	// block whose m and n dimension are multiples of mr and nr, and (b) know
	// how much zero-padding is necessary for edge cases. 
	// NOTE: these alignments end up getting applied to matrices packed for
	// level-2 operations, even though they are not needed, and/or smaller
	// alignments may be sufficient. For simplicity, we choose to tweak the
	// dimensions of all pack matrix buffers the same amount.
	packm_mult_ldim
	=
	bli_blksz_obj_create( BLIS_DEFAULT_MR_S, 0,
	                      BLIS_DEFAULT_MR_D, 0,
	                      BLIS_DEFAULT_MR_C, 0,
	                      BLIS_DEFAULT_MR_Z, 0 );

	packm_mult_nvec
	=
	bli_blksz_obj_create( BLIS_DEFAULT_NR_S, 0,
	                      BLIS_DEFAULT_NR_D, 0,
	                      BLIS_DEFAULT_NR_C, 0,
	                      BLIS_DEFAULT_NR_Z, 0 );

	// Generally speaking, the BLIS_PACKED_ROWS and BLIS_PACKED_COLUMNS
	// are used by the level-2 operations. These schemas amount to simple
	// copies to row or column storage. These simple schemas may be used
	// by level-3 operations, but they should never be used for matrices
	// with structure (since they do not densify).
	// The BLIS_PACKED_ROW_PANELS and BLIS_PACKED_COL_PANELS schemas are
	// used only in level-3 operations. They pack to (typically) skinny
	// row and column panels, where the width of the panel is determined
	// by register blocksizes. It is assumed that matrices with structure
	// will be densified.

	// Create control trees to pack by rows.
	packm_cntl_row
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to rows:
	                           packm_mult_nvec,  // - nvec multiple is used for m dimension
	                           packm_mult_ldim,  // - ldim multiple is used for n dimension
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_ROWS,
	                           BLIS_BUFFER_FOR_GEN_USE );


	// Create control trees to pack by columns.
	packm_cntl_col
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to columns:
	                           packm_mult_ldim,  // - ldim multiple is used for m dimension
	                           packm_mult_nvec,  // - nvec multiple is used for n dimension
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );


	// Set defaults when we don't care whether the packing is by rows or
	// by columns.
	packm_cntl = packm_cntl_col;
}

void bli_packm_cntl_finalize()
{
	bli_func_obj_free( packm_struc_cxk_kers );
	bli_func_obj_free( packm_struc_cxk_4m_kers );
	bli_func_obj_free( packm_struc_cxk_3m_kers );
	bli_func_obj_free( packm_struc_cxk_rih_kers );

	bli_cntl_obj_free( packm_cntl_row );
	bli_cntl_obj_free( packm_cntl_col );

	bli_blksz_obj_free( packm_mult_ldim );
	bli_blksz_obj_free( packm_mult_nvec );
}

packm_t* bli_packm_cntl_obj_create( impl_t     impl_type,
                                    varnum_t   var_num,
                                    blksz_t*   mr,
                                    blksz_t*   nr,
                                    bool_t     does_invert_diag,
                                    bool_t     rev_iter_if_upper,
                                    bool_t     rev_iter_if_lower,
                                    pack_t     pack_schema,
                                    packbuf_t  pack_buf_type )
{
	packm_t* cntl;

	cntl = ( packm_t* ) bli_malloc( sizeof(packm_t) );

	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->mr                = mr;
	cntl->nr                = nr;
	cntl->does_invert_diag  = does_invert_diag;
	cntl->rev_iter_if_upper = rev_iter_if_upper;
	cntl->rev_iter_if_lower = rev_iter_if_lower;
	cntl->pack_schema       = pack_schema;
	cntl->pack_buf_type     = pack_buf_type;

	return cntl;
}

void bli_packm_cntl_obj_init( packm_t*   cntl,
                              impl_t     impl_type,
                              varnum_t   var_num,
                              blksz_t*   mr,
                              blksz_t*   nr,
                              bool_t     does_invert_diag,
                              bool_t     rev_iter_if_upper,
                              bool_t     rev_iter_if_lower,
                              pack_t     pack_schema,
                              packbuf_t  pack_buf_type )
{
	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->mr                = mr;
	cntl->nr                = nr;
	cntl->does_invert_diag  = does_invert_diag;
	cntl->rev_iter_if_upper = rev_iter_if_upper;
	cntl->rev_iter_if_lower = rev_iter_if_lower;
	cntl->pack_schema       = pack_schema;
	cntl->pack_buf_type     = pack_buf_type;
}

