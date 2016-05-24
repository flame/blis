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

packm_t* packm_cntl_row = NULL;
packm_t* packm_cntl_col = NULL;

packm_t* packm_cntl = NULL;

void bli_packm_cntl_init()
{
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
	                           BLIS_VF,          // used for m dimension
	                           BLIS_VF,          // used for n dimension
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
	                           BLIS_VF,          // used for m dimension
	                           BLIS_VF,          // used for n dimension
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
	bli_cntl_obj_free( packm_cntl_row );
	bli_cntl_obj_free( packm_cntl_col );
}

packm_t* bli_packm_cntl_obj_create( impl_t     impl_type,
                                    varnum_t   var_num,
                                    bszid_t    bmid_m,
                                    bszid_t    bmid_n,
                                    bool_t     does_invert_diag,
                                    bool_t     rev_iter_if_upper,
                                    bool_t     rev_iter_if_lower,
                                    pack_t     pack_schema,
                                    packbuf_t  pack_buf_type )
{
	packm_t* cntl;

	cntl = ( packm_t* ) bli_malloc_intl( sizeof(packm_t) );

	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->bmid_m            = bmid_m;
	cntl->bmid_n            = bmid_n;
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
                              bszid_t    bmid_m,
                              bszid_t    bmid_n,
                              bool_t     does_invert_diag,
                              bool_t     rev_iter_if_upper,
                              bool_t     rev_iter_if_lower,
                              pack_t     pack_schema,
                              packbuf_t  pack_buf_type )
{
	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->bmid_m            = bmid_m;
	cntl->bmid_n            = bmid_n;
	cntl->does_invert_diag  = does_invert_diag;
	cntl->rev_iter_if_upper = rev_iter_if_upper;
	cntl->rev_iter_if_lower = rev_iter_if_lower;
	cntl->pack_schema       = pack_schema;
	cntl->pack_buf_type     = pack_buf_type;
}

