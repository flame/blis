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

cntl_t* bli_trsm_cntl_create
     (
       side_t side
     )
{
	if ( bli_is_left( side ) ) return bli_trsm_l_cntl_create();
	else                       return bli_trsm_r_cntl_create();
}

cntl_t* bli_trsm_l_cntl_create
     (
       void
     )
{
	void* macro_kernel_p = bli_trsm_xx_ker_var2;

	const opid_t family = BLIS_TRSM;

	// Create two nodes for the macro-kernel.
	cntl_t* trsm_cntl_bu_ke = bli_trsm_cntl_create_node
	(
	  family,  // the operation family
	  BLIS_MR, // needed for bli_thrinfo_rgrow()
	  NULL,    // variant function pointer not used
	  NULL     // no sub-node; this is the leaf of the tree.
	);

	cntl_t* trsm_cntl_bp_bu = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_NR, // not used by macro-kernel, but needed for bli_thrinfo_rgrow()
	  macro_kernel_p,
	  trsm_cntl_bu_ke
	);

	// Create a node for packing matrix A.
	cntl_t* trsm_cntl_packa = bli_packm_cntl_create_node
	(
	  bli_trsm_packa,
	  bli_packm_blk_var1,
	  BLIS_MR,
	  BLIS_MR,
	  TRUE,    // do NOT invert diagonal
	  TRUE,    // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  BLIS_PACKED_ROW_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  trsm_cntl_bp_bu
	);

	// Create a node for partitioning the m dimension by MC.
	cntl_t* trsm_cntl_op_bp = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_MC,
	  bli_trsm_blk_var1,
	  trsm_cntl_packa
	);

	// Create a node for packing matrix B.
	cntl_t* trsm_cntl_packb = bli_packm_cntl_create_node
	(
	  bli_trsm_packb,
	  bli_packm_blk_var1,
	  BLIS_MR,
	  BLIS_NR,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  BLIS_PACKED_COL_PANELS,
	  BLIS_BUFFER_FOR_B_PANEL,
	  trsm_cntl_op_bp
	);

	// Create a node for partitioning the k dimension by KC.
	cntl_t* trsm_cntl_mm_op = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_KC,
	  bli_trsm_blk_var3,
	  trsm_cntl_packb
	);

	// Create a node for partitioning the n dimension by NC.
	cntl_t* trsm_cntl_vl_mm = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_NC,
	  bli_trsm_blk_var2,
	  trsm_cntl_mm_op
	);

	return trsm_cntl_vl_mm;
}

cntl_t* bli_trsm_r_cntl_create
     (
       void
     )
{
	void* macro_kernel_p = bli_trsm_xx_ker_var2;

	const opid_t family = BLIS_TRSM;

	// Create two nodes for the macro-kernel.
	cntl_t* trsm_cntl_bu_ke = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_MR, // needed for bli_thrinfo_rgrow()
	  NULL,    // variant function pointer not used
	  NULL     // no sub-node; this is the leaf of the tree.
	);

	cntl_t* trsm_cntl_bp_bu = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_NR, // not used by macro-kernel, but needed for bli_thrinfo_rgrow()
	  macro_kernel_p,
	  trsm_cntl_bu_ke
	);

	// Create a node for packing matrix A.
	cntl_t* trsm_cntl_packa = bli_packm_cntl_create_node
	(
	  bli_trsm_packa,
	  bli_packm_blk_var1,
	  BLIS_NR,
	  BLIS_MR,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  BLIS_PACKED_ROW_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  trsm_cntl_bp_bu
	);

	// Create a node for partitioning the m dimension by MC.
	cntl_t* trsm_cntl_op_bp = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_MC,
	  bli_trsm_blk_var1,
	  trsm_cntl_packa
	);

	// Create a node for packing matrix B.
	cntl_t* trsm_cntl_packb = bli_packm_cntl_create_node
	(
	  bli_trsm_packb,
	  bli_packm_blk_var1,
	  BLIS_MR,
	  BLIS_MR,
	  TRUE,    // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  TRUE,    // reverse iteration if lower?
	  BLIS_PACKED_COL_PANELS,
	  BLIS_BUFFER_FOR_B_PANEL,
	  trsm_cntl_op_bp
	);

	// Create a node for partitioning the k dimension by KC.
	cntl_t* trsm_cntl_mm_op = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_KC,
	  bli_trsm_blk_var3,
	  trsm_cntl_packb
	);

	// Create a node for partitioning the n dimension by NC.
	cntl_t* trsm_cntl_vl_mm = bli_trsm_cntl_create_node
	(
	  family,
	  BLIS_NC,
	  bli_trsm_blk_var2,
	  trsm_cntl_mm_op
	);

	return trsm_cntl_vl_mm;
}

void bli_trsm_cntl_free
     (
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	bli_cntl_free( cntl, thread );
}

// -----------------------------------------------------------------------------

cntl_t* bli_trsm_cntl_create_node
     (
       opid_t  family,
       bszid_t bszid,
       void*   var_func,
       cntl_t* sub_node
     )
{
	return bli_cntl_create_node( family, bszid, var_func, NULL, sub_node );
}

