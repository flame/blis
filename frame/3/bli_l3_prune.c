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


void bli_l3_prune_unref_mparts_m
     (
             obj_t*  a,
       const obj_t*  b,
             obj_t*  c,
       const cntl_t* cntl
     )
{
	/* Query the operation family. */
	opid_t family = bli_cntl_family( cntl );

	if      ( family == BLIS_GEMM )
	{
		/* No pruning is necessary for gemm. */
		return;
	}
	else if ( family == BLIS_GEMMT )
	{
		/* Prune any unreferenced part from the subpartition of C (that would
		   be encountered from partitioning in the m dimension) and adjust the
		   subpartition of A accordingly. */
		bli_prune_unref_mparts( c, BLIS_M, a, BLIS_M );
	}
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRSM )
	{
		/* Prune any unreferenced part from the subpartition of A (that would
		   be encountered from partitioning in the m dimension) and adjust the
		   subpartition of C accordingly. */
		bli_prune_unref_mparts( a, BLIS_M, c, BLIS_M );
	}
}

void bli_l3_prune_unref_mparts_n
     (
       const obj_t*  a,
             obj_t*  b,
             obj_t*  c,
       const cntl_t* cntl
     )
{
	/* Query the operation family. */
	opid_t family = bli_cntl_family( cntl );

	if      ( family == BLIS_GEMM )
	{
		/* No pruning is necessary for gemm. */
		return;
	}
	else if ( family == BLIS_GEMMT )
	{
		/* Prune any unreferenced part from the subpartition of C (that would
		   be encountered from partitioning in the m dimension) and adjust the
		   subpartition of B accordingly. */
		bli_prune_unref_mparts( c, BLIS_N, b, BLIS_N );
	}
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRSM )
	{
		/* Prune any unreferenced part from the subpartition of B (that would
		   be encountered from partitioning in the m dimension) and adjust the
		   subpartition of C accordingly. */
		bli_prune_unref_mparts( b, BLIS_N, c, BLIS_N );
	}
}

void bli_l3_prune_unref_mparts_k
     (
             obj_t*  a,
             obj_t*  b,
       const obj_t*  c,
       const cntl_t* cntl
     )
{
	/* Query the operation family. */
	opid_t family = bli_cntl_family( cntl );

	if      ( family == BLIS_GEMM )
	{
		/* No pruning is necessary for gemm. */
		return;
	}
	else if ( family == BLIS_GEMMT )
	{
		/* No pruning is necessary for gemmt. */
		return;
	}
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRSM )
	{
		/* Prune any unreferenced part from the subpartition of A (that would
		   be encountered from partitioning in the k dimension) and adjust the
		   subpartition of B accordingly. */
		bli_prune_unref_mparts( a, BLIS_N, b, BLIS_M );

		/* Prune any unreferenced part from the subpartition of B (that would
		   be encountered from partitioning in the k dimension) and adjust the
		   subpartition of A accordingly. */
		bli_prune_unref_mparts( b, BLIS_M, a, BLIS_N );
	}
}

