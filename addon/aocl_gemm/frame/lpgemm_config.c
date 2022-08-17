/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_config.h"

lpgemm_cntx_t global_cntx_t_list[4]; //Only one op type supported now.

BLIS_INLINE void lpgemm_set_block_sizes_global_cntx
     (
       AOCL_OPERATION_TYPE op_type,
       dim_t MC,
       dim_t NC,
       dim_t KC,
       dim_t NR,
       dim_t MR
     )
{
	global_cntx_t_list[op_type].blksz.MC = MC;
	global_cntx_t_list[op_type].blksz.NC = NC;
	global_cntx_t_list[op_type].blksz.KC = KC;
	global_cntx_t_list[op_type].blksz.NR = NR;
	global_cntx_t_list[op_type].blksz.MR = MR;
}

// Sets default block sizes for lpgemm. Currently only u8s8s32 supported.
// Thread safety is not considered now since the block sizes are not expected
// to be configurable from application.
void aocl_lpgemm_init_global_cntx()
{
    lpgemm_set_block_sizes_global_cntx( U8S8S32OS32, 144, 1024, 2048, 64, 6 );
    lpgemm_set_block_sizes_global_cntx( U8S8S16OS16, 144, 1024, 1024, 32, 6 );
    lpgemm_set_block_sizes_global_cntx( BF16BF16F32OF32, 144, 1024, 2048, 64, 6 );
}

dim_t lpgemm_get_block_size_MC_global_cntx( AOCL_OPERATION_TYPE op_type )
{
	return global_cntx_t_list[op_type].blksz.MC;
}

dim_t lpgemm_get_block_size_NC_global_cntx( AOCL_OPERATION_TYPE op_type )
{
	return global_cntx_t_list[op_type].blksz.NC;
}

dim_t lpgemm_get_block_size_KC_global_cntx( AOCL_OPERATION_TYPE op_type )
{
	return global_cntx_t_list[op_type].blksz.KC;
}

dim_t lpgemm_get_block_size_NR_global_cntx( AOCL_OPERATION_TYPE op_type )
{
	return global_cntx_t_list[op_type].blksz.NR;
}

dim_t lpgemm_get_block_size_MR_global_cntx( AOCL_OPERATION_TYPE op_type )
{
	return global_cntx_t_list[op_type].blksz.MR;
}
