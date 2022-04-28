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
#include "lpgemm_utils.h"

dim_t get_64byte_aligned_memory
     (
       void**  original_memory,
       void**  aligned_memory,
       int64_t allocate_size
     )
{
	// Get 64 byte aligned memory.
	int8_t* t1_original = ( int8_t* ) malloc( allocate_size + 64 );
	if ( t1_original == NULL )
	{
		//Error in malloc.
		*original_memory = NULL;
		*aligned_memory = NULL;
		return -1;
	}

	int8_t* ta_original = t1_original + 64;
	ta_original = ta_original - ( ( int64_t )( ta_original ) % 64 );

	*original_memory = t1_original;
	*aligned_memory = ta_original;
	return 0;
}

static lpgemm_obj_t* alloc_lpgemm_obj_t_u8s8s32
     (
       dim_t           length,
       dim_t           width,
       dim_t           stride,
       dim_t           elem_size,
       AOCL_STOR_TAG   stor_scheme,
       AOCL_MEMORY_TAG mtag
     )
{
	lpgemm_obj_t* obj = ( lpgemm_obj_t* ) malloc( sizeof( lpgemm_obj_t ) );

	if ( obj == NULL )
	{
		return NULL; //failure
	}

	// Allocate aligned buffers.
	get_64byte_aligned_memory( &obj->storage.origin_buffer,
			&obj->storage.aligned_buffer, 
			( elem_size * length * width ) );

	if ( obj->storage.origin_buffer == NULL )
	{
		// Buffer allocation failed.
		free( obj );
		return NULL;
	}

	obj->length = length;
	obj->width = width;
	obj->elem_size = elem_size;

	if ( stor_scheme == ROW_MAJOR )
	{
		obj->rs = stride;
		obj->cs = 4; // 4 elements read at a time.
	}
	else if ( stor_scheme == COLUMN_MAJOR )
	{
		obj->cs = stride;
		obj->rs = 1;
	}
	obj->mtag = mtag;

	return obj;
}

lpgemm_obj_t* alloc_unpack_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     )
{
	return alloc_lpgemm_obj_t_u8s8s32( length, width, stride, elem_size, stor_scheme, UNPACKED );
}

lpgemm_obj_t* alloc_pack_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     )
{
	return alloc_lpgemm_obj_t_u8s8s32( length, width, stride, elem_size, stor_scheme, PACK );
}

lpgemm_obj_t* alloc_reorder_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     )
{
	// Extra space since packing does width in multiples of 16.
	dim_t width_reorder = make_multiple_of_n( width, 16 );
	// Extra space since packing does length in multiples of 4.
	dim_t length_reorder = make_multiple_of_n( length, 4 );

	return alloc_lpgemm_obj_t_u8s8s32( length_reorder, width_reorder, stride, elem_size, stor_scheme, REORDERED );
}

void dealloc_lpgemm_obj_t_u8s8s32( lpgemm_obj_t* obj )
{
	free( obj->storage.origin_buffer );
	free( obj );
}
