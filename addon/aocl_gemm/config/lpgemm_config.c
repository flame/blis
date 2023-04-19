/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_func_map.h"
#include "lpgemm_blksz_map.h"
#include "lpgemm_kernels.h"
#include "lpgemm_packb_bf16.h"
#include "lpgemm_packb_s16.h"
#include "lpgemm_packa.h"
#include "lpgemm_packb.h"
#include "lpgemm_packa_s8.h"
#include "lpgemm_packb_s8.h"
#include "lpgemm_packb_s8s16.h"

static lpgemm_cntx_t global_cntx_t_list[AOCL_OPERATION_TYPE_LEN] \
					__attribute__((aligned(64))); //Only one op type supported now.
static lpgemm_util_cntx_t global_util_cntx_t_list[AOCL_UTIL_OPERATION_TYPE_LEN] \
					__attribute__((aligned(64))); //Only post-ops like utils.

static bli_pthread_once_t once_check_lpgemm_func_map_init = BLIS_PTHREAD_ONCE_INIT;

static void _lpgemm_util_cntx_init_func_map()
{
#define UMACRO(ID,FUNC_PTR) global_util_cntx_t_list[ID].kern_fun_ptr = FUNC_PTR;

	global_util_cntx_t_list[F32_GELU_TANH].kern_fun_ptr = NULL;
	global_util_cntx_t_list[F32_GELU_ERF].kern_fun_ptr = NULL;

	// Kernel dispatch object factory.
	if ( bli_cpuid_is_avx512bf16_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN4
		LPGEMM_UTIL_KERN_FUNC_MAP_AVX512_VNNI_BF16
#endif
	}
	else if ( bli_cpuid_is_avx512vnni_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN4
		LPGEMM_UTIL_KERN_FUNC_MAP_AVX512_VNNI
#endif
	}
	else if ( bli_cpuid_is_avx2fma3_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN3
		LPGEMM_UTIL_KERN_FUNC_MAP_AVX2
#endif
	}

#undef UMACRO
}

static void _lpgemm_cntx_init_func_map()
{
#define KMACRO(ID,FUNC_PTR) global_cntx_t_list[ID].kern_fun_ptr = FUNC_PTR;
#define PAMACRO(ID,FUNC_PTR) global_cntx_t_list[ID].packa_fun_ptr = FUNC_PTR;
#define PBMACRO(ID,FUNC_PTR) global_cntx_t_list[ID].packb_fun_ptr = FUNC_PTR;

	//TODO: Default initialize with reference kernels so that kernel pointer
	// will be valid even in case none of the zen optimized kernels are
	// available. This scenario could happen if the addon was built using
	// a different arch config (eg: skx).

	global_cntx_t_list[U8S8S16OS16].kern_fun_ptr = NULL;
	global_cntx_t_list[U8S8S32OS32].kern_fun_ptr = NULL;
	global_cntx_t_list[F32F32F32OF32].kern_fun_ptr = NULL;
	global_cntx_t_list[BF16BF16F32OF32].kern_fun_ptr = NULL;

	// Kernel dispatch object factory.
	if ( bli_cpuid_is_avx512bf16_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN4
		LPGEMM_KERN_FUNC_MAP_AVX512_VNNI_BF16
		LPGEMM_PACKA_FUNC_MAP_AVX512_VNNI_BF16
		LPGEMM_PACKB_FUNC_MAP_AVX512_VNNI_BF16
#endif
	}
	else if ( bli_cpuid_is_avx512vnni_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN4
		LPGEMM_KERN_FUNC_MAP_AVX512_VNNI
		LPGEMM_PACKA_FUNC_MAP_AVX512_VNNI
		LPGEMM_PACKB_FUNC_MAP_AVX512_VNNI
#endif
	}
	else if ( bli_cpuid_is_avx2fma3_supported() == TRUE )
	{
#ifdef BLIS_KERNELS_ZEN3
		LPGEMM_KERN_FUNC_MAP_AVX2
		LPGEMM_PACKA_FUNC_MAP_AVX2
		LPGEMM_PACKB_FUNC_MAP_AVX2
#endif
	}
	// If built with a config not supporting zen3/zen4/amdzen, error out
	// since reference kernels are not available.
	if ( global_cntx_t_list[F32F32F32OF32].kern_fun_ptr == NULL )
	{
		bli_print_msg( "AOCL_GEMM is not compiled using correct Zen config."
				" Compile using zen3/zen4/amdzen config.",
				__FILE__, __LINE__ );
		bli_abort();
	}

#undef PBMACRO
#undef PAMACRO
#undef KMACRO
}

BLIS_INLINE void lpgemm_set_block_sizes_global_cntx
     (
       AOCL_OPERATION_TYPE op_type,
       dim_t MC,
       dim_t NC,
       dim_t KC,
       dim_t MR,
       dim_t NR
     )
{
	global_cntx_t_list[op_type].blksz.MC = MC;
	global_cntx_t_list[op_type].blksz.NC = NC;
	global_cntx_t_list[op_type].blksz.KC = KC;
	global_cntx_t_list[op_type].blksz.MR = MR;
	global_cntx_t_list[op_type].blksz.NR = NR;
}

BLIS_INLINE void lpgemm_set_pack_strides_global_cntx
     (
       AOCL_OPERATION_TYPE op_type,
       dim_t packa_rs,
       dim_t packa_cs,
       dim_t packb_rs,
       dim_t packb_cs
     )
{
	global_cntx_t_list[op_type].pack_s.packa_rs = packa_rs;
	global_cntx_t_list[op_type].pack_s.packa_cs = packa_cs;
	global_cntx_t_list[op_type].pack_s.packb_rs = packb_rs;
	global_cntx_t_list[op_type].pack_s.packb_cs = packb_cs;
}

static void _lpgemm_cntx_init_blksz_map()
{
#define XMACRO(ID,MC,NC,KC,MR,NR,PACKA_RS,PACKA_CS,PACKB_RS,PACKB_CS) \
	lpgemm_set_block_sizes_global_cntx(ID, MC, NC, KC, MR, NR); \
	lpgemm_set_pack_strides_global_cntx(ID, PACKA_RS, PACKA_CS, PACKB_RS, PACKB_CS);

	// Ideally the blocksize needs to be set based on arch id. However
	// since this code is also expected to work on other vendor machines,
	// the blocksize for a particular version of zen id is generalized
	// for all machines that support the ISA supported by that particular
	// zen id.
	if ( bli_cpuid_is_avx512vnni_supported() == TRUE )
	{
		LPGEMM_BLKSZ_MAP_ZEN4
	}
	else if ( bli_cpuid_is_avx2fma3_supported() == TRUE )
	{
		LPGEMM_BLKSZ_MAP_ZEN
	}
	else
	{
		LPGEMM_BLKSZ_MAP_ZEN
	}

#undef XMACRO
}

static void lpgemm_cntx_init_map()
{
	_lpgemm_cntx_init_func_map();
	_lpgemm_cntx_init_blksz_map();
	_lpgemm_util_cntx_init_func_map();
}

// Sets default block sizes for lpgemm. Currently only u8s8s32 supported.
void aocl_lpgemm_init_global_cntx()
{
	bli_pthread_once
	(
	  &once_check_lpgemm_func_map_init,
	  lpgemm_cntx_init_map
	);
}

lpgemm_cntx_t* lpgemm_get_global_cntx_obj( AOCL_OPERATION_TYPE op )
{
	return &global_cntx_t_list[op];
}

lpgemm_util_cntx_t* lpgemm_util_get_global_cntx_obj( AOCL_UTIL_OPERATION_TYPE op )
{
	return &global_util_cntx_t_list[op];
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

void lpgemm_get_packa_strides( lpgemm_cntx_t* lcntx, dim_t* rs, dim_t* cs )
{
	*rs = lcntx->pack_s.packa_rs;
	*cs = lcntx->pack_s.packa_cs;
}

void lpgemm_get_packb_strides( lpgemm_cntx_t* lcntx, dim_t* rs, dim_t* cs )
{
	*rs = lcntx->pack_s.packb_rs;
	*cs = lcntx->pack_s.packb_cs;
}

void lpgemm_mod_block_size_s16
     (
       dim_t m,
       dim_t n,
       dim_t k,
       dim_t* MC,
       dim_t* NC,
       dim_t* KC
     )
{
	const dim_t range[4] = {1024, 512, 256, 128};

	if (n < *NC)
	{
		for (dim_t i = 0; i < 4; ++i)
		{
			if (n <= range[i])
			{
				*NC = range[i];
			}
		}
	}

	if (k < *KC)
	{
		for (dim_t i = 0; i < 4; ++i)
		{
			if (k <= range[i])
			{
				*KC = range[i];
			}
		}
	}
}
