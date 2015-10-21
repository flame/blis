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

#ifndef BLIS_H
#define BLIS_H


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif


// -- BLIS configuration definition --

// NOTE: We include bli_config.h first because there might be something
// defined there that is needed within one of the system headers. A good
// example: posix_memalign() needs _GNU_SOURCE on GNU systems (I think).
// 
// PLEASE DON'T CHANGE THE ORDER IN WHICH HEADERS ARE INCLUDED UNLESS YOU
// KNOW WHAT YOU ARE DOING.

#include "bli_config.h"
#include "bli_config_macro_defs.h"


// -- System headers --

#include "bli_system.h"


// -- Common BLIS definitions --

#include "bli_type_defs.h"
#include "bli_macro_defs.h"

#include "bli_level3_type_defs.h"


// -- Threading definitions --

#include "bli_threading.h"


// -- Constant definitions --

#include "bli_extern_defs.h"


// -- BLIS kernel definitions --

#include "bli_kernel.h"
#include "bli_kernel_type_defs.h"

#include "bli_kernel_pre_macro_defs.h"
#include "bli_kernel_ind_pre_macro_defs.h"

#include "bli_kernel_macro_defs.h"
#include "bli_kernel_ind_macro_defs.h"

#include "bli_kernel_post_macro_defs.h"

#include "bli_kernel_prototypes.h"
#include "bli_kernel_ind_prototypes.h"


// -- BLIS memory pool definitions --

//#include "bli_mem_pool_macro_defs.h"


// -- Base operation prototypes --

#include "bli_init.h"
#include "bli_const.h"
#include "bli_malloc.h"
#include "bli_obj.h"
#include "bli_obj_scalar.h"
#include "bli_ind.h"
#include "bli_pool.h"
#include "bli_mem.h"
#include "bli_part.h"
#include "bli_prune.h"
#include "bli_query.h"
#include "bli_blocksize.h"
#include "bli_func.h"
#include "bli_param_map.h"
#include "bli_clock.h"
#include "bli_check.h"
#include "bli_error.h"
#include "bli_f2c.h"
#include "bli_machval.h"
#include "bli_getopt.h"
#include "bli_opid.h"
#include "bli_cntl.h"
#include "bli_info.h"


// -- Level-0 operations --

#include "bli_absqsc.h"
#include "bli_addsc.h"
#include "bli_copysc.h"
#include "bli_divsc.h"
#include "bli_getsc.h"
#include "bli_mulsc.h"
#include "bli_normfsc.h"
#include "bli_setsc.h"
#include "bli_sqrtsc.h"
#include "bli_subsc.h"
#include "bli_zipsc.h"
#include "bli_unzipsc.h"


// -- Level-1 operations --

// one vector operand
#include "bli_invertv.h"
#include "bli_scalv.h"
#include "bli_setv.h"
// two vector operands
#include "bli_addv.h"
#include "bli_axpyv.h"
#include "bli_copyv.h"
#include "bli_dotv.h"
#include "bli_dotxv.h"
#include "bli_scal2v.h"
#include "bli_subv.h"
#include "bli_swapv.h"
#include "bli_packv.h"
#include "bli_unpackv.h"


// -- Level-1d operations --

// one diagonal operand
#include "bli_invertd.h"
#include "bli_scald.h"
#include "bli_setd.h"
#include "bli_setid.h"
// two diagonal operands
#include "bli_addd.h"
#include "bli_axpyd.h"
#include "bli_copyd.h"
#include "bli_scal2d.h"
#include "bli_subd.h"


// -- Level-1f operations --

#include "bli_axpy2v.h"
#include "bli_axpyf.h"
#include "bli_dotxf.h"
#include "bli_dotaxpyv.h"
#include "bli_dotxaxpyf.h"


// -- Level-1m operations --

// one matrix operand
#include "bli_scalm.h"
#include "bli_setm.h"
// two matrix operands
#include "bli_addm.h"
#include "bli_axpym.h"
#include "bli_copym.h"
#include "bli_scal2m.h"
#include "bli_subm.h"
#include "bli_packm.h"
#include "bli_unpackm.h"


// -- Level-2 operations --

#include "bli_gemv.h"
#include "bli_ger.h"
#include "bli_hemv.h"
#include "bli_her.h"
#include "bli_her2.h"
#include "bli_symv.h"
#include "bli_syr.h"
#include "bli_syr2.h"
#include "bli_trmv.h"
#include "bli_trsv.h"


// -- Level-3 operations --

#include "bli_gemm.h"
#include "bli_hemm.h"
#include "bli_herk.h"
#include "bli_her2k.h"
#include "bli_symm.h"
#include "bli_syrk.h"
#include "bli_syr2k.h"
#include "bli_trmm.h"
#include "bli_trmm3.h"
#include "bli_trsm.h"


// -- Utility operations --

#include "bli_amaxv.h"
#include "bli_asumv.h"
#include "bli_mkherm.h"
#include "bli_mksymm.h"
#include "bli_mktrim.h"
#include "bli_norm1v.h"
#include "bli_norm1m.h"
#include "bli_normfv.h"
#include "bli_normfm.h"
#include "bli_normiv.h"
#include "bli_normim.h"
#include "bli_printv.h"
#include "bli_printm.h"
#include "bli_randv.h"
#include "bli_randm.h"
#include "bli_sumsqv.h"


// -- CBLAS compatibility layer --

#include "bli_cblas.h"

// -- BLAS compatibility layer --

#include "bli_blas.h"


// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

