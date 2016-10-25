/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

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


// -- System headers --

#include "bli_system.h"


// -- BLIS configuration definition --

// NOTE: We include bli_config.h first because there might be something
// defined there that is needed within one of the system headers. A good
// example: posix_memalign() needs _GNU_SOURCE on GNU systems (I think).
// 
// PLEASE DON'T CHANGE THE ORDER IN WHICH HEADERS ARE INCLUDED UNLESS YOU
// KNOW WHAT YOU ARE DOING.

#include "bli_config.h"
#include "bli_config_macro_defs.h"


// -- Common BLIS definitions --

#include "bli_type_defs.h"
#include "bli_macro_defs.h"


// -- Threading definitions --

#include "bli_thread.h"


// -- Constant definitions --

#include "bli_extern_defs.h"


// -- BLIS kernel definitions --

#include "bli_kernel.h"

#include "bli_kernel_pre_macro_defs.h"
#include "bli_kernel_ind_pre_macro_defs.h"

#include "bli_kernel_macro_defs.h"
#include "bli_kernel_ind_macro_defs.h"

#include "bli_kernel_prototypes.h"


// -- Base operation prototypes --

#include "bli_init.h"
#include "bli_const.h"
#include "bli_obj.h"
#include "bli_obj_scalar.h"
#include "bli_cntx.h"
#include "bli_gks.h"
#include "bli_ind.h"
#include "bli_membrk.h"
#include "bli_pool.h"
#include "bli_memsys.h"
#include "bli_mem.h"
#include "bli_part.h"
#include "bli_prune.h"
#include "bli_query.h"
#include "bli_blksz.h"
#include "bli_func.h"
#include "bli_mbool.h"
#include "bli_auxinfo.h"
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

#include "bli_l0.h"


// -- Level-1v operations --

#include "bli_l1v.h"


// -- Level-1d operations --

#include "bli_l1d.h"


// -- Level-1f operations --

#include "bli_l1f.h"


// -- Level-1m operations --

#include "bli_l1m.h"


// -- Level-2 operations --

#include "bli_l2.h"


// -- Level-3 operations --

#include "bli_l3.h"


// -- Utility operations --

#include "bli_util.h"


// -- BLAS compatibility layer --

#include "bli_blas.h"


// -- CBLAS compatibility layer --

#include "bli_cblas.h"


// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

