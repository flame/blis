/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#ifndef BLIS_H
#define BLIS_H


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif


// -- BLIS configuration definition --

// NOTE: These definitions are placed here mainly because there might be
// something in bl2_config.h that is needed within one of the system
// headers. A good example: posix_memalign() needs _GNU_SOURCE on GNU
// systems (I think).

#include "bl2_config.h"
#include "bl2_arch.h"
#include "bl2_kernel.h"


// -- System headers --

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Handle the results of checking for time.h and sys/time.h.
// gettimeofday() needs this.
#if HAVE_SYS_TIME_H
  #include <sys/time.h>
#endif
#include <time.h>


// -- BLIS definitions --

#include "bl2_type_defs.h"
#include "bl2_macro_defs.h"
#include "bl2_const_defs.h"
#include "bl2_cntl.h"

#include "bl2_extern_defs.h"


// -- Base operation prototypes --

#include "bl2_init.h"
#include "bl2_malloc.h"
#include "bl2_obj.h"
#include "bl2_mem.h"
#include "bl2_part.h"
#include "bl2_query.h"
#include "bl2_blocksize.h"
#include "bl2_param_map.h"
#include "bl2_clock.h"
#include "bl2_check.h"
#include "bl2_error.h"
#include "bl2_machval.h"


// -- Level-0 operations --

#include "bl2_copysc.h"


// -- Level-1 operations --

// one vector operand
#include "bl2_invertv.h"
#include "bl2_scalv.h"
#include "bl2_setv.h"
// two vector operands
#include "bl2_axpyv.h"
#include "bl2_copyv.h"
#include "bl2_copynzv.h"
#include "bl2_dotv.h"
#include "bl2_dotxv.h"
#include "bl2_scal2v.h"
#include "bl2_packv.h"
#include "bl2_unpackv.h"


// -- Level-1d operations --

// one diagonal operand
#include "bl2_invertd.h"
#include "bl2_scald.h"
#include "bl2_setd.h"
// two diagonal operands
#include "bl2_axpyd.h"
#include "bl2_copyd.h"
#include "bl2_scal2d.h"


// -- Level-1f operations --

#include "bl2_axpy2v.h"
#include "bl2_axpyf.h"
#include "bl2_dotxf.h"
#include "bl2_dotaxpyv.h"
#include "bl2_dotxaxpyf.h"


// -- Level-1m operations --

// one matrix operand
#include "bl2_scalm.h"
#include "bl2_setm.h"
// two matrix operands
#include "bl2_axpym.h"
#include "bl2_copym.h"
#include "bl2_copynzm.h"
#include "bl2_scal2m.h"
#include "bl2_packm.h"
#include "bl2_unpackm.h"


// -- Level-2 operations --

#include "bl2_gemv.h"
#include "bl2_ger.h"
#include "bl2_hemv.h"
#include "bl2_her.h"
#include "bl2_her2.h"
#include "bl2_symv.h"
#include "bl2_syr.h"
#include "bl2_syr2.h"
#include "bl2_trmv.h"
#include "bl2_trsv.h"


// -- Helper operands for ukernels --

#include "bl2_dupl.h"


// -- Level-3 operations --

#include "bl2_gemm.h"
#include "bl2_hemm.h"
#include "bl2_herk.h"
#include "bl2_her2k.h"
#include "bl2_symm.h"
#include "bl2_syrk.h"
#include "bl2_syr2k.h"
#include "bl2_trmm.h"
#include "bl2_trmm3.h"
#include "bl2_trsm.h"


// -- Utility operations --

#include "bl2_printv.h"
#include "bl2_printm.h"
#include "bl2_randv.h"
#include "bl2_randm.h"
#include "bl2_sets.h"


// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

