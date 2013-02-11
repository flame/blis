/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

#ifndef BLIS_MACRO_DEFS_H
#define BLIS_MACRO_DEFS_H


// -- Boolean values --

#ifndef TRUE
  #define TRUE  1
#endif

#ifndef FALSE
  #define FALSE 0
#endif


// -- Concatenation macros --

#define BLIS_FUNC_PREFIX_STR      "bl2"

#define PASTEMAC0(op)             bl2_ ## op

#define PASTEMAC(ch,op)           bl2_ ## ch  ## op

#define PASTEMAC2(ch1,ch2,op)     bl2_ ## ch1 ## ch2 ## op

#define PASTEMAC3(ch1,ch2,ch3,op) bl2_ ## ch1 ## ch2 ## ch3 ## op


// -- Include other groups of macros

#include "bl2_genarray_macro_defs.h"
#include "bl2_gentfunc_macro_defs.h"
#include "bl2_gentprot_macro_defs.h"

#include "bl2_mem_macro_defs.h"
#include "bl2_obj_macro_defs.h"
#include "bl2_param_macro_defs.h"
#include "bl2_scalar_macro_defs.h"
#include "bl2_error_macro_defs.h"


#endif
