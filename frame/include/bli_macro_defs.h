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

#ifndef BLIS_MACRO_DEFS_H
#define BLIS_MACRO_DEFS_H


// -- Undefine restrict for C++ and C89/90 --

#ifdef __cplusplus
  // Language is C++; define restrict as nothing.
  #define restrict
#elif __STDC_VERSION__ >= 199901L
  // Language is C99 (or later); do nothing since restrict is recognized.
#else
  // Language is pre-C99; define restrict as nothing.
  #define restrict
#endif


// -- Boolean values --

#ifndef TRUE
  #define TRUE  1
#endif

#ifndef FALSE
  #define FALSE 0
#endif


// -- Concatenation macros --

#define BLIS_FUNC_PREFIX_STR       "bli"

// We add an extra layer the definitions of these string-pasting macros
// because sometimes it is needed if, for example, one of the PASTE
// macros is invoked with an "op" argument that is itself a macro.

#define PASTEMAC0_(op)             bli_ ## op
#define PASTEMAC0(op)              PASTEMAC0_(op)

#define PASTEMAC_(ch,op)           bli_ ## ch  ## op
#define PASTEMAC(ch,op)            PASTEMAC_(ch,op)

#define PASTEMAC2_(ch1,ch2,op)     bli_ ## ch1 ## ch2 ## op
#define PASTEMAC2(ch1,ch2,op)      PASTEMAC2_(ch1,ch2,op)

#define PASTEMAC3_(ch1,ch2,ch3,op) bli_ ## ch1 ## ch2 ## ch3 ## op
#define PASTEMAC3(ch1,ch2,ch3,op)  PASTEMAC3_(ch1,ch2,ch3,op)

#define PASTEBLACHK_(op)           bla_ ## op ## _check
#define PASTEBLACHK(op)            PASTEBLACHK_(op)

#define PASTECH_(ch,op)            ch ## op
#define PASTECH(ch,op)             PASTECH_(ch,op)

#define MKSTR(s1)                  #s1
#define STRINGIFY_INT( s )         MKSTR( s )


// -- Include other groups of macros

#include "bli_genarray_macro_defs.h"
#include "bli_gentfunc_macro_defs.h"
#include "bli_gentprot_macro_defs.h"

#include "bli_mem_macro_defs.h"
#include "bli_pool_macro_defs.h"
#include "bli_obj_macro_defs.h"
#include "bli_param_macro_defs.h"
#include "bli_complex_macro_defs.h"
#include "bli_scalar_macro_defs.h"
#include "bli_error_macro_defs.h"
#include "bli_blas_macro_defs.h"
#include "bli_auxinfo_macro_defs.h"


#endif
