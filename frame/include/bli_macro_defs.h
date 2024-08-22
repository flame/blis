/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_MACRO_DEFS_H
#define BLIS_MACRO_DEFS_H


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

#define PASTEMAC4_(ch1,ch2,ch3,ch4,op) bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## op
#define PASTEMAC4(ch1,ch2,ch3,ch4,op)  PASTEMAC4_(ch1,ch2,ch3,ch4,op)

#define PASTEMAC5_(ch1,ch2,ch3,ch4,ch5,op) bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## op
#define PASTEMAC5(ch1,ch2,ch3,ch4,ch5,op)  PASTEMAC5_(ch1,ch2,ch3,ch4,ch5,op)

#define PASTEMAC6_(ch1,ch2,ch3,ch4,ch5,ch6,op) bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## ch6 ## op
#define PASTEMAC6(ch1,ch2,ch3,ch4,ch5,ch6,op)  PASTEMAC6_(ch1,ch2,ch3,ch4,ch5,ch6,op)

#define PASTEBLACHK_(op)           bla_ ## op ## _check
#define PASTEBLACHK(op)            PASTEBLACHK_(op)

#define PASTECH0_(op)              op
#define PASTECH0(op)               PASTECH0_(op)

#define PASTECH_(ch,op)            ch ## op
#define PASTECH(ch,op)             PASTECH_(ch,op)

#define PASTECH2_(ch1,ch2,op)      ch1 ## ch2 ## op
#define PASTECH2(ch1,ch2,op)       PASTECH2_(ch1,ch2,op)

#define PASTECH3_(ch1,ch2,ch3,op)  ch1 ## ch2 ## ch3 ## op
#define PASTECH3(ch1,ch2,ch3,op)   PASTECH3_(ch1,ch2,ch3,op)

#define MKSTR(s1)                  #s1
#define STRINGIFY_INT( s )         MKSTR( s )

#define PASTEMACT(ch1, ch2, ch3, ch4)   bli_ ## ch1 ## ch2 ## _ ## ch3 ## _ ## ch4
// name-mangling macros.
#ifdef BLIS_ENABLE_NO_UNDERSCORE_API
#define PASTEF770(name)                                                  name
#define PASTEF77(ch1,name)                                        ch1 ## name
#define PASTEF772(ch1,ch2,name)                            ch1 ## ch2 ## name
#define PASTEF773(ch1,ch2,ch3,name)                 ch1 ## ch2 ## ch3 ## name
#else
#define PASTEF770(name)                                            name ## _
#define PASTEF77(ch1,name)                                  ch1 ## name ## _
#define PASTEF772(ch1,ch2,name)                      ch1 ## ch2 ## name ## _
#define PASTEF773(ch1,ch2,ch3,name)           ch1 ## ch2 ## ch3 ## name ## _
#endif

// Macros to define names _blis_impl suffix, *_blis_impl is the blis
// blis implementation of the respective API's which is invoked from CBLAS
// and BLAS wrapper. 
#define PASTEF770S(name)                                   name ## _blis_impl
#define PASTEF77S(ch1,name)                         ch1 ## name ## _blis_impl
#define PASTEF772S(ch1,ch2,name)             ch1 ## ch2 ## name ## _blis_impl
#define PASTEF773S(ch1,ch2,ch3,name)  ch1 ## ch2 ## ch3 ## name ## _blis_impl

// -- Include other groups of macros

#include "bli_genarray_macro_defs.h"
#include "bli_gentdef_macro_defs.h"
#include "bli_gentfunc_macro_defs.h"
#include "bli_gentprot_macro_defs.h"

#include "bli_misc_macro_defs.h"
#include "bli_param_macro_defs.h"
#include "bli_obj_macro_defs.h"
#include "bli_complex_macro_defs.h"
#include "bli_scalar_macro_defs.h"
#include "bli_error_macro_defs.h"
#include "bli_blas_macro_defs.h"
#include "bli_builtin_macro_defs.h"

#include "bli_oapi_macro_defs.h"
#include "bli_tapi_macro_defs.h"

// -- Include definitions for BLAS interfaces

#include "bli_blas_interface_defs.h"
#include "bli_blas_blis_impl_interface_defs.h"

#endif
