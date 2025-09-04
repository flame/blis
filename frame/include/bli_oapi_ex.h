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

// This file defines macros used to allow the _oapi.c files to produce
// object APIs that contain context parameters.

// Define a macro that allows the source code to determine which interface
// (basic or expert) we are compiling.
#undef  BLIS_OAPI_EXPERT
#define BLIS_OAPI_EXPERT

// Define the macro to add a suffix to the function names (in function
// definitions).
#undef  EX_SUF
#define EX_SUF BLIS_OAPI_EX_SUF

// Define the macro to add expert arguments to function signatures
// and prototypes.
#undef  BLIS_OAPI_EX_PARAMS
#define BLIS_OAPI_EX_PARAMS   ,cntx_t* cntx, rntm_t* rntm

// Define the macro to omit the expert variable declaration block, since
// it is not needed when expert parameters are passed in through the API.
#undef  BLIS_OAPI_EX_DECLS
#define BLIS_OAPI_EX_DECLS

