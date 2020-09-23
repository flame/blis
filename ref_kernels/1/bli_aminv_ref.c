/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
        dim_t            n, \
        ctype*  restrict x, inc_t incx, \
        dim_t*  restrict i_min, \
        cntx_t* restrict cntx  \
     ) \
{ \
    dim_t*   zero_i    = PASTEMAC(i,0); \
\
    ctype_r  chi1_r; \
    ctype_r  chi1_i; \
    ctype_r  abs_chi1; \
    ctype_r  abs_chi1_min; \
    dim_t    i_min_l; \
\
    /* If the vector length is zero, return early. This directly emulates
      the behavior of netlib BLAS's i?amin() routines. */ \
    if ( bli_zero_dim1( n ) ) \
    { \
      PASTEMAC(i,copys)( *zero_i, *i_min ); \
      return; \
    } \
\
    /* Initialize the index of the minimum absolute value to zero. */ \
    PASTEMAC(i,copys)( *zero_i, i_min_l ); \
\
	  /* Initialize the minimum absolute value with the first value of the vector,
       By this the comparison will happen across all vector elements. */ \
\
    ctype* chi1 = x; \
\
    /* Get the real and imaginary components of chi1. */ \
    PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
    /* Replace chi1_r and chi1_i with their absolute values. */ \
    PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
    PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
    PASTEMAC(chr,sets)( chi1_r, chi1_i, abs_chi1_min ); \
\
    if ( incx == 1 ) \
    { \
        ctype* chi1 = x; \
\
        for ( dim_t i = 0; i < n; ++i ) \
        {  \
            /* Get the real and imaginary components of chi1. */ \
            PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
            /* Replace chi1_r and chi1_i with their absolute values. */ \
            PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
            PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
            /* Add the real and imaginary absolute values together. */ \
            PASTEMAC(chr,set0s)( abs_chi1 ); \
            PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
            PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
            /* If the absolute value of the current element is less that of
            the previous smallest, save it and its index. If NaN is
            encountered, then treat it the same as if it were a valid
            value that was larger than any previously seen.*/ \
            if ( (abs_chi1_min > abs_chi1) || ( bli_isnan( abs_chi1 ) && !bli_isnan( abs_chi1_min ) ) ) \
            { \
              abs_chi1_min = abs_chi1; \
              i_min_l      = i; \
            } \
\
            chi1 += 1; \
        } \
    } \
    else \
    { \
        for ( dim_t i = 0; i < n; ++i ) \
        { \
            ctype* chi1 = x + (i  )*incx; \
\
            /* Get the real and imaginary components of chi1. */ \
            PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
            /* Replace chi1_r and chi1_i with their absolute values. */ \
            PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
            PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
            /* Add the real and imaginary absolute values together. */ \
            PASTEMAC(chr,set0s)( abs_chi1 ); \
            PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
            PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
            /* If the absolute value of the current element is less that of
            the previous smallest, save it and its index. If NaN is
                    encountered, then treat it the same as if it were a valid
                    value that was larger than any previously seen.*/ \
            if ( (abs_chi1_min < abs_chi1) || ( bli_isnan( abs_chi1 ) && !bli_isnan( abs_chi1_min ) ) ) \
            { \
              abs_chi1_min = abs_chi1; \
              i_min_l      = i; \
            } \
        } \
    } \
\
    /* Store the final index to the output variable. */ \
    PASTEMAC(i,copys)( i_min_l, *i_min ); \
}

INSERT_GENTFUNCR_BASIC2( aminv, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
