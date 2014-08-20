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

#ifndef BLIS_CONSTANTS_H
#define BLIS_CONSTANTS_H

// return pointers to constants

// 2

#define bli_s2 \
\
	( ( float*    ) bli_obj_buffer_for_const( BLIS_FLOAT,    BLIS_TWO ) )

#define bli_d2 \
\
	( ( double*   ) bli_obj_buffer_for_const( BLIS_DOUBLE,   BLIS_TWO ) )

#define bli_c2 \
\
	( ( scomplex* ) bli_obj_buffer_for_const( BLIS_SCOMPLEX, BLIS_TWO ) )

#define bli_z2 \
\
	( ( dcomplex* ) bli_obj_buffer_for_const( BLIS_DCOMPLEX, BLIS_TWO ) )

#define bli_i2 \
\
	( ( gint_t*   ) bli_obj_buffer_for_const( BLIS_INT,      BLIS_TWO ) )

// 1

#define bli_s1 \
\
	( ( float*    ) bli_obj_buffer_for_const( BLIS_FLOAT,    BLIS_ONE ) )

#define bli_d1 \
\
	( ( double*   ) bli_obj_buffer_for_const( BLIS_DOUBLE,   BLIS_ONE ) )

#define bli_c1 \
\
	( ( scomplex* ) bli_obj_buffer_for_const( BLIS_SCOMPLEX, BLIS_ONE ) )

#define bli_z1 \
\
	( ( dcomplex* ) bli_obj_buffer_for_const( BLIS_DCOMPLEX, BLIS_ONE ) )

#define bli_i1 \
\
	( ( gint_t*   ) bli_obj_buffer_for_const( BLIS_INT,      BLIS_ONE ) )

// 0

#define bli_s0 \
\
	( ( float*    ) bli_obj_buffer_for_const( BLIS_FLOAT,    BLIS_ZERO ) )

#define bli_d0 \
\
	( ( double*   ) bli_obj_buffer_for_const( BLIS_DOUBLE,   BLIS_ZERO ) )

#define bli_c0 \
\
	( ( scomplex* ) bli_obj_buffer_for_const( BLIS_SCOMPLEX, BLIS_ZERO ) )

#define bli_z0 \
\
	( ( dcomplex* ) bli_obj_buffer_for_const( BLIS_DCOMPLEX, BLIS_ZERO ) )

#define bli_i0 \
\
	( ( gint_t*   ) bli_obj_buffer_for_const( BLIS_INT,      BLIS_ZERO ) )

// -1

#define bli_sm1 \
\
	( ( float*    ) bli_obj_buffer_for_const( BLIS_FLOAT,    BLIS_MINUS_ONE ) )

#define bli_dm1 \
\
	( ( double*   ) bli_obj_buffer_for_const( BLIS_DOUBLE,   BLIS_MINUS_ONE ) )

#define bli_cm1 \
\
	( ( scomplex* ) bli_obj_buffer_for_const( BLIS_SCOMPLEX, BLIS_MINUS_ONE ) )

#define bli_zm1 \
\
	( ( dcomplex* ) bli_obj_buffer_for_const( BLIS_DCOMPLEX, BLIS_MINUS_ONE ) )

#define bli_im1 \
\
	( ( gint_t*   ) bli_obj_buffer_for_const( BLIS_INT,      BLIS_MINUS_ONE ) )

// -2

#define bli_sm2 \
\
	( ( float*    ) bli_obj_buffer_for_const( BLIS_FLOAT,    BLIS_MINUS_TWO ) )

#define bli_dm2 \
\
	( ( double*   ) bli_obj_buffer_for_const( BLIS_DOUBLE,   BLIS_MINUS_TWO ) )

#define bli_cm2 \
\
	( ( scomplex* ) bli_obj_buffer_for_const( BLIS_SCOMPLEX, BLIS_MINUS_TWO ) )

#define bli_zm2 \
\
	( ( dcomplex* ) bli_obj_buffer_for_const( BLIS_DCOMPLEX, BLIS_MINUS_TWO ) )

#define bli_im2 \
\
	( ( gint_t*   ) bli_obj_buffer_for_const( BLIS_INT,      BLIS_MINUS_TWO ) )


#endif

