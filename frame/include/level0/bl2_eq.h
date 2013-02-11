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

#ifndef BLIS_EQ_H
#define BLIS_EQ_H


// eq1

#define bl2_seq1( a ) \
\
	( (a) == 1.0F )

#define bl2_deq1( a ) \
\
	( (a) == 1.0 )

#define bl2_ceq1( a ) \
\
	( (a).real == 1.0F && (a).imag == 0.0F )

#define bl2_zeq1( a ) \
\
	( (a).real == 1.0 && (a).imag == 0.0 )

// eq0

#define bl2_seq0( a ) \
\
	( (a) == 0.0F )

#define bl2_deq0( a ) \
\
	( (a) == 0.0 )

#define bl2_ceq0( a ) \
\
	( (a).real == 0.0F && (a).imag == 0.0F )

#define bl2_zeq0( a ) \
\
	( (a).real == 0.0 && (a).imag == 0.0 )

// eqm1

#define bl2_seqm1( a ) \
\
	( (a) == -1.0F )

#define bl2_deqm1( a ) \
\
	( (a) == -1.0 )

#define bl2_ceqm1( a ) \
\
	( (a).real == -1.0F && (a).imag == 0.0F )

#define bl2_zeqm1( a ) \
\
	( (a).real == -1.0 && (a).imag == 0.0 )


// eq (passed by value)

#define bl2_seq( a, b ) \
\
	( (a) == (b) )

#define bl2_deq( a, b ) \
\
	( (a) == (b) )

#define bl2_ceq( a, b ) \
\
	( ( (a).real == (b).real ) && \
	  ( (a).imag == (b).imag ) )

#define bl2_zeq( a, b ) \
\
	( ( (a).real == (b).real ) && \
	  ( (a).imag == (b).imag ) )

#define bl2_ieq( a, b ) \
\
	( (a) == (b) )

// eqa (passed by address)

#define bl2_seqa( a, b ) \
\
	( *(( float* )(a)) == *(( float* )(b)) )

#define bl2_deqa( a, b ) \
\
	( *(( double* )(a)) == *(( double* )(b)) )

#define bl2_ceqa( a, b ) \
\
	( ( (( scomplex* )(a))->real == (( scomplex* )(b))->real ) && \
	  ( (( scomplex* )(a))->imag == (( scomplex* )(b))->imag ) )

#define bl2_zeqa( a, b ) \
\
	( ( (( dcomplex* )(a))->real == (( dcomplex* )(b))->real ) && \
	  ( (( dcomplex* )(a))->imag == (( dcomplex* )(b))->imag ) )

#define bl2_ieqa( a, b ) \
\
	( *(( int* )(a)) == *(( int* )(b)) )

// imageq0

#define bl2_simageq0( a ) \
\
	( TRUE )

#define bl2_dimageq0( a ) \
\
	( TRUE )

#define bl2_cimageq0( a ) \
\
	( (( scomplex* )(a))->imag == ( float  ) 0.0 )

#define bl2_zimageq0( a ) \
\
	( (( dcomplex* )(a))->imag == ( double ) 0.0 )


#endif
