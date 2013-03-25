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

#ifndef BLIS_NEG2S_H
#define BLIS_NEG2S_H

// neg2s

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bli_ssneg2s( x, y ) \
{ \
	(y) = ( float  ) -(x); \
}
#define bli_dsneg2s( x, y ) \
{ \
	(y) = ( float  ) -(x); \
}
#define bli_csneg2s( x, y ) \
{ \
	(y) = ( float  ) -(x).real; \
}
#define bli_zsneg2s( x, y ) \
{ \
	(y) = ( float  ) -(x).real; \
}

#define bli_sdneg2s( x, y ) \
{ \
	(y) = ( double ) -(x); \
}
#define bli_ddneg2s( x, y ) \
{ \
	(y) = ( double ) -(x); \
}
#define bli_cdneg2s( x, y ) \
{ \
	(y) = ( double ) -(x).real; \
}
#define bli_zdneg2s( x, y ) \
{ \
	(y) = ( double ) -(x).real; \
}

#define bli_scneg2s( x, y ) \
{ \
	(y).real = ( float  ) -(x); \
	(y).imag = 0.0F; \
}
#define bli_dcneg2s( x, y ) \
{ \
	(y).real = ( float  ) -(x); \
	(y).imag = 0.0F; \
}
#define bli_ccneg2s( x, y ) \
{ \
	(y).real = ( float  ) -(x).real; \
	(y).imag = ( float  ) -(x).imag; \
}
#define bli_zcneg2s( x, y ) \
{ \
	(y).real = ( float  ) -(x).real; \
	(y).imag = ( float  ) -(x).imag; \
}

#define bli_szneg2s( x, y ) \
{ \
	(y).real = ( double ) -(x); \
	(y).imag = 0.0; \
}
#define bli_dzneg2s( x, y ) \
{ \
	(y).real = ( double ) -(x); \
	(y).imag = 0.0; \
}
#define bli_czneg2s( x, y ) \
{ \
	(y).real = ( double ) -(x).real; \
	(y).imag = ( double ) -(x).imag; \
}
#define bli_zzneg2s( x, y ) \
{ \
	(y).real = ( double ) -(x).real; \
	(y).imag = ( double ) -(x).imag; \
}


#define bli_sneg2s( x, y ) \
{ \
	bli_ssneg2s( x, y ); \
}
#define bli_dneg2s( x, y ) \
{ \
	bli_ddneg2s( x, y ); \
}
#define bli_cneg2s( x, y ) \
{ \
	bli_ccneg2s( x, y ); \
}
#define bli_zneg2s( x, y ) \
{ \
	bli_zzneg2s( x, y ); \
}


#endif
