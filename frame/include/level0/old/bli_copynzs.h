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

#ifndef BLIS_COPYNZS_H
#define BLIS_COPYNZS_H

// copynzs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bli_sscopynzs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_dscopynzs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_cscopynzs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}
#define bli_zscopynzs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}

#define bli_sdcopynzs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_ddcopynzs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_cdcopynzs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}
#define bli_zdcopynzs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}

#define bli_sccopynzs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F; (SKIP COPYING OF ZERO) */ \
}
#define bli_dccopynzs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F (SKIP COPYING OF ZERO) */; \
}
#define bli_cccopynzs( x, y ) \
{ \
	(y).real = ( float  ) (x).real; \
	(y).imag = ( float  ) (x).imag; \
}
#define bli_zccopynzs( x, y ) \
{ \
	(y).real = ( float  ) (x).real; \
	(y).imag = ( float  ) (x).imag; \
}

#define bli_szcopynzs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bli_dzcopynzs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bli_czcopynzs( x, y ) \
{ \
	(y).real = ( double ) (x).real; \
	(y).imag = ( double ) (x).imag; \
}
#define bli_zzcopynzs( x, y ) \
{ \
	(y).real = ( double ) (x).real; \
	(y).imag = ( double ) (x).imag; \
}


#define bli_scopynzs( x, y ) \
{ \
	bli_sscopynzs( x, y ); \
}
#define bli_dcopynzs( x, y ) \
{ \
	bli_ddcopynzs( x, y ); \
}
#define bli_ccopynzs( x, y ) \
{ \
	bli_cccopynzs( x, y ); \
}
#define bli_zcopynzs( x, y ) \
{ \
	bli_zzcopynzs( x, y ); \
}


#endif
