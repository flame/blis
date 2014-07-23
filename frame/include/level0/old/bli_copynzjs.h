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

#ifndef BLIS_COPYNZJS_H
#define BLIS_COPYNZJS_H

// copynzjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - x is copied in conjugated form.

#define bli_sscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_dscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_cscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}
#define bli_zscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}

#define bli_sdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_ddcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_cdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}
#define bli_zdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}

#define bli_sccopynzjs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F; (SKIP COPYING OF ZERO) */ \
}
#define bli_dccopynzjs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F; (SKIP COPYING OF ZERO) */ \
}
#define bli_cccopynzjs( x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	(y).imag = ( float  ) -(x).imag; \
}
#define bli_zccopynzjs( x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	(y).imag = ( float  ) -(x).imag; \
}

#define bli_szcopynzjs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bli_dzcopynzjs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bli_czcopynzjs( x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	(y).imag = ( double ) -(x).imag; \
}
#define bli_zzcopynzjs( x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	(y).imag = ( double ) -(x).imag; \
}


#define bli_scopynzjs( x, y ) \
{ \
	bli_sscopynzjs( x, y ); \
}
#define bli_dcopynzjs( x, y ) \
{ \
	bli_ddcopynzjs( x, y ); \
}
#define bli_ccopynzjs( x, y ) \
{ \
	bli_cccopynzjs( x, y ); \
}
#define bli_zcopynzjs( x, y ) \
{ \
	bli_zzcopynzjs( x, y ); \
}


#endif
