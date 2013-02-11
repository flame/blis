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

#ifndef BLIS_COPYNZJS_H
#define BLIS_COPYNZJS_H

// copynzjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - x is copied in conjugated form.

#define bl2_sscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bl2_dscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bl2_cscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}
#define bl2_zscopynzjs( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}

#define bl2_sdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bl2_ddcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bl2_cdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}
#define bl2_zdcopynzjs( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}

#define bl2_sccopynzjs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F; (SKIP COPYING OF ZERO) */ \
}
#define bl2_dccopynzjs( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	/* (y).imag = 0.0F; (SKIP COPYING OF ZERO) */ \
}
#define bl2_cccopynzjs( x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	(y).imag = ( float  ) -(x).imag; \
}
#define bl2_zccopynzjs( x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	(y).imag = ( float  ) -(x).imag; \
}

#define bl2_szcopynzjs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bl2_dzcopynzjs( x, y ) \
{ \
	(y).real = ( double ) (x); \
	/* (y).imag = 0.0; (SKIP COPYING OF ZERO) */ \
}
#define bl2_czcopynzjs( x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	(y).imag = ( double ) -(x).imag; \
}
#define bl2_zzcopynzjs( x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	(y).imag = ( double ) -(x).imag; \
}


#define bl2_scopynzjs( x, y ) \
{ \
	bl2_sscopynzjs( x, y ); \
}
#define bl2_dcopynzjs( x, y ) \
{ \
	bl2_ddcopynzjs( x, y ); \
}
#define bl2_ccopynzjs( x, y ) \
{ \
	bl2_cccopynzjs( x, y ); \
}
#define bl2_zcopynzjs( x, y ) \
{ \
	bl2_zzcopynzjs( x, y ); \
}


#endif
