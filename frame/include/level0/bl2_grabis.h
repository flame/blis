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

#ifndef BLIS_GRABIS_H
#define BLIS_GRABIS_H

// grabis

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bl2_ssgrabis( x, y ) \
{ \
	(y) = 0.0F; \
}
#define bl2_dsgrabis( x, y ) \
{ \
	(y) = 0.0F; \
}
#define bl2_csgrabis( x, y ) \
{ \
	(y) = ( float  ) (x).imag; \
}
#define bl2_zsgrabis( x, y ) \
{ \
	(y) = ( float  ) (x).imag; \
}

#define bl2_sdgrabis( x, y ) \
{ \
	(y) = 0.0; \
}
#define bl2_ddgrabis( x, y ) \
{ \
	(y) = 0.0; \
}
#define bl2_cdgrabis( x, y ) \
{ \
	(y) = ( double ) (x).imag; \
}
#define bl2_zdgrabis( x, y ) \
{ \
	(y) = ( double ) (x).imag; \
}

#define bl2_scgrabis( x, y ) \
{ \
	(y).real = 0.0F; \
	(y).imag = 0.0F; \
}
#define bl2_dcgrabis( x, y ) \
{ \
	(y).real = 0.0F; \
	(y).imag = 0.0F; \
}
#define bl2_ccgrabis( x, y ) \
{ \
	(y).real = ( float  ) (x).imag; \
	(y).imag = 0.0F; \
}
#define bl2_zcgrabis( x, y ) \
{ \
	(y).real = ( float  ) (x).imag; \
	(y).imag = 0.0F; \
}

#define bl2_szgrabis( x, y ) \
{ \
	(y).real = 0.0; \
	(y).imag = 0.0; \
}
#define bl2_dzgrabis( x, y ) \
{ \
	(y).real = 0.0; \
	(y).imag = 0.0; \
}
#define bl2_czgrabis( x, y ) \
{ \
	(y).real = ( double ) (x).imag; \
	(y).imag = 0.0; \
}
#define bl2_zzgrabis( x, y ) \
{ \
	(y).real = ( double ) (x).imag; \
	(y).imag = 0.0; \
}


#define bl2_sgrabis( x, y ) \
{ \
	bl2_ssgrabis( x, y ); \
}
#define bl2_dgrabis( x, y ) \
{ \
	bl2_ddgrabis( x, y ); \
}
#define bl2_cgrabis( x, y ) \
{ \
	bl2_ccgrabis( x, y ); \
}
#define bl2_zgrabis( x, y ) \
{ \
	bl2_zzgrabis( x, y ); \
}


#endif
