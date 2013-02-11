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

#ifndef BLIS_SETRIS_H
#define BLIS_SETRIS_H

// setris

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bl2_sssetris( xr, xi, y ) \
{ \
	(y)      = ( float  ) (xr); \
}
#define bl2_dssetris( xr, xi, y ) \
{ \
	(y)      = ( float  ) (xr); \
}


#define bl2_sdsetris( xr, xi, y ) \
{ \
	(y)      = ( double ) (xr); \
}
#define bl2_ddsetris( xr, xi, y ) \
{ \
	(y)      = ( double ) (xr); \
}


#define bl2_scsetris( xr, xi, y ) \
{ \
	(y).real = ( float  ) (xr); \
	(y).imag = ( float  ) (xi); \
}
#define bl2_dcsetris( xr, xi, y ) \
{ \
	(y).real = ( float  ) (xr); \
	(y).imag = ( float  ) (xi); \
}


#define bl2_szsetris( xr, xi, y ) \
{ \
	(y).real = ( double ) (xr); \
	(y).imag = ( double ) (xi); \
}
#define bl2_dzsetris( xr, xi, y ) \
{ \
	(y).real = ( double ) (xr); \
	(y).imag = ( double ) (xi); \
}


#define bl2_ssetris( xr, xi, y ) \
{ \
	bl2_sssetris( xr, xi, y ); \
}
#define bl2_dsetris( xr, xi, y ) \
{ \
	bl2_ddsetris( xr, xi, y ); \
}
#define bl2_csetris( xr, xi, y ) \
{ \
	bl2_scsetris( xr, xi, y ); \
}
#define bl2_zsetris( xr, xi, y ) \
{ \
	bl2_dzsetris( xr, xi, y ); \
}


#endif
