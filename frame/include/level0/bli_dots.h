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

#ifndef BLIS_DOTS_H
#define BLIS_DOTS_H

// dots

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - The third char encodes the type of rho.

// -- (xyr) = (ss?) ------------------------------------------------------------

#define bli_sssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_ssddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_sscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sd?) ------------------------------------------------------------

#define bli_sdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_sdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_sdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sc?) ------------------------------------------------------------

#define bli_scsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_scddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_sccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sz?) ------------------------------------------------------------

#define bli_szsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_szddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_szcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_szzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (ds?) ------------------------------------------------------------

#define bli_dssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_dsddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_dscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dd?) ------------------------------------------------------------

#define bli_ddsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_ddddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_ddcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_ddzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dc?) ------------------------------------------------------------

#define bli_dcsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_dcddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_dccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dz?) ------------------------------------------------------------

#define bli_dzsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_dzddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_dzcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dzzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (cs?) ------------------------------------------------------------

#define bli_cssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_csddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_cscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_cszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (cd?) ------------------------------------------------------------

#define bli_cdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_cdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_cdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_cdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (cc?) ------------------------------------------------------------

#define bli_ccsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_ccddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_cccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_cczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (cz?) ------------------------------------------------------------

#define bli_czsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_czddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_czcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_czzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zs?) ------------------------------------------------------------

#define bli_zssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_zsddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_zscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_zszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zd?) ------------------------------------------------------------

#define bli_zdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_zdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_zdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_zdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zc?) ------------------------------------------------------------

#define bli_zcsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_zcddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_zccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_zczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zz?) ------------------------------------------------------------

#define bli_zzsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_zzddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_zzcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_zzzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}



#define bli_sdots( x, y, a ) \
{ \
	bli_sssdots( x, y, a ); \
}
#define bli_ddots( x, y, a ) \
{ \
	bli_ddddots( x, y, a ); \
}
#define bli_cdots( x, y, a ) \
{ \
	bli_cccdots( x, y, a ); \
}
#define bli_zdots( x, y, a ) \
{ \
	bli_zzzdots( x, y, a ); \
}


#endif
