/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#define bl2_sssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_ssddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_sscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sd?) ------------------------------------------------------------

#define bl2_sdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_sdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_sdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sc?) ------------------------------------------------------------

#define bl2_scsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_scddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_sccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sz?) ------------------------------------------------------------

#define bl2_szsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_szddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_szcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_szzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (ds?) ------------------------------------------------------------

#define bl2_dssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_dsddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_dscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dd?) ------------------------------------------------------------

#define bl2_ddsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_ddddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_ddcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_ddzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dc?) ------------------------------------------------------------

#define bl2_dcsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_dcddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_dccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dz?) ------------------------------------------------------------

#define bl2_dzsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_dzddots( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_dzcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dzzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (cs?) ------------------------------------------------------------

#define bl2_cssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_csddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_cscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_cszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (cd?) ------------------------------------------------------------

#define bl2_cdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_cdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_cdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_cdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (cc?) ------------------------------------------------------------

#define bl2_ccsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_ccddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_cccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_cczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (cz?) ------------------------------------------------------------

#define bl2_czsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_czddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_czcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_czzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zs?) ------------------------------------------------------------

#define bl2_zssdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_zsddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_zscdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_zszdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zd?) ------------------------------------------------------------

#define bl2_zdsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_zdddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_zdcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_zdzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zc?) ------------------------------------------------------------

#define bl2_zcsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_zcddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_zccdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_zczdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zz?) ------------------------------------------------------------

#define bl2_zzsdots( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_zzddots( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_zzcdots( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag + ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_zzzdots( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag + ( double ) (x).imag * ( double ) (y).real; \
}



#define bl2_sdots( x, y, a ) \
{ \
	bl2_sssdots( x, y, a ); \
}
#define bl2_ddots( x, y, a ) \
{ \
	bl2_ddddots( x, y, a ); \
}
#define bl2_cdots( x, y, a ) \
{ \
	bl2_cccdots( x, y, a ); \
}
#define bl2_zdots( x, y, a ) \
{ \
	bl2_zzzdots( x, y, a ); \
}


#endif
