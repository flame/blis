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

#ifndef BLIS_DOTJS_H
#define BLIS_DOTJS_H

// dotjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - The third char encodes the type of rho.
// - x is used in conjugated form.

// -- (xyr) = (ss?) ------------------------------------------------------------

#define bl2_sssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_ssddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_sscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sd?) ------------------------------------------------------------

#define bl2_sdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_sdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_sdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sc?) ------------------------------------------------------------

#define bl2_scsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_scddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_sccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_sczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sz?) ------------------------------------------------------------

#define bl2_szsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_szddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_szcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_szzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (ds?) ------------------------------------------------------------

#define bl2_dssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_dsddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_dscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dd?) ------------------------------------------------------------

#define bl2_ddsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bl2_ddddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bl2_ddcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_ddzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dc?) ------------------------------------------------------------

#define bl2_dcsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_dcddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_dccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dz?) ------------------------------------------------------------

#define bl2_dzsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bl2_dzddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bl2_dzcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bl2_dzzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (cs?) ------------------------------------------------------------

#define bl2_cssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_csddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_cscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  )-(x).imag * ( float  ) (y); \
}
#define bl2_cszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double )-(x).imag * ( double ) (y); \
}

// -- (xyr) = (cd?) ------------------------------------------------------------

#define bl2_cdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_cdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_cdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  )-(x).imag * ( float  ) (y); \
}
#define bl2_cdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double )-(x).imag * ( double ) (y); \
}

// -- (xyr) = (cc?) ------------------------------------------------------------

#define bl2_ccsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_ccddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_cccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_cczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (cz?) ------------------------------------------------------------

#define bl2_czsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_czddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_czcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_czzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zs?) ------------------------------------------------------------

#define bl2_zssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_zsddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_zscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_zszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zd?) ------------------------------------------------------------

#define bl2_zdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bl2_zdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bl2_zdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bl2_zdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zc?) ------------------------------------------------------------

#define bl2_zcsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_zcddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_zccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_zczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zz?) ------------------------------------------------------------

#define bl2_zzsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bl2_zzddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bl2_zzcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bl2_zzzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}



#define bl2_sdotjs( x, y, a ) \
{ \
	bl2_sssdotjs( x, y, a ); \
}
#define bl2_ddotjs( x, y, a ) \
{ \
	bl2_ddddotjs( x, y, a ); \
}
#define bl2_cdotjs( x, y, a ) \
{ \
	bl2_cccdotjs( x, y, a ); \
}
#define bl2_zdotjs( x, y, a ) \
{ \
	bl2_zzzdotjs( x, y, a ); \
}


#endif
