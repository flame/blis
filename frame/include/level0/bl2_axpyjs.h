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

#ifndef BLIS_AXPYJS_H
#define BLIS_AXPYJS_H

// axpyjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.
// - x is used in conjugated form.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bl2_sssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_ssdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bl2_sdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bl2_scsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_scdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_sccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( float  )( ( float  ) (a)      * ( float  )-(x).imag ); \
}
#define bl2_sczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( double )( ( float  ) (a)      * ( float  )-(x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bl2_szsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bl2_szzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bl2_dssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dsdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bl2_ddsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bl2_dcsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dcdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bl2_dczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bl2_dzsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bl2_dzzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bl2_cssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_csdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_cscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_cszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bl2_cdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_cdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bl2_ccsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_ccdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_cccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_cczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bl2_czsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_czzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bl2_zssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zsdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_zszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bl2_zdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_zdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bl2_zcsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zcdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_zczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bl2_zzsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_zzzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}






#define bl2_saxpyjs( a, x, y ) \
{ \
	bl2_sssaxpyjs( a, x, y ); \
}
#define bl2_daxpyjs( a, x, y ) \
{ \
	bl2_dddaxpyjs( a, x, y ); \
}
#define bl2_caxpyjs( a, x, y ) \
{ \
	bl2_cccaxpyjs( a, x, y ); \
}
#define bl2_zaxpyjs( a, x, y ) \
{ \
	bl2_zzzaxpyjs( a, x, y ); \
}


#endif
