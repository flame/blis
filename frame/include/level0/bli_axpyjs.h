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

#define bli_sssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_ssdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bli_sdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bli_scsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_scdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_sccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( float  )( ( float  ) (a)      * ( float  )-(x).imag ); \
}
#define bli_sczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( double )( ( float  ) (a)      * ( float  )-(x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bli_szsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_szzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bli_dssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dsdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bli_ddsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bli_dcsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dcdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_dczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bli_dzsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_dzzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bli_cssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_csdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_cscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_cszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bli_cdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_cdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bli_ccsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_ccdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_cccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_cczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bli_czsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_czzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bli_zssaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zsdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zscaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_zszaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bli_zdsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zddaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zdcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_zdzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bli_zcsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zcdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zccaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_zczaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bli_zzsaxpyjs( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzdaxpyjs( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzcaxpyjs( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_zzzaxpyjs( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}






#define bli_saxpyjs( a, x, y ) \
{ \
	bli_sssaxpyjs( a, x, y ); \
}
#define bli_daxpyjs( a, x, y ) \
{ \
	bli_dddaxpyjs( a, x, y ); \
}
#define bli_caxpyjs( a, x, y ) \
{ \
	bli_cccaxpyjs( a, x, y ); \
}
#define bli_zaxpyjs( a, x, y ) \
{ \
	bli_zzzaxpyjs( a, x, y ); \
}


#endif
