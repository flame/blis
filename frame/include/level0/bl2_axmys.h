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

#ifndef BLIS_AXMYS_H
#define BLIS_AXMYS_H

// axmys

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bl2_sssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_ssdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bl2_sdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bl2_scsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_scdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_sccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag -= ( float  )( ( float  ) (a)      * ( float  ) (x).imag ); \
}
#define bl2_sczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag -= ( double )( ( float  ) (a)      * ( float  ) (x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bl2_szsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_szzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bl2_dssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dsdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bl2_ddsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bl2_dcsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dcdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_dczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bl2_dzsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_dzzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bl2_cssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_csdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_cscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_cszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bl2_cdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_cdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bl2_ccsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_ccdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_cccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_cczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bl2_czsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_czzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bl2_zssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zsdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_zszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bl2_zdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_zdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bl2_zcsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zcdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_zczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bl2_zzsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_zzzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}






#define bl2_saxmys( a, x, y ) \
{ \
	bl2_sssaxmys( a, x, y ); \
}
#define bl2_daxmys( a, x, y ) \
{ \
	bl2_dddaxmys( a, x, y ); \
}
#define bl2_caxmys( a, x, y ) \
{ \
	bl2_cccaxmys( a, x, y ); \
}
#define bl2_zaxmys( a, x, y ) \
{ \
	bl2_zzzaxmys( a, x, y ); \
}


#endif
