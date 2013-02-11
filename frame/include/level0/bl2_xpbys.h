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

#ifndef BLIS_XPBYS_H
#define BLIS_XPBYS_H

// xpbys

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of b.
// - The third char encodes the type of y.

// -- (xby) = (ss?) ------------------------------------------------------------

#define bl2_sssxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( float  ) (x)      + ( float  ) (b)      * ( float  ) (y)      ); \
}
#define bl2_ssdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_sscxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( float  ) (x)      + ( float  ) (b)      * ( float  ) (y).real ); \
}
#define bl2_sszxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}

// -- (xby) = (sd?) ------------------------------------------------------------

#define bl2_sdsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_sddxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_sdcxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}
#define bl2_sdzxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}

// -- (xby) = (sc?) ------------------------------------------------------------

#define bl2_scsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( float  ) (x)      + ( float  ) (b).real * ( float  ) (y)      ); \
}
#define bl2_scdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_sccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( float  ) (x)      + ( float  ) (b).real * ( float  ) (y).real - \
	                                                 ( float  ) (b).imag * ( float  ) (y).imag ); \
	float  tempi = ( float  )( ( float  ) (x)      + ( float  ) (b).imag * ( float  ) (y).real + \
	                                                 ( float  ) (b).real * ( float  ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_sczxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (sz?) ------------------------------------------------------------

#define bl2_szsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_szdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_szcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_szzxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}


// -- (xby) = (ds?) ------------------------------------------------------------

#define bl2_dssxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( float  ) (x)      + ( float  ) (b)      * ( float  ) (y)      ); \
}
#define bl2_dsdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_dscxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( float  ) (x)      + ( float  ) (b)      * ( float  ) (y).real ); \
}
#define bl2_dszxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}

// -- (xby) = (dd?) ------------------------------------------------------------

#define bl2_ddsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_dddxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_ddcxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}
#define bl2_ddzxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x)      + ( double ) (b)      * ( double ) (y).real ); \
}

// -- (xby) = (dc?) ------------------------------------------------------------

#define bl2_dcsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( float  ) (x)      + ( float  ) (b).real * ( float  ) (y)      ); \
}
#define bl2_dcdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_dccxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( float  ) (x)      + ( float  ) (b).real * ( float  ) (y).real - \
	                                                 ( float  ) (b).imag * ( float  ) (y).imag ); \
	double tempi = ( double )( ( float  ) (x)      + ( float  ) (b).imag * ( float  ) (y).real + \
	                                                 ( float  ) (b).real * ( float  ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_dczxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (dz?) ------------------------------------------------------------

#define bl2_dzsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_dzdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_dzcxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_dzzxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x)      + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x)      + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (cs?) ------------------------------------------------------------

#define bl2_cssxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( float  ) (x).real + ( float  ) (b)      * ( float  ) (y)      ); \
}
#define bl2_csdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_cscxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( float  ) (x).real + ( float  ) (b)      * ( float  ) (y).real ); \
	(y).imag     = ( float  )( ( float  ) (x).imag + ( float  ) (b)      * ( float  ) (y).imag ); \
}
#define bl2_cszxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( float  )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}

// -- (xby) = (cd?) ------------------------------------------------------------

#define bl2_cdsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_cddxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_cdcxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( float  )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}
#define bl2_cdzxpbys( x, b, y ) \
{ \
	(y).real     = ( float  )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( float  )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}

// -- (xby) = (cc?) ------------------------------------------------------------

#define bl2_ccsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( float  ) (x).real + ( float  ) (b).real * ( float  ) (y)      ); \
}
#define bl2_ccdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_cccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( float  ) (x).real + ( float  ) (b).real * ( float  ) (y).real - \
	                                                 ( float  ) (b).imag * ( float  ) (y).imag ); \
	float  tempi = ( float  )( ( float  ) (x).imag + ( float  ) (b).imag * ( float  ) (y).real + \
	                                                 ( float  ) (b).real * ( float  ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_cczxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (cz?) ------------------------------------------------------------

#define bl2_czsxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_czdxpbys( x, b, y ) \
{ \
	(y)          = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_czcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_czzxpbys( x, b, y ) \
{ \
	float  tempr = ( float  )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	float  tempi = ( float  )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (zs?) ------------------------------------------------------------

#define bl2_zssxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_zsdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_zscxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( double )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}
#define bl2_zszxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( double )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}

// -- (xby) = (zd?) ------------------------------------------------------------

#define bl2_zdsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_zddxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y)      ); \
}
#define bl2_zdcxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( double )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}
#define bl2_zdzxpbys( x, b, y ) \
{ \
	(y).real     = ( double )( ( double ) (x).real + ( double ) (b)      * ( double ) (y).real ); \
	(y).imag     = ( double )( ( double ) (x).imag + ( double ) (b)      * ( double ) (y).imag ); \
}

// -- (xby) = (zc?) ------------------------------------------------------------

#define bl2_zcsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_zcdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_zccxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_zczxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}

// -- (xby) = (zz?) ------------------------------------------------------------

#define bl2_zzsxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_zzdxpbys( x, b, y ) \
{ \
	(y)          = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y)      ); \
}
#define bl2_zzcxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}
#define bl2_zzzxpbys( x, b, y ) \
{ \
	double tempr = ( double )( ( double ) (x).real + ( double ) (b).real * ( double ) (y).real - \
	                                                 ( double ) (b).imag * ( double ) (y).imag ); \
	double tempi = ( double )( ( double ) (x).imag + ( double ) (b).imag * ( double ) (y).real + \
	                                                 ( double ) (b).real * ( double ) (y).imag ); \
	(y).real  = tempr; \
	(y).imag  = tempi; \
}




#define bl2_sxpbys( x, b, y ) \
{ \
	bl2_sssxpbys( x, b, y ); \
}
#define bl2_dxpbys( x, b, y ) \
{ \
	bl2_dddxpbys( x, b, y ); \
}
#define bl2_cxpbys( x, b, y ) \
{ \
	bl2_cccxpbys( x, b, y ); \
}
#define bl2_zxpbys( x, b, y ) \
{ \
	bl2_zzzxpbys( x, b, y ); \
}


#endif
