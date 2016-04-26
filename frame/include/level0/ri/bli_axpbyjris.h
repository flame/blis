/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#ifndef BLIS_AXPBYJRIS_H
#define BLIS_AXPBYJRIS_H

// axpbyjris

#define bli_saxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
	(yr)        = (ar) * (xr)               + (br) * (yr); \
}

#define bli_daxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
	(yr)        = (ar) * (xr)               + (br) * (yr); \
}

#define bli_caxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r = (ar) * (xr) + (ai) * (xi) + (br) * (yr) - (bi) * (yi); \
    float  yt_i = (ai) * (xr) - (ar) * (xi) + (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_sccaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r =  (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    float  yt_i = -(ar) * (xi) + (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_ccsaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r = (ar) * (xr) + (ai) * (xi) + (br) * (yr); \
    float  yt_i = (ai) * (xr) - (ar) * (xi) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_cscaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r = (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    float  yt_i = (ai) * (xr) + (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_sscaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r = (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    float  yt_i =               (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_cssaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r = (ar) * (xr) + (br) * (yr); \
    float  yt_i = (ai) * (xr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_scsaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    float  yt_r =  (ar) * (xr) + (br) * (yr); \
    float  yt_i = -(ar) * (xi) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_zaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
	double yt_r = (ar) * (xr) + (ai) * (xi) + (br) * (yr) - (bi) * (yi); \
	double yt_i = (ai) * (xr) - (ar) * (xi) + (bi) * (yr) + (br) * (yi); \
	(yr) = yt_r; \
	(yi) = yt_i; \
}

#define bli_dzzaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r =  (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    double yt_i = -(ar) * (xi) + (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_zzdaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r = (ar) * (xr) + (ai) * (xi) + (br) * (yr); \
    double yt_i = (ai) * (xr) - (ar) * (xi) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_zdzaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r = (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    double yt_i = (ai) * (xr) + (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_ddzaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r = (ar) * (xr) + (br) * (yr) - (bi) * (yi); \
    double yt_i =               (bi) * (yr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_zddaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r = (ar) * (xr) + (br) * (yr); \
    double yt_i = (ai) * (xr) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#define bli_dzdaxpbyjris( ar, ai, xr, xi, br, bi, yr, yi ) \
{ \
    double yt_r =  (ar) * (xr) + (br) * (yr); \
    double yt_i = -(ar) * (xi) + (br) * (yi); \
    (yr) = yt_r; \
    (yi) = yt_i; \
}

#endif

