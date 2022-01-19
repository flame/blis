/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

// file define different formats of BLAS APIs- uppercase with
// and without underscore, lowercase without underscore.

#ifndef BLIS_ENABLE_NO_UNDERSCORE_API
#ifndef BLIS_ENABLE_UPPERCASE_API
//Level 1 APIs
BLIS_EXPORT_BLIS void SROTG(float  *sa, float  *sb, float  *c, float  *s);

BLIS_EXPORT_BLIS void srotg(float  *sa, float  *sb, float  *c, float  *s);

BLIS_EXPORT_BLIS void SROTG_(float  *sa, float  *sb, float  *c, float  *s);



BLIS_EXPORT_BLIS void SROTMG(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);

BLIS_EXPORT_BLIS void srotmg(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);

BLIS_EXPORT_BLIS void SROTMG_(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);



BLIS_EXPORT_BLIS void SROT(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void srot(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void SROT_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);



BLIS_EXPORT_BLIS void SROTM(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);

BLIS_EXPORT_BLIS void srotm(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);

BLIS_EXPORT_BLIS void SROTM_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);



BLIS_EXPORT_BLIS void SSWAP(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void sswap(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SSWAP_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS void SSCAL(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS void sscal(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS void SSCAL_(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);



BLIS_EXPORT_BLIS void SCOPY(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void scopy(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SCOPY_(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS void SAXPY(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void saxpy(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SAXPY_(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS float SDOT(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float sdot(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float SDOT_(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);



BLIS_EXPORT_BLIS float SDSDOT(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float sdsdot(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float SDSDOT_(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);



BLIS_EXPORT_BLIS float SNRM2(const f77_int *n, const float  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float snrm2(const f77_int *n, const float  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float SNRM2_(const f77_int *n, const float  *x,  const f77_int *incx);



BLIS_EXPORT_BLIS float SCNRM2(const f77_int *n, const scomplex  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float scnrm2(const f77_int *n, const scomplex  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float SCNRM2_(const f77_int *n, const scomplex  *x,  const f77_int *incx);



BLIS_EXPORT_BLIS float SASUM(const f77_int *n, const float  *sx,  const f77_int *incx);

BLIS_EXPORT_BLIS float sasum(const f77_int *n, const float  *sx,  const f77_int *incx);

BLIS_EXPORT_BLIS float SASUM_(const f77_int *n, const float  *sx,  const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ISAMAX(const f77_int *n, const float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int isamax(const f77_int *n, const float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int ISAMAX_(const f77_int *n, const float  *sx, const f77_int *incx);



BLIS_EXPORT_BLIS void DROTG(double *da, double *db, double *c, double *s);

BLIS_EXPORT_BLIS void drotg(double *da, double *db, double *c, double *s);

BLIS_EXPORT_BLIS void DROTG_(double *da, double *db, double *c, double *s);



BLIS_EXPORT_BLIS void DROTMG(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);

BLIS_EXPORT_BLIS void drotmg(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);

BLIS_EXPORT_BLIS void DROTMG_(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);



BLIS_EXPORT_BLIS void DROT(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void drot(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void DROT_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);



BLIS_EXPORT_BLIS void DROTM(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);

BLIS_EXPORT_BLIS void drotm(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);

BLIS_EXPORT_BLIS void DROTM_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);



BLIS_EXPORT_BLIS void DSWAP(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void dswap(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DSWAP_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS void DSCAL(const f77_int *n, const double *da, double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS void dscal(const f77_int *n, const double *da, double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS void DSCAL_(const f77_int *n, const double *da, double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS void DCOPY(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void dcopy(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DCOPY_(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS void DAXPY(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void daxpy(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DAXPY_(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS double DDOT(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS double ddot(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS double DDOT_(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS double DSDOT(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS double dsdot(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS double DSDOT_(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS double DNRM2(const f77_int *n, const double *x, const f77_int *incx);

BLIS_EXPORT_BLIS double dnrm2(const f77_int *n, const double *x, const f77_int *incx);

BLIS_EXPORT_BLIS double DNRM2_(const f77_int *n, const double *x, const f77_int *incx);



BLIS_EXPORT_BLIS double DZNRM2(const f77_int *n, const dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS double dznrm2(const f77_int *n, const dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS double DZNRM2_(const f77_int *n, const dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS double DASUM(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS double dasum(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS double DASUM_(const f77_int *n, const double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int IDAMAX(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int idamax(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int IDAMAX_(const f77_int *n, const double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS void CROTG(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);

BLIS_EXPORT_BLIS void crotg(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);

BLIS_EXPORT_BLIS void CROTG_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);



BLIS_EXPORT_BLIS void CSROT(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void csrot(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void CSROT_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);



BLIS_EXPORT_BLIS void CSWAP(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cswap(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CSWAP_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void CSCAL(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void cscal(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void CSCAL_(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);


BLIS_EXPORT_BLIS void CSSCAL(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void csscal(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void CSSCAL_(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);


BLIS_EXPORT_BLIS void CCOPY(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void ccopy(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CCOPY_(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);


BLIS_EXPORT_BLIS void CAXPY(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void caxpy(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CAXPY_(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx,scomplex  *cy, const f77_int *incy);


#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL

BLIS_EXPORT_BLIS scomplex CDOTC(const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex cdotc(const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex CDOTC_ (const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS scomplex CDOTU(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex cdotu(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex CDOTU_(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS dcomplex ZDOTC(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex zdotc (const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex ZDOTC_ (const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS dcomplex ZDOTU(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex zdotu (const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex ZDOTU_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

#else

BLIS_EXPORT_BLIS void CDOTC(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cdotc(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CDOTC_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void CDOTU(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cdotu(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CDOTU_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZDOTC(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zdotc(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZDOTC_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZDOTU(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zdotu(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZDOTU_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

#endif


BLIS_EXPORT_BLIS float SCASUM(const f77_int *n, const scomplex  *cx,  const f77_int *incx);

BLIS_EXPORT_BLIS float scasum(const f77_int *n, const scomplex  *cx,  const f77_int *incx);

BLIS_EXPORT_BLIS float SCASUM_(const f77_int *n, const scomplex  *cx,  const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ICAMAX(const f77_int *n, const scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int icamax(const f77_int *n, const scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int ICAMAX_(const f77_int *n, const scomplex  *cx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZROTG(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);

BLIS_EXPORT_BLIS void zrotg(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);

BLIS_EXPORT_BLIS void ZROTG_(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);



BLIS_EXPORT_BLIS void ZDROT(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void zdrot(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void ZDROT_(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);



BLIS_EXPORT_BLIS void ZSWAP(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zswap(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZSWAP_(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZSCAL(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void zscal(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void ZSCAL_(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZDSCAL(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void zdscal(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void ZDSCAL_(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZCOPY(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zcopy(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZCOPY_(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZAXPY(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zaxpy(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZAXPY_(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS double DZASUM(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS double dzasum(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS double DZASUM_(const f77_int *n, const dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int IZAMAX(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int izamax(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int IZAMAX_(const f77_int *n, const dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ICAMIN( const f77_int* n,  const scomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int icamin( const f77_int* n,  const scomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int ICAMIN_( const f77_int* n,  const scomplex* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int IDAMIN( const f77_int* n,  const double* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int idamin( const f77_int* n,  const double* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int IDAMIN_( const f77_int* n,  const double* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int ISAMIN( const f77_int* n,  const float* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int isamin( const f77_int* n,  const float* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int ISAMIN_( const f77_int* n,  const float* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int IZAMIN( const f77_int* n,  const dcomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int izamin( const f77_int* n,  const dcomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int IZAMIN_( const f77_int* n,  const dcomplex* x,  const f77_int* incx);



//Level 2 APIs
BLIS_EXPORT_BLIS void SGEMV(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sgemv(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SGEMV_(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SGBMV(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sgbmv(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SGBMV_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSYMV(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ssymv(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSYMV_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSBMV(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ssbmv(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSBMV_(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSPMV(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sspmv(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSPMV_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void STRMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void strmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STRMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STBMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stbmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STBMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STPMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stpmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STPMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STRSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void strsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STRSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STBSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stbsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STBSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STPSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stpsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STPSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void SGER(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void sger(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SGER_(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSYR(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ssyr(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SSYR_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSPR(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);

BLIS_EXPORT_BLIS void sspr(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);

BLIS_EXPORT_BLIS void SSPR_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);



BLIS_EXPORT_BLIS void SSYR2(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ssyr2(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SSYR2_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSPR2(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);

BLIS_EXPORT_BLIS void sspr2(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);

BLIS_EXPORT_BLIS void SSPR2_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);



BLIS_EXPORT_BLIS void DGEMV(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dgemv(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DGEMV_(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DGBMV(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dgbmv(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DGBMV_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSYMV(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dsymv(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSYMV_(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSBMV(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dsbmv(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSBMV_(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSPMV(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dspmv(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSPMV_(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DTRMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtrmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTRMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTBMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtbmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTBMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTPMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtpmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTPMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTRSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtrsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTRSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTBSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtbsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTBSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTPSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtpsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTPSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DGER(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dger(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DGER_(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSYR(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dsyr(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DSYR_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSPR(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);

BLIS_EXPORT_BLIS void dspr(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);

BLIS_EXPORT_BLIS void DSPR_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);



BLIS_EXPORT_BLIS void DSYR2(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dsyr2(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DSYR2_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSPR2(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);

BLIS_EXPORT_BLIS void dspr2(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);

BLIS_EXPORT_BLIS void DSPR2_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);



BLIS_EXPORT_BLIS void CGEMV(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void cgemv(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CGEMV_(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CGBMV(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void cgbmv(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CGBMV_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHEMV(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chemv(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHEMV_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHBMV(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chbmv(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHBMV_(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a,const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHPMV(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chpmv(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHPMV_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CTRMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctrmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTRMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTBMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctbmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTBMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTPMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctpmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTPMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTRSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const  scomplex *a, const f77_int *lda, scomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctrsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const  scomplex *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTRSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTBSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctbsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTBSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTPSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctpsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTPSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CGERC(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cgerc(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CGERC_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CGERU(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cgeru(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CGERU_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHER(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cher(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CHER_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHPR(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);

BLIS_EXPORT_BLIS void chpr(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);

BLIS_EXPORT_BLIS void CHPR_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);



BLIS_EXPORT_BLIS void CHER2(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cher2(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CHER2_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHPR2(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);

BLIS_EXPORT_BLIS void chpr2(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);

BLIS_EXPORT_BLIS void CHPR2_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);



BLIS_EXPORT_BLIS void ZGEMV(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zgemv(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZGEMV_(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZGBMV(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zgbmv(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZGBMV_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHEMV(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhemv(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHEMV_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHBMV(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhbmv(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHBMV_(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHPMV(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhpmv(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHPMV_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZTRMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztrmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTRMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTBMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztbmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTBMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTPMV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztpmv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTPMV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTRSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztrsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTRSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTBSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztbsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTBSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTPSV(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztpsv(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTPSV_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZGERU(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zgeru(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZGERU_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZGERC(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zgerc(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZGERC_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHER(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zher(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZHER_(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHPR(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);

BLIS_EXPORT_BLIS void zhpr(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);

BLIS_EXPORT_BLIS void ZHPR_(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);



BLIS_EXPORT_BLIS void ZHER2(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zher2(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZHER2_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHPR2(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);

BLIS_EXPORT_BLIS void zhpr2(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);

BLIS_EXPORT_BLIS void ZHPR2_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);



//Level 3 APIs
BLIS_EXPORT_BLIS void SGEMM(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void sgemm(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SGEMM_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssymm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYRK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssyrk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYRK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYR2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssyr2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYR2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void STRMM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void strmm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void STRMM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void STRSM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void strsm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void STRSM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void DGEMM(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dgemm(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DGEMM_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DSYMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsymm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DSYRK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsyrk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYRK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DSYR2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsyr2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYR2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DTRMM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void dtrmm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void DTRMM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void DTRSM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void dtrsm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void DTRSM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void CGEMM(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cgemm(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CGEMM_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csymm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHEMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void chemm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHEMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYRK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csyrk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYRK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHERK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cherk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHERK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYR2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csyr2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYR2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHER2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cher2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHER2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CTRMM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ctrmm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void CTRMM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void CTRSM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ctrsm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void CTRSM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void ZGEMM(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zgemm(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZGEMM_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsymm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHEMM(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zhemm(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHEMM_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYRK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsyrk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYRK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHERK(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zherk(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHERK_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYR2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsyr2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYR2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHER2K(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zher2k(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHER2K_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZTRMM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ztrmm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ZTRMM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void ZTRSM(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ztrsm(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ZTRSM_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);



// Miscellaneous APIs
BLIS_EXPORT_BLIS void CDOTCSUB( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void cdotcsub( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void CDOTCSUB_( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);



BLIS_EXPORT_BLIS void CDOTUSUB( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void cdotusub( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void CDOTUSUB_( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);



BLIS_EXPORT_BLIS void DASUMSUB(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dasumsub(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DASUMSUB_(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void DDOTSUB(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);

BLIS_EXPORT_BLIS void ddotsub(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);

BLIS_EXPORT_BLIS void DDOTSUB_(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);



BLIS_EXPORT_BLIS void DNRM2SUB(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);

BLIS_EXPORT_BLIS void dnrm2sub(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);

BLIS_EXPORT_BLIS void DNRM2SUB_(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);



BLIS_EXPORT_BLIS void DZASUMSUB(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dzasumsub(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DZASUMSUB_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void DZNRM2SUB(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dznrm2sub(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DZNRM2SUB_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void ICAMAXSUB(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void icamaxsub(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ICAMAXSUB_(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ICAMINSUB( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void icaminsub( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ICAMINSUB_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IDAMAXSUB( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void idamaxsub( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IDAMAXSUB_( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IDAMINSUB(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void idaminsub(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IDAMINSUB_(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ISAMAXSUB( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void isamaxsub( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ISAMAXSUB_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ISAMINSUB( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void isaminsub( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ISAMINSUB_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IZAMINSUB( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void izaminsub( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IZAMINSUB_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IZAMAXSUB( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void izamaxsub( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IZAMAXSUB_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void SASUMSUB( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void sasumsub( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SASUMSUB_( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SCASUMSUB( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void scasumsub( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SCASUMSUB_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SCNRM2SUB( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void scnrm2sub( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SCNRM2SUB_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SDOTSUB( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);

BLIS_EXPORT_BLIS void sdotsub( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);

BLIS_EXPORT_BLIS void SDOTSUB_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);



BLIS_EXPORT_BLIS void SNRM2SUB( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);

BLIS_EXPORT_BLIS void snrm2sub( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);

BLIS_EXPORT_BLIS void SNRM2SUB_( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);



BLIS_EXPORT_BLIS void ZDOTCSUB( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void zdotcsub( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void ZDOTCSUB_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);



BLIS_EXPORT_BLIS void ZDOTUSUB( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void zdotusub( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void ZDOTUSUB_( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);



BLIS_EXPORT_BLIS void SDSDOTSUB( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);

BLIS_EXPORT_BLIS void sdsdotsub( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);

BLIS_EXPORT_BLIS void SDSDOTSUB_( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);



BLIS_EXPORT_BLIS void DSDOTSUB( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);

BLIS_EXPORT_BLIS void dsdotsub( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);

BLIS_EXPORT_BLIS void DSDOTSUB_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);



BLIS_EXPORT_BLIS f77_int LSAME(const char   *ca, const char   *cb, const f77_int a, const f77_int b);

BLIS_EXPORT_BLIS f77_int lsame(const char   *ca, const char   *cb, const f77_int a, const f77_int b);

BLIS_EXPORT_BLIS f77_int LSAME_(const char   *ca, const char   *cb, const f77_int a, const f77_int b);



BLIS_EXPORT_BLIS int XERBLA(const char   *srname, const f77_int *info, ftnlen n);

BLIS_EXPORT_BLIS int xerbla(const char   *srname, const f77_int *info, ftnlen n);

BLIS_EXPORT_BLIS int XERBLA_(const char   *srname, const f77_int *info, ftnlen n);



//Auxiliary APIs
BLIS_EXPORT_BLIS double DCABS1(bla_dcomplex *z);

BLIS_EXPORT_BLIS double dcabs1(bla_dcomplex *z);

BLIS_EXPORT_BLIS double DCABS1_(bla_dcomplex *z);



BLIS_EXPORT_BLIS float SCABS1(bla_scomplex* z);

BLIS_EXPORT_BLIS float scabs1(bla_scomplex* z);

BLIS_EXPORT_BLIS float SCABS1_(bla_scomplex* z);



//BLAS Extension APIs
BLIS_EXPORT_BLIS void CAXPBY( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void caxpby( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void CAXPBY_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void CGEMM3M( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void cgemm3m( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void CGEMM3M_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void CGEMM_BATCH( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void cgemm_batch( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void CGEMM_BATCH_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void CGEMMT( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void cgemmt( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void CGEMMT_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void DAXPBY(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void daxpby(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void DAXPBY_(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void DGEMM_BATCH( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void dgemm_batch( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void DGEMM_BATCH_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void DGEMMT( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void dgemmt( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void DGEMMT_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void SAXPBY( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void saxpby( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void SAXPBY_( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void SGEMM_BATCH(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void sgemm_batch(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void SGEMM_BATCH_(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void SGEMMT( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void sgemmt( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void SGEMMT_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void ZAXPBY( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void zaxpby( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void ZAXPBY_( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void ZGEMM3M( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void zgemm3m( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void ZGEMM3M_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void ZGEMM_BATCH(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void zgemm_batch(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void ZGEMM_BATCH_(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void ZGEMMT( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void zgemmt( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void ZGEMMT_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void CIMATCOPY(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha, scomplex* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void cimatcopy(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha, scomplex* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void CIMATCOPY_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha, scomplex* aptr,  f77_int* lda,  f77_int* ldb);



BLIS_EXPORT_BLIS void COMATADD(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const scomplex* alpha,  const scomplex* A,  f77_int* lda, const scomplex* beta,  scomplex* B,  f77_int* ldb,  scomplex* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void comatadd(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const scomplex* alpha,  const scomplex* A,  f77_int* lda, const scomplex* beta,  scomplex* B,  f77_int* ldb,  scomplex* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void COMATADD_(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const scomplex* alpha,  const scomplex* A,  f77_int* lda, const scomplex* beta,  scomplex* B,  f77_int* ldb,  scomplex* C,  f77_int* ldc);



BLIS_EXPORT_BLIS void COMATCOPY2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda, f77_int* stridea,  scomplex* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void comatcopy2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda, f77_int* stridea,  scomplex* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void COMATCOPY2_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda, f77_int* stridea,  scomplex* bptr,  f77_int* ldb, f77_int* strideb);



BLIS_EXPORT_BLIS void COMATCOPY(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda,  scomplex* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void comatcopy(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda,  scomplex* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void COMATCOPY_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const scomplex* alpha,  const scomplex* aptr,  f77_int* lda,  scomplex* bptr,  f77_int* ldb);



BLIS_EXPORT_BLIS void DOMATADD(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const double* alpha,  const double* A,  f77_int* lda,  const double* beta,  const double* B,  f77_int* ldb,  double* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void domatadd(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const double* alpha,  const double* A,  f77_int* lda,  const double* beta,  const double* B,  f77_int* ldb,  double* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void DOMATADD_(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const double* alpha,  const double* A,  f77_int* lda,  const double* beta,  const double* B,  f77_int* ldb,  double* C,  f77_int* ldc);



BLIS_EXPORT_BLIS void DOMATCOPY2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda, f77_int* stridea,  double* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void domatcopy2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda, f77_int* stridea,  double* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void DOMATCOPY2_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda, f77_int* stridea,  double* bptr,  f77_int* ldb, f77_int* strideb);



BLIS_EXPORT_BLIS void DOMATCOPY(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda,  double* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void domatcopy(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda,  double* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void DOMATCOPY_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const double* alpha,  const double* aptr,  f77_int* lda,  double* bptr,  f77_int* ldb);



BLIS_EXPORT_BLIS void SIMATCOPY( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha, float* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void simatcopy( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha, float* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void SIMATCOPY_( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha, float* aptr,  f77_int* lda,  f77_int* ldb);



BLIS_EXPORT_BLIS void SOMATADD( f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const float* alpha,  const float* A,  f77_int* lda,  const float* beta,  const float* B,  f77_int* ldb,  float* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void somatadd( f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const float* alpha,  const float* A,  f77_int* lda,  const float* beta,  const float* B,  f77_int* ldb,  float* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void SOMATADD_( f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const float* alpha,  const float* A,  f77_int* lda,  const float* beta,  const float* B,  f77_int* ldb,  float* C,  f77_int* ldc);



BLIS_EXPORT_BLIS void SOMATCOPY2( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda, f77_int* stridea,  float* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void somatcopy2( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda, f77_int* stridea,  float* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void SOMATCOPY2_( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda, f77_int* stridea,  float* bptr,  f77_int* ldb, f77_int* strideb);



BLIS_EXPORT_BLIS void SOMATCOPY( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda,  float* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void somatcopy( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda,  float* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void SOMATCOPY_( f77_char* trans,  f77_int* rows,  f77_int* cols,  const float* alpha,  const float* aptr,  f77_int* lda,  float* bptr,  f77_int* ldb);



BLIS_EXPORT_BLIS void ZIMATCOPY(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha, dcomplex* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void zimatcopy(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha, dcomplex* aptr,  f77_int* lda,  f77_int* ldb);

BLIS_EXPORT_BLIS void ZIMATCOPY_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha, dcomplex* aptr,  f77_int* lda,  f77_int* ldb);



BLIS_EXPORT_BLIS void ZOMATADD(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const dcomplex* alpha,  const dcomplex* A,  f77_int* lda, const dcomplex* beta,  dcomplex* B,  f77_int* ldb,  dcomplex* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void zomatadd(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const dcomplex* alpha,  const dcomplex* A,  f77_int* lda, const dcomplex* beta,  dcomplex* B,  f77_int* ldb,  dcomplex* C,  f77_int* ldc);

BLIS_EXPORT_BLIS void ZOMATADD_(f77_char* transa, f77_char* transb,  f77_int* m,  f77_int* n,  const dcomplex* alpha,  const dcomplex* A,  f77_int* lda, const dcomplex* beta,  dcomplex* B,  f77_int* ldb,  dcomplex* C,  f77_int* ldc);



BLIS_EXPORT_BLIS void ZOMATCOPY2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda, f77_int* stridea,  dcomplex* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void zomatcopy2(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda, f77_int* stridea,  dcomplex* bptr,  f77_int* ldb, f77_int* strideb);

BLIS_EXPORT_BLIS void ZOMATCOPY2_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda, f77_int* stridea,  dcomplex* bptr,  f77_int* ldb, f77_int* strideb);



BLIS_EXPORT_BLIS void ZOMATCOPY(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda,  dcomplex* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void zomatcopy(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda,  dcomplex* bptr,  f77_int* ldb);

BLIS_EXPORT_BLIS void ZOMATCOPY_(f77_char* trans,  f77_int* rows,  f77_int* cols,  const dcomplex* alpha,  const dcomplex* aptr,  f77_int* lda,  dcomplex* bptr,  f77_int* ldb);


#endif
#endif
