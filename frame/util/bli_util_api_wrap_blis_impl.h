/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLI_UTIL_API_WRAP_BLIS_IMPL_H_
#define BLI_UTIL_API_WRAP_BLIS_IMPL_H_

// file define different formats of BLAS _blis_impl APIs- uppercase with
// and without underscore, lowercase without underscore.

#ifndef BLIS_ENABLE_NO_UNDERSCORE_API
#ifndef BLIS_ENABLE_UPPERCASE_API
//Level 1 APIs
BLIS_EXPORT_BLIS void SROTG_BLIS_IMPL(float  *sa, float  *sb, float  *c, float  *s);

BLIS_EXPORT_BLIS void srotg_blis_impl_(float  *sa, float  *sb, float  *c, float  *s);

BLIS_EXPORT_BLIS void SROTG_BLIS_IMPL_(float  *sa, float  *sb, float  *c, float  *s);



BLIS_EXPORT_BLIS void SROTMG_BLIS_IMPL(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);

BLIS_EXPORT_BLIS void srotmg_blis_impl_(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);

BLIS_EXPORT_BLIS void SROTMG_BLIS_IMPL_(float  *sd1, float  *sd2, float  *sx1, const float  *sy1, float  *sparam);



BLIS_EXPORT_BLIS void SROT_BLIS_IMPL(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void srot_blis_impl_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void SROT_BLIS_IMPL_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *c, const float  *s);



BLIS_EXPORT_BLIS void SROTM_BLIS_IMPL(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);

BLIS_EXPORT_BLIS void srotm_blis_impl_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);

BLIS_EXPORT_BLIS void SROTM_BLIS_IMPL_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy, const float  *sparam);



BLIS_EXPORT_BLIS void SSWAP_BLIS_IMPL(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void sswap_blis_impl_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SSWAP_BLIS_IMPL_(const f77_int *n, float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS void SSCAL_BLIS_IMPL(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS void sscal_blis_impl_(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS void SSCAL_BLIS_IMPL_(const f77_int *n, const float  *sa, float  *sx, const f77_int *incx);



BLIS_EXPORT_BLIS void SCOPY_BLIS_IMPL(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void scopy_blis_impl_(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SCOPY_BLIS_IMPL_(const f77_int *n, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS void SAXPY_BLIS_IMPL(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void saxpy_blis_impl_(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS void SAXPY_BLIS_IMPL_(const f77_int *n, const float  *sa, const float  *sx, const f77_int *incx, float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS float SDOT_BLIS_IMPL(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float sdot_blis_impl_(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float SDOT_BLIS_IMPL_(const f77_int *n, const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);



BLIS_EXPORT_BLIS float SDSDOT_BLIS_IMPL(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float sdsdot_blis_impl_(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);

BLIS_EXPORT_BLIS float SDSDOT_BLIS_IMPL_(const f77_int *n, const float  *sb,  const float  *sx,  const f77_int *incx,  const float  *sy,  const f77_int *incy);



BLIS_EXPORT_BLIS float SNRM2_BLIS_IMPL(const f77_int *n, const float  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float snrm2_blis_impl_(const f77_int *n, const float  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float SNRM2_BLIS_IMPL_(const f77_int *n, const float  *x,  const f77_int *incx);



BLIS_EXPORT_BLIS float SCNRM2_BLIS_IMPL(const f77_int *n, const scomplex  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float scnrm2_blis_impl_(const f77_int *n, const scomplex  *x,  const f77_int *incx);

BLIS_EXPORT_BLIS float SCNRM2_BLIS_IMPL_(const f77_int *n, const scomplex  *x,  const f77_int *incx);



BLIS_EXPORT_BLIS float SASUM_BLIS_IMPL(const f77_int *n, const float  *sx,  const f77_int *incx);

BLIS_EXPORT_BLIS float sasum_blis_impl_(const f77_int *n, const float  *sx,  const f77_int *incx);

BLIS_EXPORT_BLIS float SASUM_BLIS_IMPL_(const f77_int *n, const float  *sx,  const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ISAMAX_BLIS_IMPL(const f77_int *n, const float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int isamax_blis_impl_(const f77_int *n, const float  *sx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int ISAMAX_BLIS_IMPL_(const f77_int *n, const float  *sx, const f77_int *incx);



BLIS_EXPORT_BLIS void DROTG_BLIS_IMPL(double *da, double *db, double *c, double *s);

BLIS_EXPORT_BLIS void drotg_blis_impl_(double *da, double *db, double *c, double *s);

BLIS_EXPORT_BLIS void DROTG_BLIS_IMPL_(double *da, double *db, double *c, double *s);



BLIS_EXPORT_BLIS void DROTMG_BLIS_IMPL(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);

BLIS_EXPORT_BLIS void drotmg_blis_impl_(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);

BLIS_EXPORT_BLIS void DROTMG_BLIS_IMPL_(double *dd1, double *dd2, double *dx1, const double *dy1, double *dparam);



BLIS_EXPORT_BLIS void DROT_BLIS_IMPL(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void drot_blis_impl_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void DROT_BLIS_IMPL_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *c, const double *s);



BLIS_EXPORT_BLIS void DROTM_BLIS_IMPL(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);

BLIS_EXPORT_BLIS void drotm_blis_impl_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);

BLIS_EXPORT_BLIS void DROTM_BLIS_IMPL_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy, const double *dparam);



BLIS_EXPORT_BLIS void DSWAP_BLIS_IMPL(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void dswap_blis_impl_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DSWAP_BLIS_IMPL_(const f77_int *n, double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS void DSCAL_BLIS_IMPL(const f77_int *n, const double *da, double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS void dscal_blis_impl_(const f77_int *n, const double *da, double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS void DSCAL_BLIS_IMPL_(const f77_int *n, const double *da, double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS void DCOPY_BLIS_IMPL(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void dcopy_blis_impl_(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DCOPY_BLIS_IMPL_(const f77_int *n, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS void DAXPY_BLIS_IMPL(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void daxpy_blis_impl_(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS void DAXPY_BLIS_IMPL_(const f77_int *n, const double *da, const double *dx, const f77_int *incx, double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS double DDOT_BLIS_IMPL(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS double ddot_blis_impl_(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);

BLIS_EXPORT_BLIS double DDOT_BLIS_IMPL_(const f77_int *n, const double *dx, const f77_int *incx, const double *dy, const f77_int *incy);



BLIS_EXPORT_BLIS double DSDOT_BLIS_IMPL(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS double dsdot_blis_impl_(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);

BLIS_EXPORT_BLIS double DSDOT_BLIS_IMPL_(const f77_int *n, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy);



BLIS_EXPORT_BLIS double DNRM2_BLIS_IMPL(const f77_int *n, const double *x, const f77_int *incx);

BLIS_EXPORT_BLIS double dnrm2_blis_impl_(const f77_int *n, const double *x, const f77_int *incx);

BLIS_EXPORT_BLIS double DNRM2_BLIS_IMPL_(const f77_int *n, const double *x, const f77_int *incx);



BLIS_EXPORT_BLIS double DZNRM2_BLIS_IMPL(const f77_int *n, const dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS double dznrm2_blis_impl_(const f77_int *n, const dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS double DZNRM2_BLIS_IMPL_(const f77_int *n, const dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS double DASUM_BLIS_IMPL(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS double dasum_blis_impl_(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS double DASUM_BLIS_IMPL_(const f77_int *n, const double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int IDAMAX_BLIS_IMPL(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int idamax_blis_impl_(const f77_int *n, const double *dx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int IDAMAX_BLIS_IMPL_(const f77_int *n, const double *dx, const f77_int *incx);



BLIS_EXPORT_BLIS void CROTG_BLIS_IMPL(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);

BLIS_EXPORT_BLIS void crotg_blis_impl_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);

BLIS_EXPORT_BLIS void CROTG_BLIS_IMPL_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c, scomplex  *s);



BLIS_EXPORT_BLIS void CSROT_BLIS_IMPL(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void csrot_blis_impl_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);

BLIS_EXPORT_BLIS void CSROT_BLIS_IMPL_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy, const float  *c, const float  *s);



BLIS_EXPORT_BLIS void CSWAP_BLIS_IMPL(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cswap_blis_impl_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CSWAP_BLIS_IMPL_(const f77_int *n, scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void CSCAL_BLIS_IMPL(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void cscal_blis_impl_(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void CSCAL_BLIS_IMPL_(const f77_int *n, const scomplex  *ca, scomplex  *cx, const f77_int *incx);


BLIS_EXPORT_BLIS void CSSCAL_BLIS_IMPL(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void csscal_blis_impl_(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS void CSSCAL_BLIS_IMPL_(const f77_int *n, const float  *sa, scomplex  *cx, const f77_int *incx);


BLIS_EXPORT_BLIS void CCOPY_BLIS_IMPL(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void ccopy_blis_impl_(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CCOPY_BLIS_IMPL_(const f77_int *n, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);


BLIS_EXPORT_BLIS void CAXPY_BLIS_IMPL(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void caxpy_blis_impl_(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx, scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CAXPY_BLIS_IMPL_(const f77_int *n, const scomplex  *ca, const scomplex  *cx, const f77_int *incx,scomplex  *cy, const f77_int *incy);


#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL

BLIS_EXPORT_BLIS scomplex CDOTC_BLIS_IMPL(const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex cdotc_blis_impl_(const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex CDOTC_BLIS_IMPL_(const f77_int* n, const scomplex*   x, const f77_int* incx, const scomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS scomplex CDOTU_BLIS_IMPL(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex cdotu_blis_impl_(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS scomplex CDOTU_BLIS_IMPL_(const f77_int* n, const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS dcomplex ZDOTC_BLIS_IMPL(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex zdotc_blis_impl_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex ZDOTC_BLIS_IMPL_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);



BLIS_EXPORT_BLIS dcomplex ZDOTU_BLIS_IMPL(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex zdotu_blis_impl_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

BLIS_EXPORT_BLIS dcomplex ZDOTU_BLIS_IMPL_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy);

#else

BLIS_EXPORT_BLIS void CDOTC_BLIS_IMPL(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cdotc_blis_impl_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CDOTC_BLIS_IMPL_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void CDOTU_BLIS_IMPL(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void cdotu_blis_impl_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);

BLIS_EXPORT_BLIS void CDOTU_BLIS_IMPL_(scomplex* retval, const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZDOTC_BLIS_IMPL(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zdotc_blis_impl_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZDOTC_BLIS_IMPL_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZDOTU_BLIS_IMPL(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zdotu_blis_impl_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZDOTU_BLIS_IMPL_(dcomplex* retval, const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy);

#endif


BLIS_EXPORT_BLIS float SCASUM_BLIS_IMPL(const f77_int *n, const scomplex  *cx,  const f77_int *incx);

BLIS_EXPORT_BLIS float scasum_blis_impl_(const f77_int *n, const scomplex  *cx,  const f77_int *incx);

BLIS_EXPORT_BLIS float SCASUM_BLIS_IMPL_(const f77_int *n, const scomplex  *cx,  const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ICAMAX_BLIS_IMPL(const f77_int *n, const scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int icamax_blis_impl_(const f77_int *n, const scomplex  *cx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int ICAMAX_BLIS_IMPL_(const f77_int *n, const scomplex  *cx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZROTG_BLIS_IMPL(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);

BLIS_EXPORT_BLIS void zrotg_blis_impl_(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);

BLIS_EXPORT_BLIS void ZROTG_BLIS_IMPL_(dcomplex *ca, bla_dcomplex *cb, bla_double *c, dcomplex *s);



BLIS_EXPORT_BLIS void ZDROT_BLIS_IMPL(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void zdrot_blis_impl_(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);

BLIS_EXPORT_BLIS void ZDROT_BLIS_IMPL_(const f77_int *n, dcomplex *cx, const f77_int *incx, dcomplex *cy, const f77_int *incy, const double *c, const double *s);



BLIS_EXPORT_BLIS void ZSWAP_BLIS_IMPL(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zswap_blis_impl_(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZSWAP_BLIS_IMPL_(const f77_int *n, dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZSCAL_BLIS_IMPL(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void zscal_blis_impl_(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void ZSCAL_BLIS_IMPL_(const f77_int *n, const dcomplex *za, dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZDSCAL_BLIS_IMPL(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void zdscal_blis_impl_(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS void ZDSCAL_BLIS_IMPL_(const f77_int *n, const double *da, dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS void ZCOPY_BLIS_IMPL(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zcopy_blis_impl_(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZCOPY_BLIS_IMPL_(const f77_int *n, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS void ZAXPY_BLIS_IMPL(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void zaxpy_blis_impl_(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);

BLIS_EXPORT_BLIS void ZAXPY_BLIS_IMPL_(const f77_int *n, const dcomplex *za, const dcomplex *zx, const f77_int *incx, dcomplex *zy, const f77_int *incy);



BLIS_EXPORT_BLIS double DZASUM_BLIS_IMPL(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS double dzasum_blis_impl_(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS double DZASUM_BLIS_IMPL_(const f77_int *n, const dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int IZAMAX_BLIS_IMPL(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int izamax_blis_impl_(const f77_int *n, const dcomplex *zx, const f77_int *incx);

BLIS_EXPORT_BLIS f77_int IZAMAX_BLIS_IMPL_(const f77_int *n, const dcomplex *zx, const f77_int *incx);



BLIS_EXPORT_BLIS f77_int ICAMIN_BLIS_IMPL( const f77_int* n,  const scomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int icamin_blis_impl_( const f77_int* n,  const scomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int ICAMIN_BLIS_IMPL_( const f77_int* n,  const scomplex* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int IDAMIN_BLIS_IMPL( const f77_int* n,  const double* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int idamin_blis_impl_( const f77_int* n,  const double* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int IDAMIN_BLIS_IMPL_( const f77_int* n,  const double* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int ISAMIN_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int isamin_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int ISAMIN_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx);



BLIS_EXPORT_BLIS f77_int IZAMIN_BLIS_IMPL( const f77_int* n,  const dcomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int izamin_blis_impl_( const f77_int* n,  const dcomplex* x,  const f77_int* incx);

BLIS_EXPORT_BLIS f77_int IZAMIN_BLIS_IMPL_( const f77_int* n,  const dcomplex* x,  const f77_int* incx);



//Level 2 APIs
BLIS_EXPORT_BLIS void SGEMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sgemv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SGEMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SGBMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sgbmv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SGBMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSYMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ssymv_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSYMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSBMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ssbmv_blis_impl_(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSBMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void SSPMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void sspmv_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void SSPMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *ap, const float  *x, const f77_int *incx, const float  *beta, float  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void STRMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void strmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STRMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STBMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stbmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STBMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STPMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stpmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STPMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STRSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void strsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STRSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STBSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stbsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STBSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const float  *a, const f77_int *lda, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void STPSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void stpsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void STPSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const float  *ap, float  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void SGER_BLIS_IMPL(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void sger_blis_impl_(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SGER_BLIS_IMPL_(const f77_int *m, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSYR_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ssyr_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SSYR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSPR_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);

BLIS_EXPORT_BLIS void sspr_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);

BLIS_EXPORT_BLIS void SSPR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, float  *ap);



BLIS_EXPORT_BLIS void SSYR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ssyr2_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void SSYR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void SSPR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);

BLIS_EXPORT_BLIS void sspr2_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);

BLIS_EXPORT_BLIS void SSPR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const float  *x, const f77_int *incx, const float  *y, const f77_int *incy, float  *ap);



BLIS_EXPORT_BLIS void DGEMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dgemv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DGEMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DGBMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dgbmv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DGBMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSYMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dsymv_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSYMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSBMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dsbmv_blis_impl_(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSBMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DSPMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void dspmv_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);

BLIS_EXPORT_BLIS void DSPMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *ap, const double *x, const f77_int *incx, const double *beta, double *y, const f77_int *incy);



BLIS_EXPORT_BLIS void DTRMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtrmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTRMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTBMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtbmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTBMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTPMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtpmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTPMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTRSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtrsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTRSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTBSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtbsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTBSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const double *a, const f77_int *lda, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DTPSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void dtpsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);

BLIS_EXPORT_BLIS void DTPSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const double *ap, double *x, const f77_int *incx);



BLIS_EXPORT_BLIS void DGER_BLIS_IMPL(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dger_blis_impl_(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DGER_BLIS_IMPL_(const f77_int *m, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSYR_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dsyr_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DSYR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSPR_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);

BLIS_EXPORT_BLIS void dspr_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);

BLIS_EXPORT_BLIS void DSPR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, double *ap);



BLIS_EXPORT_BLIS void DSYR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void dsyr2_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);

BLIS_EXPORT_BLIS void DSYR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *a, const f77_int *lda);



BLIS_EXPORT_BLIS void DSPR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);

BLIS_EXPORT_BLIS void dspr2_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);

BLIS_EXPORT_BLIS void DSPR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const double *x, const f77_int *incx, const double *y, const f77_int *incy, double *ap);



BLIS_EXPORT_BLIS void CGEMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void cgemv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CGEMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CGBMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void cgbmv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CGBMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHEMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chemv_blis_impl_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHEMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHBMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chbmv_blis_impl_(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHBMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a,const f77_int *lda, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CHPMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void chpmv_blis_impl_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);

BLIS_EXPORT_BLIS void CHPMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *ap, const scomplex  *x, const f77_int *incx, const scomplex  *beta, scomplex  *y, const f77_int *incy);



BLIS_EXPORT_BLIS void CTRMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctrmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTRMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTBMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctbmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTBMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTPMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctpmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTPMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTRSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const  scomplex *a, const f77_int *lda, scomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctrsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const  scomplex *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTRSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTBSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctbsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTBSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const scomplex  *a, const f77_int *lda, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CTPSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ctpsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);

BLIS_EXPORT_BLIS void CTPSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const scomplex  *ap, scomplex  *x, const f77_int *incx);



BLIS_EXPORT_BLIS void CGERC_BLIS_IMPL(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cgerc_blis_impl_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CGERC_BLIS_IMPL_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CGERU_BLIS_IMPL(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cgeru_blis_impl_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CGERU_BLIS_IMPL_(const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHER_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cher_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CHER_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHPR_BLIS_IMPL(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);

BLIS_EXPORT_BLIS void chpr_blis_impl_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);

BLIS_EXPORT_BLIS void CHPR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const float  *alpha, const scomplex  *x, const f77_int *incx, scomplex  *ap);



BLIS_EXPORT_BLIS void CHER2_BLIS_IMPL(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void cher2_blis_impl_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);

BLIS_EXPORT_BLIS void CHER2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *a, const f77_int *lda);



BLIS_EXPORT_BLIS void CHPR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);

BLIS_EXPORT_BLIS void chpr2_blis_impl_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);

BLIS_EXPORT_BLIS void CHPR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const scomplex  *alpha, const scomplex  *x, const f77_int *incx, const scomplex  *y, const f77_int *incy, scomplex  *ap);



BLIS_EXPORT_BLIS void ZGEMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zgemv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZGEMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZGBMV_BLIS_IMPL(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zgbmv_blis_impl_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZGBMV_BLIS_IMPL_(const char   *trans, const f77_int *m, const f77_int *n, const f77_int *kl, const f77_int *ku, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHEMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhemv_blis_impl_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHEMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHBMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhbmv_blis_impl_(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHBMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZHPMV_BLIS_IMPL(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void zhpmv_blis_impl_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);

BLIS_EXPORT_BLIS void ZHPMV_BLIS_IMPL_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *ap, const dcomplex *x, const f77_int *incx, const dcomplex *beta, dcomplex *y, const f77_int *incy);



BLIS_EXPORT_BLIS void ZTRMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztrmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTRMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTBMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztbmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTBMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTPMV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztpmv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTPMV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTRSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztrsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTRSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTBSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztbsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTBSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const f77_int *k, const dcomplex *a, const f77_int *lda, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZTPSV_BLIS_IMPL(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ztpsv_blis_impl_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);

BLIS_EXPORT_BLIS void ZTPSV_BLIS_IMPL_(const char   *uplo, const char   *trans, const char   *diag, const f77_int *n, const dcomplex *ap, dcomplex *x, const f77_int *incx);



BLIS_EXPORT_BLIS void ZGERU_BLIS_IMPL(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zgeru_blis_impl_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZGERU_BLIS_IMPL_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZGERC_BLIS_IMPL(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zgerc_blis_impl_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZGERC_BLIS_IMPL_(const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHER_BLIS_IMPL(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zher_blis_impl_(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZHER_BLIS_IMPL_(const char   *uplo, const f77_int *n, const double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHPR_BLIS_IMPL(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);

BLIS_EXPORT_BLIS void zhpr_blis_impl_(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);

BLIS_EXPORT_BLIS void ZHPR_BLIS_IMPL_(const char   *uplo, const f77_int *n, const bla_double *alpha, const dcomplex *x, const f77_int *incx, dcomplex *ap);



BLIS_EXPORT_BLIS void ZHER2_BLIS_IMPL(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void zher2_blis_impl_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);

BLIS_EXPORT_BLIS void ZHER2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *a, const f77_int *lda);



BLIS_EXPORT_BLIS void ZHPR2_BLIS_IMPL(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);

BLIS_EXPORT_BLIS void zhpr2_blis_impl_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);

BLIS_EXPORT_BLIS void ZHPR2_BLIS_IMPL_(const char   *uplo, const f77_int *n, const dcomplex *alpha, const dcomplex *x, const f77_int *incx, const dcomplex *y, const f77_int *incy, dcomplex *ap);



//Level 3 APIs
BLIS_EXPORT_BLIS void SGEMM_BLIS_IMPL(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void sgemm_blis_impl_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SGEMM_BLIS_IMPL_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssymm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYRK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssyrk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYRK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void SSYR2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ssyr2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void SSYR2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const float  *a, const f77_int *lda, const float  *b, const f77_int *ldb, const float  *beta, float  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void STRMM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void strmm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void STRMM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void STRSM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void strsm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void STRSM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const float  *alpha, const float  *a, const f77_int *lda, float  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void DGEMM_BLIS_IMPL(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dgemm_blis_impl_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DGEMM_BLIS_IMPL_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DZGEMM_BLIS_IMPL( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc );

BLIS_EXPORT_BLIS void dzgemm_blis_impl_( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc );

BLIS_EXPORT_BLIS void DZGEMM_BLIS_IMPL_( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc );



BLIS_EXPORT_BLIS void DSYMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsymm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DSYRK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsyrk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYRK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DSYR2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void dsyr2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void DSYR2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const double *a, const f77_int *lda, const double *b, const f77_int *ldb, const double *beta, double *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void DTRMM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void dtrmm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void DTRMM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void DTRSM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void dtrsm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void DTRSM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const double *alpha, const double *a, const f77_int *lda, double *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void CGEMM_BLIS_IMPL(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cgemm_blis_impl_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CGEMM_BLIS_IMPL_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csymm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHEMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void chemm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHEMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYRK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csyrk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYRK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHERK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cherk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHERK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const float  *alpha, const scomplex  *a, const f77_int *lda, const float  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CSYR2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void csyr2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CSYR2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const scomplex  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CHER2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void cher2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void CHER2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, const scomplex  *b, const f77_int *ldb, const float  *beta, scomplex  *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void CTRMM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ctrmm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void CTRMM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void CTRSM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ctrsm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void CTRSM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const scomplex  *alpha, const scomplex  *a, const f77_int *lda, scomplex  *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void ZGEMM_BLIS_IMPL(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zgemm_blis_impl_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZGEMM_BLIS_IMPL_(const char   *transa, const char   *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsymm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHEMM_BLIS_IMPL(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zhemm_blis_impl_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHEMM_BLIS_IMPL_(const char   *side, const char   *uplo, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYRK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsyrk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYRK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHERK_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zherk_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHERK_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const double *alpha, const dcomplex *a, const f77_int *lda, const double *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZSYR2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zsyr2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZSYR2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZHER2K_BLIS_IMPL(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void zher2k_blis_impl_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);

BLIS_EXPORT_BLIS void ZHER2K_BLIS_IMPL_(const char   *uplo, const char   *trans, const f77_int *n, const f77_int *k, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const double *beta, dcomplex *c, const f77_int *ldc);



BLIS_EXPORT_BLIS void ZTRMM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ztrmm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ZTRMM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);



BLIS_EXPORT_BLIS void ZTRSM_BLIS_IMPL(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ztrsm_blis_impl_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);

BLIS_EXPORT_BLIS void ZTRSM_BLIS_IMPL_(const char   *side, const char   *uplo, const char   *transa, const char   *diag, const f77_int *m, const f77_int *n, const dcomplex *alpha, const dcomplex *a, const f77_int *lda, dcomplex *b, const f77_int *ldb);



// Miscellaneous APIs

#ifdef BLIS_ENABLE_CBLAS

BLIS_EXPORT_BLIS void CDOTCSUB_BLIS_IMPL( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void cdotcsub_blis_impl_( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void CDOTCSUB_BLIS_IMPL_( const f77_int* n,  const scomplex* x, const f77_int* incx,  const scomplex* y,  const f77_int* incy,  scomplex* rval);



BLIS_EXPORT_BLIS void CDOTUSUB_BLIS_IMPL( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void cdotusub_blis_impl_( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);

BLIS_EXPORT_BLIS void CDOTUSUB_BLIS_IMPL_( const f77_int* n,  const scomplex* x, const f77_int* incxy,  const scomplex* y,  const f77_int* incy,  scomplex* rval);



BLIS_EXPORT_BLIS void DASUMSUB_BLIS_IMPL(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dasumsub_blis_impl_(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DASUMSUB_BLIS_IMPL_(const f77_int* n,  const double* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void DDOTSUB_BLIS_IMPL(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);

BLIS_EXPORT_BLIS void ddotsub_blis_impl_(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);

BLIS_EXPORT_BLIS void DDOTSUB_BLIS_IMPL_(const f77_int* n,  const double* x,  const f77_int* incx,  const double* y,  const f77_int* incy,  double* rval);



BLIS_EXPORT_BLIS void DNRM2SUB_BLIS_IMPL(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);

BLIS_EXPORT_BLIS void dnrm2sub_blis_impl_(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);

BLIS_EXPORT_BLIS void DNRM2SUB_BLIS_IMPL_(const f77_int* n,  const double* x,  const f77_int* incx,  double *rval);



BLIS_EXPORT_BLIS void DZASUMSUB_BLIS_IMPL(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dzasumsub_blis_impl_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DZASUMSUB_BLIS_IMPL_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void DZNRM2SUB_BLIS_IMPL(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void dznrm2sub_blis_impl_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);

BLIS_EXPORT_BLIS void DZNRM2SUB_BLIS_IMPL_(const f77_int* n,  const dcomplex* x,  const f77_int* incx,  double* rval);



BLIS_EXPORT_BLIS void ICAMAXSUB_BLIS_IMPL(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void icamaxsub_blis_impl_(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ICAMAXSUB_BLIS_IMPL_(const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ICAMINSUB_BLIS_IMPL( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void icaminsub_blis_impl_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ICAMINSUB_BLIS_IMPL_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IDAMAXSUB_BLIS_IMPL( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void idamaxsub_blis_impl_( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IDAMAXSUB_BLIS_IMPL_( const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IDAMINSUB_BLIS_IMPL(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void idaminsub_blis_impl_(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IDAMINSUB_BLIS_IMPL_(const f77_int* n,  const double* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ISAMAXSUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void isamaxsub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ISAMAXSUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void ISAMINSUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void isaminsub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void ISAMINSUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IZAMINSUB_BLIS_IMPL( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void izaminsub_blis_impl_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IZAMINSUB_BLIS_IMPL_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void IZAMAXSUB_BLIS_IMPL( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void izamaxsub_blis_impl_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);

BLIS_EXPORT_BLIS void IZAMAXSUB_BLIS_IMPL_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  f77_int* rval);



BLIS_EXPORT_BLIS void SASUMSUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void sasumsub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SASUMSUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SCASUMSUB_BLIS_IMPL( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void scasumsub_blis_impl_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SCASUMSUB_BLIS_IMPL_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SCNRM2SUB_BLIS_IMPL( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void scnrm2sub_blis_impl_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);

BLIS_EXPORT_BLIS void SCNRM2SUB_BLIS_IMPL_( const f77_int* n,  const scomplex* x,  const f77_int* incx,  float* rval);



BLIS_EXPORT_BLIS void SDOTSUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);

BLIS_EXPORT_BLIS void sdotsub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);

BLIS_EXPORT_BLIS void SDOTSUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* rval);



BLIS_EXPORT_BLIS void SNRM2SUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);

BLIS_EXPORT_BLIS void snrm2sub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);

BLIS_EXPORT_BLIS void SNRM2SUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  float *rval);



BLIS_EXPORT_BLIS void ZDOTCSUB_BLIS_IMPL( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void zdotcsub_blis_impl_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void ZDOTCSUB_BLIS_IMPL_( const f77_int* n,  const dcomplex* x,  const f77_int* incx,  const dcomplex* y,  const f77_int* incy,  dcomplex* rval);



BLIS_EXPORT_BLIS void ZDOTUSUB_BLIS_IMPL( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void zdotusub_blis_impl_( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);

BLIS_EXPORT_BLIS void ZDOTUSUB_BLIS_IMPL_( const f77_int* n,  const dcomplex* x,  const f77_int* incx, const dcomplex* y,  const f77_int* incy,  dcomplex* rval);



BLIS_EXPORT_BLIS void SDSDOTSUB_BLIS_IMPL( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);

BLIS_EXPORT_BLIS void sdsdotsub_blis_impl_( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);

BLIS_EXPORT_BLIS void SDSDOTSUB_BLIS_IMPL_( const f77_int* n,  float* sb,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  float* dot);



BLIS_EXPORT_BLIS void DSDOTSUB_BLIS_IMPL( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);

BLIS_EXPORT_BLIS void dsdotsub_blis_impl_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);

BLIS_EXPORT_BLIS void DSDOTSUB_BLIS_IMPL_( const f77_int* n,  const float* x,  const f77_int* incx,  const float* y,  const f77_int* incy,  double* dot);

#endif // BLIS_ENABLE_CBLAS


BLIS_EXPORT_BLIS f77_int LSAME_BLIS_IMPL(const char   *ca, const char   *cb, const f77_int a, const f77_int b);

BLIS_EXPORT_BLIS f77_int lsame_blis_impl_(const char   *ca, const char   *cb, const f77_int a, const f77_int b);

BLIS_EXPORT_BLIS f77_int LSAME_BLIS_IMPL_(const char   *ca, const char   *cb, const f77_int a, const f77_int b);



BLIS_EXPORT_BLIS void XERBLA_BLIS_IMPL(const char   *srname, const f77_int *info, ftnlen n);

BLIS_EXPORT_BLIS void xerbla_blis_impl_(const char   *srname, const f77_int *info, ftnlen n);

BLIS_EXPORT_BLIS void XERBLA_BLIS_IMPL_(const char   *srname, const f77_int *info, ftnlen n);



//Auxiliary APIs
BLIS_EXPORT_BLIS double DCABS1_BLIS_IMPL(bla_dcomplex *z);

BLIS_EXPORT_BLIS double dcabs1_blis_impl_(bla_dcomplex *z);

BLIS_EXPORT_BLIS double DCABS1_BLIS_IMPL_(bla_dcomplex *z);



BLIS_EXPORT_BLIS float SCABS1_BLIS_IMPL(bla_scomplex* z);

BLIS_EXPORT_BLIS float scabs1_blis_impl_(bla_scomplex* z);

BLIS_EXPORT_BLIS float SCABS1_BLIS_IMPL_(bla_scomplex* z);



//BLAS Extension APIs
BLIS_EXPORT_BLIS void CAXPBY_BLIS_IMPL( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void caxpby_blis_impl_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void CAXPBY_BLIS_IMPL_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void CGEMM3M_BLIS_IMPL( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void cgemm3m_blis_impl_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void CGEMM3M_BLIS_IMPL_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void CGEMM_BATCH_BLIS_IMPL( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void cgemm_batch_blis_impl_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void CGEMM_BATCH_BLIS_IMPL_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const scomplex* alpha_array,  const scomplex** a_array,  const  f77_int *lda_array,  const scomplex** b_array,  const f77_int *ldb_array,  const scomplex* beta_array,  scomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void CGEMMT_BLIS_IMPL( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void cgemmt_blis_impl_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void CGEMMT_BLIS_IMPL_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  scomplex* alpha,  const scomplex* a,  const f77_int* lda,  const scomplex* b,  const f77_int* ldb,  const scomplex* beta,  scomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void DAXPBY_BLIS_IMPL(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void daxpby_blis_impl_(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void DAXPBY_BLIS_IMPL_(const f77_int* n,  const double* alpha,  const double *x,  const f77_int* incx,  const double* beta,  double *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void DGEMM_BATCH_BLIS_IMPL( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void dgemm_batch_blis_impl_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void DGEMM_BATCH_BLIS_IMPL_( const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const double* alpha_array,  const double** a_array,  const  f77_int *lda_array,  const double** b_array,  const f77_int *ldb_array,  const double* beta_array,  double** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS f77_int DGEMM_PACK_GET_SIZE_BLIS_IMPL(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);

BLIS_EXPORT_BLIS f77_int dgemm_pack_get_size_blis_impl_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);

BLIS_EXPORT_BLIS f77_int DGEMM_PACK_GET_SIZE_BLIS_IMPL_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);



BLIS_EXPORT_BLIS void DGEMM_PACK_BLIS_IMPL( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest );

BLIS_EXPORT_BLIS void dgemm_pack_blis_impl_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest );

BLIS_EXPORT_BLIS void DGEMM_PACK_BLIS_IMPL_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest );



BLIS_EXPORT_BLIS void DGEMM_COMPUTE_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc );

BLIS_EXPORT_BLIS void dgemm_compute_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc );

BLIS_EXPORT_BLIS void DGEMM_COMPUTE_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc );



BLIS_EXPORT_BLIS void DGEMMT_BLIS_IMPL( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void dgemmt_blis_impl_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void DGEMMT_BLIS_IMPL_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  double* alpha,  const double* a,  const f77_int* lda,  const double* b,  const f77_int* ldb,  const double* beta,  double* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void SAXPBY_BLIS_IMPL( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void saxpby_blis_impl_( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void SAXPBY_BLIS_IMPL_( const f77_int* n,  const float* alpha,  const float *x,  const f77_int* incx,  const float* beta,  float *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void SGEMM_BATCH_BLIS_IMPL(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void sgemm_batch_blis_impl_(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void SGEMM_BATCH_BLIS_IMPL_(const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const float* alpha_array,  const float** a_array,  const  f77_int *lda_array,  const float** b_array,  const f77_int *ldb_array,  const float* beta_array,  float** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS f77_int SGEMM_PACK_GET_SIZE_BLIS_IMPL(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);

BLIS_EXPORT_BLIS f77_int sgemm_pack_get_size_blis_impl_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);

BLIS_EXPORT_BLIS f77_int SGEMM_PACK_GET_SIZE_BLIS_IMPL_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk);



BLIS_EXPORT_BLIS void SGEMM_PACK_BLIS_IMPL( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest );

BLIS_EXPORT_BLIS void sgemm_pack_blis_impl_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest );

BLIS_EXPORT_BLIS void SGEMM_PACK_BLIS_IMPL_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest );



BLIS_EXPORT_BLIS void SGEMM_COMPUTE_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc );

BLIS_EXPORT_BLIS void sgemm_compute_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc );

BLIS_EXPORT_BLIS void SGEMM_COMPUTE_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc );



BLIS_EXPORT_BLIS void SGEMMT_BLIS_IMPL( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void sgemmt_blis_impl_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void SGEMMT_BLIS_IMPL_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  float* alpha,  const float* a,  const f77_int* lda,  const float* b,  const f77_int* ldb,  const float* beta,  float* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void ZAXPBY_BLIS_IMPL( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void zaxpby_blis_impl_( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);

BLIS_EXPORT_BLIS void ZAXPBY_BLIS_IMPL_( const f77_int* n,  const dcomplex* alpha,  const dcomplex *x,  const f77_int* incx,  const dcomplex* beta,  dcomplex *y,  const f77_int* incy);



BLIS_EXPORT_BLIS void ZGEMM3M_BLIS_IMPL( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void zgemm3m_blis_impl_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void ZGEMM3M_BLIS_IMPL_( const f77_char* transa,  const f77_char* transb,  const f77_int* m,  const f77_int* n,  const f77_int* k,  const dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);



BLIS_EXPORT_BLIS void ZGEMM_BATCH_BLIS_IMPL(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void zgemm_batch_blis_impl_(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);

BLIS_EXPORT_BLIS void ZGEMM_BATCH_BLIS_IMPL_(  const f77_char* transa_array,  const f77_char* transb_array, const f77_int *m_array,  const f77_int *n_array,  const f77_int *k_array, const dcomplex* alpha_array,  const dcomplex** a_array,  const  f77_int *lda_array,  const dcomplex** b_array,  const f77_int *ldb_array,  const dcomplex* beta_array,  dcomplex** c_array,  const f77_int *ldc_array,  const f77_int* group_count,  const f77_int *group_size);



BLIS_EXPORT_BLIS void ZGEMMT_BLIS_IMPL( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void zgemmt_blis_impl_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

BLIS_EXPORT_BLIS void ZGEMMT_BLIS_IMPL_( const f77_char* uploc,  const f77_char* transa,  const f77_char* transb,  const f77_int* n,  const f77_int* k,  const  dcomplex* alpha,  const dcomplex* a,  const f77_int* lda,  const dcomplex* b,  const f77_int* ldb,  const dcomplex* beta,  dcomplex* c,  const f77_int* ldc);

#endif
#endif

#endif // BLI_UTIL_API_WRAP_BLIS_IMPL_H_
