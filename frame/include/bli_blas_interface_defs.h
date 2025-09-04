/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_BLAS_INTERFACE_DEFS_H
#define BLIS_BLAS_INTERFACE_DEFS_H

#ifdef BLIS_ENABLE_NO_UNDERSCORE_API
#ifdef BLIS_ENABLE_BLAS

#define isamax_ isamax
#define idamax_ idamax
#define icamax_ icamax
#define izamax_ izamax
#define sasum_  sasum
#define dasum_  dasum
#define scasum_ scasum
#define dzasum_ dzasum
#define saxpy_  saxpy
#define daxpy_  daxpy
#define caxpy_  caxpy
#define zaxpy_  zaxpy
#define scopy_  scopy
#define dcopy_  dcopy
#define ccopy_  ccopy
#define zcopy_  zcopy
#define sdot_   sdot
#define ddot_   ddot
#define cdotc_  cdotc
#define zdotc_  zdotc
#define cdotu_  cdotu
#define zdotu_  zdotu
#define snrm2_  snrm2
#define dnrm2_  dnrm2
#define scnrm2_ scnrm2
#define dznrm2_ dznrm2
#define sscal_  sscal
#define dscal_  dscal
#define cscal_  cscal
#define csscal_ csscal
#define zscal_  zscal
#define zdscal_ zdscal
#define sswap_  sswap
#define dswap_  dswap
#define cswap_  cswap
#define zswap_  zswap
#define sgemv_  sgemv
#define dgemv_  dgemv
#define cgemv_  cgemv
#define zgemv_  zgemv
#define sger_   sger
#define dger_   dger
#define cgerc_  cgerc
#define cgeru_  cgeru
#define zgerc_  zgerc
#define zgeru_  zgeru
#define chemv_  chemv
#define zhemv_  zhemv
#define cher_   cher
#define zher_   zher
#define cher2_  cher2
#define zher2_  zher2
#define ssymv_  ssymv
#define dsymv_  dsymv
#define csymm_  csymm
#define zsymm_  zsymm
#define ssyr_   ssyr
#define dsyr_   dsyr
#define csyrk_  csyrk
#define csyrk_  csyrk
#define zsyrk_  zsyrk
#define ssyr2_  ssyr2
#define dsyr2_  dsyr2
#define csyr2k_ csyr2k
#define zsyr2k_ zsyr2k
#define strmv_  strmv
#define dtrmv_  dtrmv
#define ctrmv_  ctrmv
#define ztrmv_  ztrmv
#define strsv_  strsv
#define dtrsv_  dtrsv
#define ctrsv_  ctrsv
#define ztrsv_  ztrsv
#define sgemm_  sgemm
#define dgemm_  dgemm
#define cgemm_  cgemm
#define zgemm_  zgemm
#define chemm_  chemm
#define zhemm_  zhemm
#define dgemmt_ dgemmt
#define sgemmt_ sgemmt
#define zgemmt_ zgemmt
#define cgemmt_ cgemmt
#define sgemm_batch_ sgemm_batch
#define dgemm_batch_ dgemm_batch
#define cgemm_batch_ cgemm_batch
#define zgemm_batch_ zgemm_batch
#define sgemm_compute_ sgemm_compute
#define dgemm_compute_ dgemm_compute
#define sgemm_pack_get_size_ sgemm_pack_get_size
#define dgemm_pack_get_size_ dgemm_pack_get_size
#define sgemm_pack_ sgemm_pack
#define dgemm_pack_ dgemm_pack
#define saxpby_ saxpby
#define daxpby_ daxpby
#define caxpby_ caxpby
#define zaxpby_ zaxpby
#define cher2k_ cher2k
#define zher2k_ zher2k
#define cherk_  cherk
#define zherk_  zherk
#define ssymm_  ssymm
#define dsymm_  dsymm
#define ssyr2k_ ssyr2k
#define dsyr2k_ dsyr2k
#define ssyrk_  ssyrk
#define dsyrk_  dsyrk
#define strmm_  strmm
#define dtrmm_  dtrmm
#define ctrmm_  ctrmm
#define ztrmm_  ztrmm
#define strsm_  strsm
#define dtrsm_  dtrsm
#define ctrsm_  ctrsm
#define ztrsm_  ztrsm
#define lsame_  lsame

#define cimatcopy_    cimatcopy
#define comatadd_     comatadd
#define comatcopy2_   comatcopy2
#define comatcopy_    comatcopy
#define dimatcopy_    dimatcopy
#define domatadd_     domatadd
#define domatcopy2_   domatcopy2
#define domatcopy_    domatcopy
#define simatcopy_    simatcopy
#define somatadd_     somatadd
#define somatcopy2_   somatcopy2
#define somatcopy_    somatcopy
#define zimatcopy_    zimatcopy
#define zomatadd_     zomatadd
#define zomatcopy2_   zomatcopy2
#define zomatcopy_    zomatcopy

#endif // BLIS_ENABLE_BLAS
#endif // BLIS_ENABLE_NO_UNDERSCORE_API

#ifdef BLIS_ENABLE_UPPERCASE_API
#ifdef BLIS_ENABLE_BLAS

#define caxpby                    CAXPBY
#define caxpy                     CAXPY
#define ccopy                     CCOPY
#define cdotc                     CDOTC
#define cdotcsub                  CDOTCSUB
#define cdotu                     CDOTU
#define cdotusub                  CDOTUSUB
#define cgbmv                     CGBMV
#define cgemm                     CGEMM
#define cgemm3m                   CGEMM3M
#define cgemm_batch               CGEMM_BATCH
#define cgemmt                    CGEMMT
#define cgemv                     CGEMV
#define cgerc                     CGERC
#define cgeru                     CGERU
#define chbmv                     CHBMV
#define chemm                     CHEMM
#define chemv                     CHEMV
#define cher                      CHER
#define cher2                     CHER2
#define cher2k                    CHER2K
#define cherk                     CHERK
#define chpmv                     CHPMV
#define chpr                      CHPR
#define chpr2                     CHPR2
#define crotg                     CROTG
#define cscal                     CSCAL
#define csrot                     CSROT
#define csscal                    CSSCAL
#define cswap                     CSWAP
#define csymm                     CSYMM
#define csyr2k                    CSYR2K
#define csyrk                     CSYRK
#define ctbmv                     CTBMV
#define ctbsv                     CTBSV
#define ctpmv                     CTPMV
#define ctpsv                     CTPSV
#define ctrmm                     CTRMM
#define ctrmv                     CTRMV
#define ctrsm                     CTRSM
#define ctrsv                     CTRSV
#define dasum                     DASUM
#define dasumsub                  DASUMSUB
#define daxpby                    DAXPBY
#define daxpy                     DAXPY
#define dcabs1                    DCABS1
#define dcopy                     DCOPY
#define ddot                      DDOT
#define ddotsub                   DDOTSUB
#define dgbmv                     DGBMV
#define dgemm                     DGEMM
#define dgemm_batch               DGEMM_BATCH
#define dgemm_compute             DGEMM_COMPUTE
#define dgemm_pack_get_size       DGEMM_PACK_GET_SIZE
#define dgemm_pack                DGEMM_PACK
#define dgemmt                    DGEMMT
#define dgemv                     DGEMV
#define dger                      DGER
#define dnrm2                     DNRM2
#define dnrm2sub                  DNRM2SUB
#define drot                      DROT
#define drotg                     DROTG
#define drotm                     DROTM
#define drotmg                    DROTMG
#define dsbmv                     DSBMV
#define dscal                     DSCAL
#define dsdot                     DSDOT
#define dsdotsub                  DSDOTSUB
#define dspmv                     DSPMV
#define dspr                      DSPR
#define dspr2                     DSPR2
#define dswap                     DSWAP
#define dsymm                     DSYMM
#define dsymv                     DSYMV
#define dsyr                      DSYR
#define dsyr2                     DSYR2
#define dsyr2k                    DSYR2K
#define dsyrk                     DSYRK
#define dtbmv                     DTBMV
#define dtbsv                     DTBSV
#define dtpmv                     DTPMV
#define dtpsv                     DTPSV
#define dtrmm                     DTRMM
#define dtrmv                     DTRMV
#define dtrsm                     DTRSM
#define dtrsv                     DTRSV
#define dzasum                    DZASUM
#define dzasumsub                 DZASUMSUB
#define dznrm2                    DZNRM2
#define dznrm2sub                 DZNRM2SUB
#define icamax                    ICAMAX
#define icamaxsub                 ICAMAXSUB
#define icamin                    ICAMIN
#define icaminsub                 ICAMINSUB
#define idamax                    IDAMAX
#define idamaxsub                 IDAMAXSUB
#define idamin                    IDAMIN
#define idaminsub                 IDAMINSUB
#define isamax                    ISAMAX
#define isamaxsub                 ISAMAXSUB
#define isamin                    ISAMIN
#define isaminsub                 ISAMINSUB
#define izamax                    IZAMAX
#define izamaxsub                 IZAMAXSUB
#define izamin                    IZAMIN
#define izaminsub                 IZAMINSUB
#define lsame                     LSAME
#define sasum                     SASUM
#define sasumsub                  SASUMSUB
#define saxpby                    SAXPBY
#define saxpy                     SAXPY
#define scabs1                    SCABS1
#define scasum                    SCASUM
#define scasumsub                 SCASUMSUB
#define scnrm2                    SCNRM2
#define scnrm2sub                 SCNRM2SUB
#define scopy                     SCOPY
#define sdot                      SDOT
#define sdotsub                   SDOTSUB
#define sdsdot                    SDSDOT
#define sdsdotsub                 SDSDOTSUB
#define sgbmv                     SGBMV
#define sgemm                     SGEMM
#define sgemm_batch               SGEMM_BATCH
#define sgemm_compute             SGEMM_COMPUTE
#define sgemm_pack_get_size       SGEMM_PACK_GET_SIZE
#define sgemm_pack                SGEMM_PACK
#define sgemmt                    SGEMMT
#define sgemv                     SGEMV
#define sger                      SGER
#define snrm2                     SNRM2
#define snrm2sub                  SNRM2SUB
#define srot                      SROT
#define srotg                     SROTG
#define srotm                     SROTM
#define srotmg                    SROTMG
#define ssbmv                     SSBMV
#define sscal                     SSCAL
#define sspmv                     SSPMV
#define sspr                      SSPR
#define sspr2                     SSPR2
#define sswap                     SSWAP
#define ssymm                     SSYMM
#define ssymv                     SSYMV
#define ssyr                      SSYR
#define ssyr2                     SSYR2
#define ssyr2k                    SSYR2K
#define ssyrk                     SSYRK
#define stbmv                     STBMV
#define stbsv                     STBSV
#define stpmv                     STPMV
#define stpsv                     STPSV
#define strmm                     STRMM
#define strmv                     STRMV
#define strsm                     STRSM
#define strsv                     STRSV
#define xerbla                    XERBLA
#define zaxpby                    ZAXPBY
#define zaxpy                     ZAXPY
#define zcopy                     ZCOPY
#define zdotc                     ZDOTC
#define zdotcsub                  ZDOTCSUB
#define zdotu                     ZDOTU
#define zdotusub                  ZDOTUSUB
#define zdrot                     ZDROT
#define zdscal                    ZDSCAL
#define zgbmv                     ZGBMV
#define zgemm                     ZGEMM
#define zgemm3m                   ZGEMM3M
#define zgemm_batch               ZGEMM_BATCH
#define zgemmt                    ZGEMMT
#define zgemv                     ZGEMV
#define zgerc                     ZGERC
#define zgeru                     ZGERU
#define zhbmv                     ZHBMV
#define zhemm                     ZHEMM
#define zhemv                     ZHEMV
#define zher                      ZHER
#define zher2                     ZHER2
#define zher2k                    ZHER2K
#define zherk                     ZHERK
#define zhpmv                     ZHPMV
#define zhpr                      ZHPR
#define zhpr2                     ZHPR2
#define zrotg                     ZROTG
#define zscal                     ZSCAL
#define zswap                     ZSWAP
#define zsymm                     ZSYMM
#define zsyr2k                    ZSYR2K
#define zsyrk                     ZSYRK
#define ztbmv                     ZTBMV
#define ztbsv                     ZTBSV
#define ztpmv                     ZTPMV
#define ztpsv                     ZTPSV
#define ztrmm                     ZTRMM
#define ztrmv                     ZTRMV
#define ztrsm                     ZTRSM
#define ztrsv                     ZTRSV

#define cimatcopy                 CIMATCOPY
#define comatadd                  COMATADD
#define comatcopy2                COMATCOPY2
#define comatcopy                 COMATCOPY
#define dimatcopy                 DIMATCOPY
#define domatadd                  DOMATADD
#define domatcopy2                DOMATCOPY2
#define domatcopy                 DOMATCOPY
#define simatcopy                 SIMATCOPY
#define somatadd                  SOMATADD
#define somatcopy2                SOMATCOPY2
#define somatcopy                 SOMATCOPY
#define zimatcopy                 ZIMATCOPY
#define zomatadd                  ZOMATADD
#define zomatcopy2                ZOMATCOPY2
#define zomatcopy                 ZOMATCOPY

#endif // BLIS_ENABLE_BLAS
#endif // BLIS_ENABLE_UPPERCASE_API

#endif
