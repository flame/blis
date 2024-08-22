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

#define isamax_blis_impl_ isamax_blis_impl
#define idamax_blis_impl_ idamax_blis_impl
#define icamax_blis_impl_ icamax_blis_impl
#define izamax_blis_impl_ izamax_blis_impl
#define sasum_blis_impl_  sasum_blis_impl
#define dasum_blis_impl_  dasum_blis_impl
#define scasum_blis_impl_ scasum_blis_impl
#define dzasum_blis_impl_ dzasum_blis_impl
#define saxpy_blis_impl_  saxpy_blis_impl
#define daxpy_blis_impl_  daxpy_blis_impl
#define caxpy_blis_impl_  caxpy_blis_impl
#define zaxpy_blis_impl_  zaxpy_blis_impl
#define scopy_blis_impl_  scopy_blis_impl
#define dcopy_blis_impl_  dcopy_blis_impl
#define ccopy_blis_impl_  ccopy_blis_impl
#define zcopy_blis_impl_  zcopy_blis_impl
#define sdot_blis_impl_   sdot_blis_impl
#define ddot_blis_impl_   ddot_blis_impl
#define cdotc_blis_impl_  cdotc_blis_impl
#define zdotc_blis_impl_  zdotc_blis_impl
#define cdotu_blis_impl_  cdotu_blis_impl
#define zdotu_blis_impl_  zdotu_blis_impl
#define snrm2_blis_impl_  snrm2_blis_impl
#define dnrm2_blis_impl_  dnrm2_blis_impl
#define scnrm2_blis_impl_ scnrm2_blis_impl
#define dznrm2_blis_impl_ dznrm2_blis_impl
#define sscal_blis_impl_  sscal_blis_impl
#define dscal_blis_impl_  dscal_blis_impl
#define cscal_blis_impl_  cscal_blis_impl
#define csscal_blis_impl_ csscal_blis_impl
#define zscal_blis_impl_  zscal_blis_impl
#define zdscal_blis_impl_ zdscal_blis_impl
#define sswap_blis_impl_  sswap_blis_impl
#define dswap_blis_impl_  dswap_blis_impl
#define cswap_blis_impl_  cswap_blis_impl
#define zswap_blis_impl_  zswap_blis_impl
#define sgemv_blis_impl_  sgemv_blis_impl
#define dgemv_blis_impl_  dgemv_blis_impl
#define cgemv_blis_impl_  cgemv_blis_impl
#define zgemv_blis_impl_  zgemv_blis_impl
#define sger_blis_impl_   sger_blis_impl
#define dger_blis_impl_   dger_blis_impl
#define cgerc_blis_impl_  cgerc_blis_impl
#define cgeru_blis_impl_  cgeru_blis_impl
#define zgerc_blis_impl_  zgerc_blis_impl
#define zgeru_blis_impl_  zgeru_blis_impl
#define chemv_blis_impl_  chemv_blis_impl
#define zhemv_blis_impl_  zhemv_blis_impl
#define cher_blis_impl_   cher_blis_impl
#define zher_blis_impl_   zher_blis_impl
#define cher2_blis_impl_  cher2_blis_impl
#define zher2_blis_impl_  zher2_blis_impl
#define ssymv_blis_impl_  ssymv_blis_impl
#define dsymv_blis_impl_  dsymv_blis_impl
#define csymm_blis_impl_  csymm_blis_impl
#define zsymm_blis_impl_  zsymm_blis_impl
#define ssyr_blis_impl_   ssyr_blis_impl
#define dsyr_blis_impl_   dsyr_blis_impl
#define csyrk_blis_impl_  csyrk_blis_impl
#define csyrk_blis_impl_  csyrk_blis_impl
#define zsyrk_blis_impl_  zsyrk_blis_impl
#define ssyr2_blis_impl_  ssyr2_blis_impl
#define dsyr2_blis_impl_  dsyr2_blis_impl
#define csyr2k_blis_impl_ csyr2k_blis_impl
#define zsyr2k_blis_impl_ zsyr2k_blis_impl
#define strmv_blis_impl_  strmv_blis_impl
#define dtrmv_blis_impl_  dtrmv_blis_impl
#define ctrmv_blis_impl_  ctrmv_blis_impl
#define ztrmv_blis_impl_  ztrmv_blis_impl
#define strsv_blis_impl_  strsv_blis_impl
#define dtrsv_blis_impl_  dtrsv_blis_impl
#define ctrsv_blis_impl_  ctrsv_blis_impl
#define ztrsv_blis_impl_  ztrsv_blis_impl
#define sgemm_blis_impl_  sgemm_blis_impl
#define dgemm_blis_impl_  dgemm_blis_impl
#define cgemm_blis_impl_  cgemm_blis_impl
#define zgemm_blis_impl_  zgemm_blis_impl
#define chemm_blis_impl_  chemm_blis_impl
#define zhemm_blis_impl_  zhemm_blis_impl
#define dgemmt_blis_impl_ dgemmt_blis_impl
#define sgemmt_blis_impl_ sgemmt_blis_impl
#define zgemmt_blis_impl_ zgemmt_blis_impl
#define cgemmt_blis_impl_ cgemmt_blis_impl
#define sgemm_batch_blis_impl_ sgemm_batch_blis_impl
#define dgemm_batch_blis_impl_ dgemm_batch_blis_impl
#define cgemm_batch_blis_impl_ cgemm_batch_blis_impl
#define zgemm_batch_blis_impl_ zgemm_batch_blis_impl
#define sgemm_compute_blis_impl_ sgemm_compute_blis_impl
#define dgemm_compute_blis_impl_ dgemm_compute_blis_impl
#define sgemm_pack_get_size_blis_impl_ sgemm_pack_get_size_blis_impl
#define dgemm_pack_get_size_blis_impl_ dgemm_pack_get_size_blis_impl
#define sgemm_pack_blis_impl_ sgemm_pack_blis_impl
#define dgemm_pack_blis_impl_ dgemm_pack_blis_impl
#define saxpby_blis_impl_ saxpby_blis_impl
#define daxpby_blis_impl_ daxpby_blis_impl
#define caxpby_blis_impl_ caxpby_blis_impl
#define zaxpby_blis_impl_ zaxpby_blis_impl
#define cher2k_blis_impl_ cher2k_blis_impl
#define zher2k_blis_impl_ zher2k_blis_impl
#define cherk_blis_impl_  cherk_blis_impl
#define zherk_blis_impl_  zherk_blis_impl
#define ssymm_blis_impl_  ssymm_blis_impl
#define dsymm_blis_impl_  dsymm_blis_impl
#define ssyr2k_blis_impl_ ssyr2k_blis_impl
#define dsyr2k_blis_impl_ dsyr2k_blis_impl
#define ssyrk_blis_impl_  ssyrk_blis_impl
#define dsyrk_blis_impl_  dsyrk_blis_impl
#define strmm_blis_impl_  strmm_blis_impl
#define dtrmm_blis_impl_  dtrmm_blis_impl
#define ctrmm_blis_impl_  ctrmm_blis_impl
#define ztrmm_blis_impl_  ztrmm_blis_impl
#define strsm_blis_impl_  strsm_blis_impl
#define dtrsm_blis_impl_  dtrsm_blis_impl
#define ctrsm_blis_impl_  ctrsm_blis_impl
#define ztrsm_blis_impl_  ztrsm_blis_impl
#define lsame_blis_impl_  lsame_blis_impl

#endif // BLIS_ENABLE_BLAS
#endif // BLIS_ENABLE_NO_UNDERSCORE_API

#ifdef BLIS_ENABLE_UPPERCASE_API
#ifdef BLIS_ENABLE_BLAS

#define caxpby_blis_impl                    CAXPBY_BLIS_IMPL
#define caxpy_blis_impl                     CAXPY_BLIS_IMPL
#define ccopy_blis_impl                     CCOPY_BLIS_IMPL
#define cdotc_blis_impl                     CDOTC_BLIS_IMPL
#define cdotcsub_blis_impl                  CDOTCSUB_BLIS_IMPL
#define cdotu_blis_impl                     CDOTU_BLIS_IMPL
#define cdotusub_blis_impl                  CDOTUSUB_BLIS_IMPL
#define cgbmv_blis_impl                     CGBMV_BLIS_IMPL
#define cgemm_blis_impl                     CGEMM_BLIS_IMPL
#define cgemm3m_blis_impl                   CGEMM3M_BLIS_IMPL
#define cgemm_batch_blis_impl               CGEMM_BATCH_BLIS_IMPL
#define cgemmt_blis_impl                    CGEMMT_BLIS_IMPL
#define cgemv_blis_impl                     CGEMV_BLIS_IMPL
#define cgerc_blis_impl                     CGERC_BLIS_IMPL
#define cgeru_blis_impl                     CGERU_BLIS_IMPL
#define chbmv_blis_impl                     CHBMV_BLIS_IMPL
#define chemm_blis_impl                     CHEMM_BLIS_IMPL
#define chemv_blis_impl                     CHEMV_BLIS_IMPL
#define cher_blis_impl                      CHER_BLIS_IMPL
#define cher2_blis_impl                     CHER2_BLIS_IMPL
#define cher2k_blis_impl                    CHER2K_BLIS_IMPL
#define cherk_blis_impl                     CHERK_BLIS_IMPL
#define chpmv_blis_impl                     CHPMV_BLIS_IMPL
#define chpr_blis_impl                      CHPR_BLIS_IMPL
#define chpr2_blis_impl                     CHPR2_BLIS_IMPL
#define crotg_blis_impl                     CROTG_BLIS_IMPL
#define cscal_blis_impl                     CSCAL_BLIS_IMPL
#define csrot_blis_impl                     CSROT_BLIS_IMPL
#define csscal_blis_impl                    CSSCAL_BLIS_IMPL
#define cswap_blis_impl                     CSWAP_BLIS_IMPL
#define csymm_blis_impl                     CSYMM_BLIS_IMPL
#define csyr2k_blis_impl                    CSYR2K_BLIS_IMPL
#define csyrk_blis_impl                     CSYRK_BLIS_IMPL
#define ctbmv_blis_impl                     CTBMV_BLIS_IMPL
#define ctbsv_blis_impl                     CTBSV_BLIS_IMPL
#define ctpmv_blis_impl                     CTPMV_BLIS_IMPL
#define ctpsv_blis_impl                     CTPSV_BLIS_IMPL
#define ctrmm_blis_impl                     CTRMM_BLIS_IMPL
#define ctrmv_blis_impl                     CTRMV_BLIS_IMPL
#define ctrsm_blis_impl                     CTRSM_BLIS_IMPL
#define ctrsv_blis_impl                     CTRSV_BLIS_IMPL
#define dasum_blis_impl                     DASUM_BLIS_IMPL
#define dasumsub_blis_impl                  DASUMSUB_BLIS_IMPL
#define daxpby_blis_impl                    DAXPBY_BLIS_IMPL
#define daxpy_blis_impl                     DAXPY_BLIS_IMPL
#define dcabs1_blis_impl                    DCABS1_BLIS_IMPL
#define dcopy_blis_impl                     DCOPY_BLIS_IMPL
#define ddot_blis_impl                      DDOT_BLIS_IMPL
#define ddotsub_blis_impl                   DDOTSUB_BLIS_IMPL
#define dgbmv_blis_impl                     DGBMV_BLIS_IMPL
#define dgemm_blis_impl                     DGEMM_BLIS_IMPL
#define dgemm_batch_blis_impl               DGEMM_BATCH_BLIS_IMPL
#define dgemm_compute_blis_impl             DGEMM_COMPUTE_BLIS_IMPL
#define dgemm_pack_get_size_blis_impl       DGEMM_PACK_GET_SIZE_BLIS_IMPL
#define dgemm_pack_blis_impl                DGEMM_PACK_BLIS_IMPL
#define dgemmt_blis_impl                    DGEMMT_BLIS_IMPL
#define dgemv_blis_impl                     DGEMV_BLIS_IMPL
#define dger_blis_impl                      DGER_BLIS_IMPL
#define dnrm2_blis_impl                     DNRM2_BLIS_IMPL
#define dnrm2sub_blis_impl                  DNRM2SUB_BLIS_IMPL
#define drot_blis_impl                      DROT_BLIS_IMPL
#define drotg_blis_impl                     DROTG_BLIS_IMPL
#define drotm_blis_impl                     DROTM_BLIS_IMPL
#define drotmg_blis_impl                    DROTMG_BLIS_IMPL
#define dsbmv_blis_impl                     DSBMV_BLIS_IMPL
#define dscal_blis_impl                     DSCAL_BLIS_IMPL
#define dsdot_blis_impl                     DSDOT_BLIS_IMPL
#define dsdotsub_blis_impl                  DSDOTSUB_BLIS_IMPL
#define dspmv_blis_impl                     DSPMV_BLIS_IMPL
#define dspr_blis_impl                      DSPR_BLIS_IMPL
#define dspr2_blis_impl                     DSPR2_BLIS_IMPL
#define dswap_blis_impl                     DSWAP_BLIS_IMPL
#define dsymm_blis_impl                     DSYMM_BLIS_IMPL
#define dsymv_blis_impl                     DSYMV_BLIS_IMPL
#define dsyr_blis_impl                      DSYR_BLIS_IMPL
#define dsyr2_blis_impl                     DSYR2_BLIS_IMPL
#define dsyr2k_blis_impl                    DSYR2K_BLIS_IMPL
#define dsyrk_blis_impl                     DSYRK_BLIS_IMPL
#define dtbmv_blis_impl                     DTBMV_BLIS_IMPL
#define dtbsv_blis_impl                     DTBSV_BLIS_IMPL
#define dtpmv_blis_impl                     DTPMV_BLIS_IMPL
#define dtpsv_blis_impl                     DTPSV_BLIS_IMPL
#define dtrmm_blis_impl                     DTRMM_BLIS_IMPL
#define dtrmv_blis_impl                     DTRMV_BLIS_IMPL
#define dtrsm_blis_impl                     DTRSM_BLIS_IMPL
#define dtrsv_blis_impl                     DTRSV_BLIS_IMPL
#define dzasum_blis_impl                    DZASUM_BLIS_IMPL
#define dzasumsub_blis_impl                 DZASUMSUB_BLIS_IMPL
#define dznrm2_blis_impl                    DZNRM2_BLIS_IMPL
#define dznrm2sub_blis_impl                 DZNRM2SUB_BLIS_IMPL
#define icamax_blis_impl                    ICAMAX_BLIS_IMPL
#define icamaxsub_blis_impl                 ICAMAXSUB_BLIS_IMPL
#define icamin_blis_impl                    ICAMIN_BLIS_IMPL
#define icaminsub_blis_impl                 ICAMINSUB_BLIS_IMPL
#define idamax_blis_impl                    IDAMAX_BLIS_IMPL
#define idamaxsub_blis_impl                 IDAMAXSUB_BLIS_IMPL
#define idamin_blis_impl                    IDAMIN_BLIS_IMPL
#define idaminsub_blis_impl                 IDAMINSUB_BLIS_IMPL
#define isamax_blis_impl                    ISAMAX_BLIS_IMPL
#define isamaxsub_blis_impl                 ISAMAXSUB_BLIS_IMPL
#define isamin_blis_impl                    ISAMIN_BLIS_IMPL
#define isaminsub_blis_impl                 ISAMINSUB_BLIS_IMPL
#define izamax_blis_impl                    IZAMAX_BLIS_IMPL
#define izamaxsub_blis_impl                 IZAMAXSUB_BLIS_IMPL
#define izamin_blis_impl                    IZAMIN_BLIS_IMPL
#define izaminsub_blis_impl                 IZAMINSUB_BLIS_IMPL
#define lsame_blis_impl                     LSAME_BLIS_IMPL
#define sasum_blis_impl                     SASUM_BLIS_IMPL
#define sasumsub_blis_impl                  SASUMSUB_BLIS_IMPL
#define saxpby_blis_impl                    SAXPBY_BLIS_IMPL
#define saxpy_blis_impl                     SAXPY_BLIS_IMPL
#define scabs1_blis_impl                    SCABS1_BLIS_IMPL
#define scasum_blis_impl                    SCASUM_BLIS_IMPL
#define scasumsub_blis_impl                 SCASUMSUB_BLIS_IMPL
#define scnrm2_blis_impl                    SCNRM2_BLIS_IMPL
#define scnrm2sub_blis_impl                 SCNRM2SUB_BLIS_IMPL
#define scopy_blis_impl                     SCOPY_BLIS_IMPL
#define sdot_blis_impl                      SDOT_BLIS_IMPL
#define sdotsub_blis_impl                   SDOTSUB_BLIS_IMPL
#define sdsdot_blis_impl                    SDSDOT_BLIS_IMPL
#define sdsdotsub_blis_impl                 SDSDOTSUB_BLIS_IMPL
#define sgbmv_blis_impl                     SGBMV_BLIS_IMPL
#define sgemm_blis_impl                     SGEMM_BLIS_IMPL
#define sgemm_batch_blis_impl               SGEMM_BATCH_BLIS_IMPL
#define sgemm_compute_blis_impl             SGEMM_COMPUTE_BLIS_IMPL
#define sgemm_pack_get_size_blis_impl       SGEMM_PACK_GET_SIZE_BLIS_IMPL
#define sgemm_pack_blis_impl                SGEMM_PACK_BLIS_IMPL
#define sgemmt_blis_impl                    SGEMMT_BLIS_IMPL
#define sgemv_blis_impl                     SGEMV_BLIS_IMPL
#define sger_blis_impl                      SGER_BLIS_IMPL
#define snrm2_blis_impl                     SNRM2_BLIS_IMPL
#define snrm2sub_blis_impl                  SNRM2SUB_BLIS_IMPL
#define srot_blis_impl                      SROT_BLIS_IMPL
#define srotg_blis_impl                     SROTG_BLIS_IMPL
#define srotm_blis_impl                     SROTM_BLIS_IMPL
#define srotmg_blis_impl                    SROTMG_BLIS_IMPL
#define ssbmv_blis_impl                     SSBMV_BLIS_IMPL
#define sscal_blis_impl                     SSCAL_BLIS_IMPL
#define sspmv_blis_impl                     SSPMV_BLIS_IMPL
#define sspr_blis_impl                      SSPR_BLIS_IMPL
#define sspr2_blis_impl                     SSPR2_BLIS_IMPL
#define sswap_blis_impl                     SSWAP_BLIS_IMPL
#define ssymm_blis_impl                     SSYMM_BLIS_IMPL
#define ssymv_blis_impl                     SSYMV_BLIS_IMPL
#define ssyr_blis_impl                      SSYR_BLIS_IMPL
#define ssyr2_blis_impl                     SSYR2_BLIS_IMPL
#define ssyr2k_blis_impl                    SSYR2K_BLIS_IMPL
#define ssyrk_blis_impl                     SSYRK_BLIS_IMPL
#define stbmv_blis_impl                     STBMV_BLIS_IMPL
#define stbsv_blis_impl                     STBSV_BLIS_IMPL
#define stpmv_blis_impl                     STPMV_BLIS_IMPL
#define stpsv_blis_impl                     STPSV_BLIS_IMPL
#define strmm_blis_impl                     STRMM_BLIS_IMPL
#define strmv_blis_impl                     STRMV_BLIS_IMPL
#define strsm_blis_impl                     STRSM_BLIS_IMPL
#define strsv_blis_impl                     STRSV_BLIS_IMPL
#define xerbla_blis_impl                    XERBLA_BLIS_IMPL
#define zaxpby_blis_impl                    ZAXPBY_BLIS_IMPL
#define zaxpy_blis_impl                     ZAXPY_BLIS_IMPL
#define zcopy_blis_impl                     ZCOPY_BLIS_IMPL
#define zdotc_blis_impl                     ZDOTC_BLIS_IMPL
#define zdotcsub_blis_impl                  ZDOTCSUB_BLIS_IMPL
#define zdotu_blis_impl                     ZDOTU_BLIS_IMPL
#define zdotusub_blis_impl                  ZDOTUSUB_BLIS_IMPL
#define zdrot_blis_impl                     ZDROT_BLIS_IMPL
#define zdscal_blis_impl                    ZDSCAL_BLIS_IMPL
#define zgbmv_blis_impl                     ZGBMV_BLIS_IMPL
#define zgemm_blis_impl                     ZGEMM_BLIS_IMPL
#define zgemm3m_blis_impl                   ZGEMM3M_BLIS_IMPL
#define zgemm_batch_blis_impl               ZGEMM_BATCH_BLIS_IMPL
#define zgemmt_blis_impl                    ZGEMMT_BLIS_IMPL
#define zgemv_blis_impl                     ZGEMV_BLIS_IMPL
#define zgerc_blis_impl                     ZGERC_BLIS_IMPL
#define zgeru_blis_impl                     ZGERU_BLIS_IMPL
#define zhbmv_blis_impl                     ZHBMV_BLIS_IMPL
#define zhemm_blis_impl                     ZHEMM_BLIS_IMPL
#define zhemv_blis_impl                     ZHEMV_BLIS_IMPL
#define zher_blis_impl                      ZHER_BLIS_IMPL
#define zher2_blis_impl                     ZHER2_BLIS_IMPL
#define zher2k_blis_impl                    ZHER2K_BLIS_IMPL
#define zherk_blis_impl                     ZHERK_BLIS_IMPL
#define zhpmv_blis_impl                     ZHPMV_BLIS_IMPL
#define zhpr_blis_impl                      ZHPR_BLIS_IMPL
#define zhpr2_blis_impl                     ZHPR2_BLIS_IMPL
#define zrotg_blis_impl                     ZROTG_BLIS_IMPL
#define zscal_blis_impl                     ZSCAL_BLIS_IMPL
#define zswap_blis_impl                     ZSWAP_BLIS_IMPL
#define zsymm_blis_impl                     ZSYMM_BLIS_IMPL
#define zsyr2k_blis_impl                    ZSYR2K_BLIS_IMPL
#define zsyrk_blis_impl                     ZSYRK_BLIS_IMPL
#define ztbmv_blis_impl                     ZTBMV_BLIS_IMPL
#define ztbsv_blis_impl                     ZTBSV_BLIS_IMPL
#define ztpmv_blis_impl                     ZTPMV_BLIS_IMPL
#define ztpsv_blis_impl                     ZTPSV_BLIS_IMPL
#define ztrmm_blis_impl                     ZTRMM_BLIS_IMPL
#define ztrmv_blis_impl                     ZTRMV_BLIS_IMPL
#define ztrsm_blis_impl                     ZTRSM_BLIS_IMPL
#define ztrsv_blis_impl                     ZTRSV_BLIS_IMPL

#endif // BLIS_ENABLE_BLAS
#endif // BLIS_ENABLE_UPPERCASE_API

#endif
