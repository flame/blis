/*
 * cblas_f77.h
 * Written by Keita Teranishi
 *
 * Updated by Jeff Horner
 * Merged cblas_f77.h and cblas_fortran_header.h
 *
 * (Heavily hacked down from the original)
 *
 * Copyright (C) 2020 - 2021, Advanced Micro Devices, Inc. All rights reserved.
 *
 */

#ifndef CBLAS_F77_H
#define CBLAS_F77_H

#if defined(BLIS_ENABLE_NO_UNDERSCORE_API)
 /*
  * Level 1 BLAS
  */
#define F77_xerbla     xerbla
#define F77_srotg      srotg
#define F77_srotmg     srotmg
#define F77_srot       srot
#define F77_srotm      srotm
#define F77_drotg      drotg
#define F77_drotmg     drotmg
#define F77_drot       drot
#define F77_drotm      drotm
#define F77_sswap      sswap
#define F77_scopy      scopy
#define F77_saxpy      saxpy
#define F77_isamax_sub isamaxsub
#define F77_dswap      dswap
#define F77_dcopy      dcopy
#define F77_daxpy      daxpy
#define F77_idamax_sub idamaxsub
#define F77_cswap      cswap
#define F77_ccopy      ccopy
#define F77_caxpy      caxpy
#define F77_icamax_sub icamaxsub
#define F77_zswap      zswap
#define F77_zcopy      zcopy
#define F77_zaxpy      zaxpy
#define F77_zaxpby     zaxpby
#define F77_izamax_sub izamaxsub
#define F77_sdot_sub   sdotsub
#define F77_ddot_sub   ddotsub
#define F77_dsdot_sub  dsdotsub
#define F77_sscal      sscal
#define F77_dscal      dscal
#define F77_cscal      cscal
#define F77_zscal      zscal
#define F77_csscal     csscal
#define F77_zdscal     zdscal
#define F77_cdotu_sub  cdotusub
#define F77_cdotc_sub  cdotcsub
#define F77_zdotu_sub  zdotusub
#define F77_zdotc_sub  zdotcsub
#define F77_snrm2_sub  snrm2sub
#define F77_sasum_sub  sasumsub
#define F77_dnrm2_sub  dnrm2sub
#define F77_dasum_sub  dasumsub
#define F77_scnrm2_sub scnrm2sub
#define F77_scasum_sub scasumsub
#define F77_dznrm2_sub dznrm2sub
#define F77_dzasum_sub dzasumsub
#define F77_sdsdot_sub sdsdotsub
/*
* Level 2 BLAS
*/
#define F77_ssymv ssymv
#define F77_ssbmv ssbmv
#define F77_sspmv sspmv
#define F77_sger  sger
#define F77_ssyr  ssyr
#define F77_sspr  sspr
#define F77_ssyr2 ssyr2
#define F77_sspr2 sspr2
#define F77_dsymv dsymv
#define F77_dsbmv dsbmv
#define F77_dspmv dspmv
#define F77_dger  dger
#define F77_dsyr  dsyr
#define F77_dspr  dspr
#define F77_dsyr2 dsyr2
#define F77_dspr2 dspr2
#define F77_chemv chemv
#define F77_chbmv chbmv
#define F77_chpmv chpmv
#define F77_cgeru cgeru
#define F77_cgerc cgerc
#define F77_cher  cher
#define F77_chpr  chpr
#define F77_cher2 cher2
#define F77_chpr2 chpr2
#define F77_zhemv zhemv
#define F77_zhbmv zhbmv
#define F77_zhpmv zhpmv
#define F77_zgeru zgeru
#define F77_zgerc zgerc
#define F77_zher  zher
#define F77_zhpr  zhpr
#define F77_zher2 zher2
#define F77_zhpr2 zhpr2
#define F77_sgemv sgemv
#define F77_sgbmv sgbmv
#define F77_strmv strmv
#define F77_stbmv stbmv
#define F77_stpmv stpmv
#define F77_strsv strsv
#define F77_stbsv stbsv
#define F77_stpsv stpsv
#define F77_dgemv dgemv
#define F77_dgbmv dgbmv
#define F77_dtrmv dtrmv
#define F77_dtbmv dtbmv
#define F77_dtpmv dtpmv
#define F77_dtrsv dtrsv
#define F77_dtbsv dtbsv
#define F77_dtpsv dtpsv
#define F77_cgemv cgemv
#define F77_cgbmv cgbmv
#define F77_ctrmv ctrmv
#define F77_ctbmv ctbmv
#define F77_ctpmv ctpmv
#define F77_ctrsv ctrsv
#define F77_ctbsv ctbsv
#define F77_ctpsv ctpsv
#define F77_zgemv zgemv
#define F77_zgbmv zgbmv
#define F77_ztrmv ztrmv
#define F77_ztbmv ztbmv
#define F77_ztpmv ztpmv
#define F77_ztrsv ztrsv
#define F77_ztbsv ztbsv
#define F77_ztpsv ztpsv
/*
* Level 3 BLAS
*/
#define F77_chemm  chemm
#define F77_cherk  cherk
#define F77_cher2k cher2k
#define F77_zhemm  zhemm
#define F77_zherk  zherk
#define F77_zher2k zher2k
#define F77_sgemm  sgemm
#define F77_ssymm  ssymm
#define F77_ssyrk  ssyrk
#define F77_ssyr2k ssyr2k
#define F77_strmm  strmm
#define F77_strsm  strsm
#define F77_dgemm  dgemm
#define F77_dsymm  dsymm
#define F77_dsyrk  dsyrk
#define F77_dsyr2k dsyr2k
#define F77_dtrmm  dtrmm
#define F77_dtrsm  dtrsm
#define F77_cgemm  cgemm
#define F77_csymm  csymm
#define F77_csyrk  csyrk
#define F77_csyr2k csyr2k
#define F77_ctrmm  ctrmm
#define F77_ctrsm  ctrsm
#define F77_zgemm  zgemm
#define F77_zsymm  zsymm
#define F77_zsyrk  zsyrk
#define F77_zsyr2k zsyr2k
#define F77_ztrmm  ztrmm
#define F77_ztrsm  ztrsm
#define F77_dgemmt  dgemmt
#define F77_sgemmt  sgemmt
#define F77_cgemmt  cgemmt
#define F77_zgemmt  zgemmt

/*
* Aux Function
*/
#define F77_scabs1 scabs1
#define F77_dcabs1 dcabs1

/*
 * -- BLAS Extension APIs --
 */

#define F77_saxpby     saxpby
#define F77_daxpby     daxpby
#define F77_caxpby     caxpby
#define F77_zaxpby     zaxpby
#define F77_cgemm3m    cgemm3m
#define F77_zgemm3m    zgemm3m

#define F77_isamin_sub isaminsub
#define F77_idamin_sub idaminsub
#define F77_icamin_sub icaminsub
#define F77_izamin_sub izaminsub

// -- Batch APIs --
#define F77_sgemm_batch  sgemm_batch
#define F77_dgemm_batch  dgemm_batch
#define F77_cgemm_batch  cgemm_batch
#define F77_zgemm_batch  zgemm_batch

// (BLIS_ENABLE_NO_UNDERSCORE_API) ends
#else
/*
 * Level 1 BLAS
 */
#define F77_xerbla     xerbla_
#define F77_srotg      srotg_
#define F77_srotmg     srotmg_
#define F77_srot       srot_
#define F77_srotm      srotm_
#define F77_drotg      drotg_
#define F77_drotmg     drotmg_
#define F77_drot       drot_
#define F77_drotm      drotm_
#define F77_sswap      sswap_
#define F77_scopy      scopy_
#define F77_saxpy      saxpy_
#define F77_isamax_sub isamaxsub_
#define F77_dswap      dswap_
#define F77_dcopy      dcopy_
#define F77_daxpy      daxpy_
#define F77_idamax_sub idamaxsub_
#define F77_cswap      cswap_
#define F77_ccopy      ccopy_
#define F77_caxpy      caxpy_
#define F77_icamax_sub icamaxsub_
#define F77_zswap      zswap_
#define F77_zcopy      zcopy_
#define F77_zaxpy      zaxpy_
#define F77_zaxpby     zaxpby_
#define F77_izamax_sub izamaxsub_
#define F77_sdot_sub   sdotsub_
#define F77_ddot_sub   ddotsub_
#define F77_dsdot_sub  dsdotsub_
#define F77_sscal      sscal_
#define F77_dscal      dscal_
#define F77_cscal      cscal_
#define F77_zscal      zscal_
#define F77_csscal     csscal_
#define F77_zdscal     zdscal_
#define F77_cdotu_sub  cdotusub_
#define F77_cdotc_sub  cdotcsub_
#define F77_zdotu_sub  zdotusub_
#define F77_zdotc_sub  zdotcsub_
#define F77_snrm2_sub  snrm2sub_
#define F77_sasum_sub  sasumsub_
#define F77_dnrm2_sub  dnrm2sub_
#define F77_dasum_sub  dasumsub_
#define F77_scnrm2_sub scnrm2sub_
#define F77_scasum_sub scasumsub_
#define F77_dznrm2_sub dznrm2sub_
#define F77_dzasum_sub dzasumsub_
#define F77_sdsdot_sub sdsdotsub_
/*
* Level 2 BLAS
*/
#define F77_ssymv ssymv_
#define F77_ssbmv ssbmv_
#define F77_sspmv sspmv_
#define F77_sger  sger_
#define F77_ssyr  ssyr_
#define F77_sspr  sspr_
#define F77_ssyr2 ssyr2_
#define F77_sspr2 sspr2_
#define F77_dsymv dsymv_
#define F77_dsbmv dsbmv_
#define F77_dspmv dspmv_
#define F77_dger  dger_
#define F77_dsyr  dsyr_
#define F77_dspr  dspr_
#define F77_dsyr2 dsyr2_
#define F77_dspr2 dspr2_
#define F77_chemv chemv_
#define F77_chbmv chbmv_
#define F77_chpmv chpmv_
#define F77_cgeru cgeru_
#define F77_cgerc cgerc_
#define F77_cher  cher_
#define F77_chpr  chpr_
#define F77_cher2 cher2_
#define F77_chpr2 chpr2_
#define F77_zhemv zhemv_
#define F77_zhbmv zhbmv_
#define F77_zhpmv zhpmv_
#define F77_zgeru zgeru_
#define F77_zgerc zgerc_
#define F77_zher  zher_
#define F77_zhpr  zhpr_
#define F77_zher2 zher2_
#define F77_zhpr2 zhpr2_
#define F77_sgemv sgemv_
#define F77_sgbmv sgbmv_
#define F77_strmv strmv_
#define F77_stbmv stbmv_
#define F77_stpmv stpmv_
#define F77_strsv strsv_
#define F77_stbsv stbsv_
#define F77_stpsv stpsv_
#define F77_dgemv dgemv_
#define F77_dgbmv dgbmv_
#define F77_dtrmv dtrmv_
#define F77_dtbmv dtbmv_
#define F77_dtpmv dtpmv_
#define F77_dtrsv dtrsv_
#define F77_dtbsv dtbsv_
#define F77_dtpsv dtpsv_
#define F77_cgemv cgemv_
#define F77_cgbmv cgbmv_
#define F77_ctrmv ctrmv_
#define F77_ctbmv ctbmv_
#define F77_ctpmv ctpmv_
#define F77_ctrsv ctrsv_
#define F77_ctbsv ctbsv_
#define F77_ctpsv ctpsv_
#define F77_zgemv zgemv_
#define F77_zgbmv zgbmv_
#define F77_ztrmv ztrmv_
#define F77_ztbmv ztbmv_
#define F77_ztpmv ztpmv_
#define F77_ztrsv ztrsv_
#define F77_ztbsv ztbsv_
#define F77_ztpsv ztpsv_
/*
* Level 3 BLAS
*/
#define F77_chemm  chemm_
#define F77_cherk  cherk_
#define F77_cher2k cher2k_
#define F77_zhemm  zhemm_
#define F77_zherk  zherk_
#define F77_zher2k zher2k_
#define F77_sgemm  sgemm_
#define F77_ssymm  ssymm_
#define F77_ssyrk  ssyrk_
#define F77_ssyr2k ssyr2k_
#define F77_strmm  strmm_
#define F77_strsm  strsm_
#define F77_dgemm  dgemm_
#define F77_dsymm  dsymm_
#define F77_dsyrk  dsyrk_
#define F77_dsyr2k dsyr2k_
#define F77_dtrmm  dtrmm_
#define F77_dtrsm  dtrsm_
#define F77_cgemm  cgemm_
#define F77_csymm  csymm_
#define F77_csyrk  csyrk_
#define F77_csyr2k csyr2k_
#define F77_ctrmm  ctrmm_
#define F77_ctrsm  ctrsm_
#define F77_zgemm  zgemm_
#define F77_zsymm  zsymm_
#define F77_zsyrk  zsyrk_
#define F77_zsyr2k zsyr2k_
#define F77_ztrmm  ztrmm_
#define F77_ztrsm  ztrsm_
#define F77_dgemmt  dgemmt_
#define F77_sgemmt  sgemmt_
#define F77_cgemmt  cgemmt_
#define F77_zgemmt  zgemmt_

/*
* Aux Function
*/
#define F77_scabs1 scabs1_
#define F77_dcabs1 dcabs1_

/*
 * -- BLAS Extension APIs --
 */

#define F77_saxpby     saxpby_
#define F77_daxpby     daxpby_
#define F77_caxpby     caxpby_
#define F77_zaxpby     zaxpby_
#define F77_cgemm3m    cgemm3m_
#define F77_zgemm3m    zgemm3m_

#define F77_isamin_sub isaminsub_
#define F77_idamin_sub idaminsub_
#define F77_icamin_sub icaminsub_
#define F77_izamin_sub izaminsub_

// -- Batch APIs --
#define F77_sgemm_batch  sgemm_batch_
#define F77_dgemm_batch  dgemm_batch_
#define F77_cgemm_batch  cgemm_batch_
#define F77_zgemm_batch  zgemm_batch_
#endif

#endif /*  CBLAS_F77_H */