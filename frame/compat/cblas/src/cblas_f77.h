/*
 * cblas_f77.h
 * Written by Keita Teranishi
 *
 * Updated by Jeff Horner
 * Merged cblas_f77.h and cblas_fortran_header.h
 *
 * (Heavily hacked down from the original)
 *
 * Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.
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
#define F77_dzgemm  dzgemm

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

// -- Pack-Compute APIs --
#define F77_sgemm_pack_get_size  sgemm_pack_get_size_blis_impl
#define F77_dgemm_pack_get_size  dgemm_pack_get_size_blis_impl
#define F77_sgemm_pack  sgemm_pack_blis_impl
#define F77_dgemm_pack  dgemm_pack_blis_impl
#define F77_sgemm_compute  sgemm_compute_blis_impl
#define F77_dgemm_compute  dgemm_compute_blis_impl

// (BLIS_ENABLE_NO_UNDERSCORE_API) ends
#else
/*
 * Level 1 BLAS
 */
#define F77_xerbla     xerbla_
#define F77_srotg      srotg_blis_impl
#define F77_srotmg     srotmg_blis_impl
#define F77_srot       srot_blis_impl
#define F77_srotm      srotm_blis_impl
#define F77_drotg      drotg_blis_impl
#define F77_drotmg     drotmg_blis_impl
#define F77_drot       drot_blis_impl
#define F77_drotm      drotm_blis_impl
#define F77_sswap      sswap_blis_impl
#define F77_scopy      scopy_blis_impl
#define F77_saxpy      saxpy_blis_impl
#define F77_isamax_sub isamaxsub_blis_impl
#define F77_dswap      dswap_blis_impl
#define F77_dcopy      dcopy_blis_impl
#define F77_daxpy      daxpy_blis_impl
#define F77_idamax_sub idamaxsub_blis_impl
#define F77_cswap      cswap_blis_impl
#define F77_ccopy      ccopy_blis_impl
#define F77_caxpy      caxpy_blis_impl
#define F77_icamax_sub icamaxsub_blis_impl
#define F77_zswap      zswap_blis_impl
#define F77_zcopy      zcopy_blis_impl
#define F77_zaxpy      zaxpy_blis_impl
#define F77_izamax_sub izamaxsub_blis_impl
#define F77_sdot_sub   sdotsub_blis_impl
#define F77_ddot_sub   ddotsub_blis_impl
#define F77_dsdot_sub  dsdotsub_blis_impl
#define F77_sscal      sscal_blis_impl
#define F77_dscal      dscal_blis_impl
#define F77_cscal      cscal_blis_impl
#define F77_zscal      zscal_blis_impl
#define F77_csscal     csscal_blis_impl
#define F77_zdscal     zdscal_blis_impl
#define F77_cdotu_sub  cdotusub_blis_impl
#define F77_cdotc_sub  cdotcsub_blis_impl
#define F77_zdotu_sub  zdotusub_blis_impl
#define F77_zdotc_sub  zdotcsub_blis_impl
#define F77_snrm2_sub  snrm2sub_blis_impl
#define F77_sasum_sub  sasumsub_blis_impl
#define F77_dnrm2_sub  dnrm2sub_blis_impl
#define F77_dasum_sub  dasumsub_blis_impl
#define F77_scnrm2_sub scnrm2sub_blis_impl
#define F77_scasum_sub scasumsub_blis_impl
#define F77_dznrm2_sub dznrm2sub_blis_impl
#define F77_dzasum_sub dzasumsub_blis_impl
#define F77_sdsdot_sub sdsdotsub_blis_impl
/*
* Level 2 BLAS
*/
#define F77_ssymv ssymv_blis_impl
#define F77_ssbmv ssbmv_blis_impl
#define F77_sspmv sspmv_blis_impl
#define F77_sger  sger_blis_impl
#define F77_ssyr  ssyr_blis_impl
#define F77_sspr  sspr_blis_impl
#define F77_ssyr2 ssyr2_blis_impl
#define F77_sspr2 sspr2_blis_impl
#define F77_dsymv dsymv_blis_impl
#define F77_dsbmv dsbmv_blis_impl
#define F77_dspmv dspmv_blis_impl
#define F77_dger  dger_blis_impl
#define F77_dsyr  dsyr_blis_impl
#define F77_dspr  dspr_blis_impl
#define F77_dsyr2 dsyr2_blis_impl
#define F77_dspr2 dspr2_blis_impl
#define F77_chemv chemv_blis_impl
#define F77_chbmv chbmv_blis_impl
#define F77_chpmv chpmv_blis_impl
#define F77_cgeru cgeru_blis_impl
#define F77_cgerc cgerc_blis_impl
#define F77_cher  cher_blis_impl
#define F77_chpr  chpr_blis_impl
#define F77_cher2 cher2_blis_impl
#define F77_chpr2 chpr2_blis_impl
#define F77_zhemv zhemv_blis_impl
#define F77_zhbmv zhbmv_blis_impl
#define F77_zhpmv zhpmv_blis_impl
#define F77_zgeru zgeru_blis_impl
#define F77_zgerc zgerc_blis_impl
#define F77_zher  zher_blis_impl
#define F77_zhpr  zhpr_blis_impl
#define F77_zher2 zher2_blis_impl
#define F77_zhpr2 zhpr2_blis_impl
#define F77_sgemv sgemv_blis_impl
#define F77_sgbmv sgbmv_blis_impl
#define F77_strmv strmv_blis_impl
#define F77_stbmv stbmv_blis_impl
#define F77_stpmv stpmv_blis_impl
#define F77_strsv strsv_blis_impl
#define F77_stbsv stbsv_blis_impl
#define F77_stpsv stpsv_blis_impl
#define F77_dgemv dgemv_blis_impl
#define F77_dgbmv dgbmv_blis_impl
#define F77_dtrmv dtrmv_blis_impl
#define F77_dtbmv dtbmv_blis_impl
#define F77_dtpmv dtpmv_blis_impl
#define F77_dtrsv dtrsv_blis_impl
#define F77_dtbsv dtbsv_blis_impl
#define F77_dtpsv dtpsv_blis_impl
#define F77_cgemv cgemv_blis_impl
#define F77_cgbmv cgbmv_blis_impl
#define F77_ctrmv ctrmv_blis_impl
#define F77_ctbmv ctbmv_blis_impl
#define F77_ctpmv ctpmv_blis_impl
#define F77_ctrsv ctrsv_blis_impl
#define F77_ctbsv ctbsv_blis_impl
#define F77_ctpsv ctpsv_blis_impl
#define F77_zgemv zgemv_blis_impl
#define F77_zgbmv zgbmv_blis_impl
#define F77_ztrmv ztrmv_blis_impl
#define F77_ztbmv ztbmv_blis_impl
#define F77_ztpmv ztpmv_blis_impl
#define F77_ztrsv ztrsv_blis_impl
#define F77_ztbsv ztbsv_blis_impl
#define F77_ztpsv ztpsv_blis_impl
/*
* Level 3 BLAS
*/
#define F77_chemm  chemm_blis_impl
#define F77_cherk  cherk_blis_impl
#define F77_cher2k cher2k_blis_impl
#define F77_zhemm  zhemm_blis_impl
#define F77_zherk  zherk_blis_impl
#define F77_zher2k zher2k_blis_impl
#define F77_sgemm  sgemm_blis_impl
#define F77_ssymm  ssymm_blis_impl
#define F77_ssyrk  ssyrk_blis_impl
#define F77_ssyr2k ssyr2k_blis_impl
#define F77_strmm  strmm_blis_impl
#define F77_strsm  strsm_blis_impl
#define F77_dgemm  dgemm_blis_impl
#define F77_dsymm  dsymm_blis_impl
#define F77_dsyrk  dsyrk_blis_impl
#define F77_dsyr2k dsyr2k_blis_impl
#define F77_dtrmm  dtrmm_blis_impl
#define F77_dtrsm  dtrsm_blis_impl
#define F77_cgemm  cgemm_blis_impl
#define F77_csymm  csymm_blis_impl
#define F77_csyrk  csyrk_blis_impl
#define F77_csyr2k csyr2k_blis_impl
#define F77_ctrmm  ctrmm_blis_impl
#define F77_ctrsm  ctrsm_blis_impl
#define F77_zgemm  zgemm_blis_impl
#define F77_zsymm  zsymm_blis_impl
#define F77_zsyrk  zsyrk_blis_impl
#define F77_zsyr2k zsyr2k_blis_impl
#define F77_ztrmm  ztrmm_blis_impl
#define F77_ztrsm  ztrsm_blis_impl
#define F77_dgemmt  dgemmt_blis_impl
#define F77_sgemmt  sgemmt_blis_impl
#define F77_cgemmt  cgemmt_blis_impl
#define F77_zgemmt  zgemmt_blis_impl
#define F77_dzgemm  dzgemm_blis_impl

/*
* Aux Function
*/
#define F77_scabs1 scabs1_
#define F77_dcabs1 dcabs1_

/*
 * -- BLAS Extension APIs --
 */

#define F77_saxpby     saxpby_blis_impl
#define F77_daxpby     daxpby_blis_impl
#define F77_caxpby     caxpby_blis_impl
#define F77_zaxpby     zaxpby_blis_impl
#define F77_cgemm3m    cgemm3m_blis_impl
#define F77_zgemm3m    zgemm3m_blis_impl

#define F77_isamin_sub isaminsub_blis_impl
#define F77_idamin_sub idaminsub_blis_impl
#define F77_icamin_sub icaminsub_blis_impl
#define F77_izamin_sub izaminsub_blis_impl

// -- Batch APIs --
#define F77_sgemm_batch  sgemm_batch_
#define F77_dgemm_batch  dgemm_batch_
#define F77_cgemm_batch  cgemm_batch_
#define F77_zgemm_batch  zgemm_batch_

// -- Pack-Compute APIs --
#define F77_sgemm_pack_get_size  sgemm_pack_get_size_blis_impl
#define F77_dgemm_pack_get_size  dgemm_pack_get_size_blis_impl
#define F77_sgemm_pack  sgemm_pack_blis_impl
#define F77_dgemm_pack  dgemm_pack_blis_impl
#define F77_sgemm_compute  sgemm_compute_blis_impl
#define F77_dgemm_compute  dgemm_compute_blis_impl
#endif

#endif /*  CBLAS_F77_H */
