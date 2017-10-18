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

#include "blis.h"

typedef void (*cntx_init_ft)( cntx_t* cntx );

static void* l3_vir_ukr_init_fp[BLIS_NUM_IND_METHODS] =
{
/* 3mh  */ bli_cntx_set_default_l3_3mh_ukrs,
/* 3m1  */ bli_cntx_set_default_l3_3m1_ukrs,
/* 4mh  */ bli_cntx_set_default_l3_4mh_ukrs,
/* 4mb  */ bli_cntx_set_default_l3_4mb_ukrs,
/* 4m1  */ bli_cntx_set_default_l3_4m1_ukrs,
/* 1m   */ bli_cntx_set_default_l3_1m_ukrs,
/* nat  */ bli_cntx_set_default_l3_nat_ukrs
};

static void* packm_vir_init_fp[BLIS_NUM_IND_METHODS] =
{
/* 3mh  */ bli_cntx_set_default_packm_rih_kers,
/* 3m1  */ bli_cntx_set_default_packm_3mis_kers,
/* 4mh  */ bli_cntx_set_default_packm_rih_kers,
/* 4mb  */ bli_cntx_set_default_packm_4mi_kers,
/* 4m1  */ bli_cntx_set_default_packm_4mi_kers,
/* 1m   */ bli_cntx_set_default_packm_1er_kers,
/* nat  */ bli_cntx_set_default_packm_kers
};

// -----------------------------------------------------------------------------

void bli_cntx_set_default_l3_vir_ukrs( ind_t method, cntx_t* cntx )
{
	cntx_init_ft func = l3_vir_ukr_init_fp[ method ];

	func( cntx );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_default_packm_ind_kers( ind_t method, cntx_t* cntx )
{
	cntx_init_ft func = packm_vir_init_fp[ method ];

	func( cntx );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_3mh_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM3MH_UKERNEL_VIR,
	  BLIS_ZGEMM3MH_UKERNEL_VIR
	);
}
void bli_func_init_default_3mh_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_3mh_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_3mh_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_3mh_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}

void bli_cntx_set_default_l3_3mh_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_3mh_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_3mh_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_3mh_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_3mh_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_3mh_trsm_u( &func[ BLIS_TRSM_U_UKR ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_3m1_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM3M1_UKERNEL_VIR,
	  BLIS_ZGEMM3M1_UKERNEL_VIR
	);
}
void bli_func_init_default_3m1_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM3M1_L_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM3M1_L_UKERNEL_VIR
	);
}
void bli_func_init_default_3m1_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM3M1_U_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM3M1_U_UKERNEL_VIR
	);
}
void bli_func_init_default_3m1_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM3M1_L_UKERNEL_VIR,
	  BLIS_ZTRSM3M1_L_UKERNEL_VIR
	);
}
void bli_func_init_default_3m1_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM3M1_U_UKERNEL_VIR,
	  BLIS_ZTRSM3M1_U_UKERNEL_VIR
	);
}

void bli_cntx_set_default_l3_3m1_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_3m1_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_3m1_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_3m1_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_3m1_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_3m1_trsm_u( &func[ BLIS_TRSM_U_UKR ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_4mh_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM4MH_UKERNEL_VIR,
	  BLIS_ZGEMM4MH_UKERNEL_VIR
	);
}
void bli_func_init_default_4mh_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mh_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mh_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mh_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}

void bli_cntx_set_default_l3_4mh_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_4mh_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_4mh_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_4mh_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_4mh_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_4mh_trsm_u( &func[ BLIS_TRSM_U_UKR ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_4mb_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM4MB_UKERNEL_VIR,
	  BLIS_ZGEMM4MB_UKERNEL_VIR
	);
}
void bli_func_init_default_4mb_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mb_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mb_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}
void bli_func_init_default_4mb_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL, NULL, NULL, NULL
	);
}

void bli_cntx_set_default_l3_4mb_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_4mb_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_4mb_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_4mb_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_4mb_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_4mb_trsm_u( &func[ BLIS_TRSM_U_UKR ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_4m1_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM4M1_UKERNEL_VIR,
	  BLIS_ZGEMM4M1_UKERNEL_VIR
	);
}
void bli_func_init_default_4m1_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM4M1_L_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM4M1_L_UKERNEL_VIR
	);
}
void bli_func_init_default_4m1_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM4M1_U_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM4M1_U_UKERNEL_VIR
	);
}
void bli_func_init_default_4m1_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM4M1_L_UKERNEL_VIR,
	  BLIS_ZTRSM4M1_L_UKERNEL_VIR
	);
}
void bli_func_init_default_4m1_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM4M1_U_UKERNEL_VIR,
	  BLIS_ZTRSM4M1_U_UKERNEL_VIR
	);
}

void bli_cntx_set_default_l3_4m1_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_4m1_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_4m1_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_4m1_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_4m1_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_4m1_trsm_u( &func[ BLIS_TRSM_U_UKR ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_1m_gemm( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMM1M_UKERNEL_VIR,
	  BLIS_ZGEMM1M_UKERNEL_VIR
	);
}
void bli_func_init_default_1m_gemmtrsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM1M_L_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM1M_L_UKERNEL_VIR
	);
}
void bli_func_init_default_1m_gemmtrsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CGEMMTRSM1M_U_UKERNEL_VIR,
	  BLIS_ZGEMMTRSM1M_U_UKERNEL_VIR
	);
}
void bli_func_init_default_1m_trsm_l( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM1M_L_UKERNEL_VIR,
	  BLIS_ZTRSM1M_L_UKERNEL_VIR
	);
}
void bli_func_init_default_1m_trsm_u( func_t* func )
{
	bli_func_init
	(
	  func,
	  NULL,
	  NULL,
	  BLIS_CTRSM1M_U_UKERNEL_VIR,
	  BLIS_ZTRSM1M_U_UKERNEL_VIR
	);
}
void bli_func_init_real_l3_1m_ukrs( cntx_t* cntx )
{
	func_t* restrict gemm_nat_ukrs = bli_cntx_get_l3_nat_ukr( BLIS_GEMM, cntx );
	func_t* restrict gemm_vir_ukrs = bli_cntx_get_l3_vir_ukr( BLIS_GEMM, cntx );

	bli_func_copy_dt( BLIS_FLOAT,  gemm_nat_ukrs, BLIS_FLOAT,  gemm_vir_ukrs );
	bli_func_copy_dt( BLIS_DOUBLE, gemm_nat_ukrs, BLIS_DOUBLE, gemm_vir_ukrs );
}

void bli_cntx_set_default_l3_1m_ukrs( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_l3_vir_ukrs_buf( cntx );

	bli_func_init_default_1m_gemm( &func[ BLIS_GEMM_UKR ] );
	bli_func_init_default_1m_gemmtrsm_l( &func[ BLIS_GEMMTRSM_L_UKR ] );
	bli_func_init_default_1m_gemmtrsm_u( &func[ BLIS_GEMMTRSM_U_UKR ] );
	bli_func_init_default_1m_trsm_l( &func[ BLIS_TRSM_L_UKR ] );
	bli_func_init_default_1m_trsm_u( &func[ BLIS_TRSM_U_UKR ] );

	// For 1m, we employ an optimization which requires that we copy
	// the native real domain gemm ukernel function pointers to the
	// corresponding slots in the virtual gemm ukernel func_t.
	bli_func_init_real_l3_1m_ukrs( cntx );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_packm_0xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_1xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_2xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_2XK_1ER_KERNEL_REF, BLIS_DPACKM_2XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_2XK_1ER_KERNEL_REF, BLIS_ZPACKM_2XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_3xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_4xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_4XK_1ER_KERNEL_REF, BLIS_DPACKM_4XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_4XK_1ER_KERNEL_REF, BLIS_ZPACKM_4XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_5xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_6xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_6XK_1ER_KERNEL_REF, BLIS_DPACKM_6XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_6XK_1ER_KERNEL_REF, BLIS_ZPACKM_6XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_7xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_8xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_8XK_1ER_KERNEL_REF, BLIS_DPACKM_8XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_8XK_1ER_KERNEL_REF, BLIS_ZPACKM_8XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_9xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_10xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_10XK_1ER_KERNEL_REF, BLIS_DPACKM_10XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_10XK_1ER_KERNEL_REF, BLIS_ZPACKM_10XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_11xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_12xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_12XK_1ER_KERNEL_REF, BLIS_DPACKM_12XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_12XK_1ER_KERNEL_REF, BLIS_ZPACKM_12XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_13xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_14xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_14XK_1ER_KERNEL_REF, BLIS_DPACKM_14XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_14XK_1ER_KERNEL_REF, BLIS_ZPACKM_14XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_15xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_16xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_16XK_1ER_KERNEL_REF, BLIS_DPACKM_16XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_16XK_1ER_KERNEL_REF, BLIS_ZPACKM_16XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_17xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_18xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_19xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_20xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_21xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_22xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_23xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_24xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_24XK_1ER_KERNEL_REF, BLIS_DPACKM_24XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_24XK_1ER_KERNEL_REF, BLIS_ZPACKM_24XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_25xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_26xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_27xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_28xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_29xk_1er( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_30xk_1er( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_30XK_1ER_KERNEL_REF, BLIS_DPACKM_30XK_1ER_KERNEL_REF,
	  BLIS_CPACKM_30XK_1ER_KERNEL_REF, BLIS_ZPACKM_30XK_1ER_KERNEL_REF
	);
}
void bli_func_init_default_packm_31xk_1er( func_t* func )
{
	bli_func_init_null( func );
}

void bli_cntx_set_default_packm_1er_kers( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_packm_kers_buf( cntx );

	bli_func_init_default_packm_0xk_1er( &func[ BLIS_PACKM_0XK_KER ] );
	bli_func_init_default_packm_1xk_1er( &func[ BLIS_PACKM_1XK_KER ] );
	bli_func_init_default_packm_2xk_1er( &func[ BLIS_PACKM_2XK_KER ] );
	bli_func_init_default_packm_3xk_1er( &func[ BLIS_PACKM_3XK_KER ] );
	bli_func_init_default_packm_4xk_1er( &func[ BLIS_PACKM_4XK_KER ] );
	bli_func_init_default_packm_5xk_1er( &func[ BLIS_PACKM_5XK_KER ] );
	bli_func_init_default_packm_6xk_1er( &func[ BLIS_PACKM_6XK_KER ] );
	bli_func_init_default_packm_7xk_1er( &func[ BLIS_PACKM_7XK_KER ] );
	bli_func_init_default_packm_8xk_1er( &func[ BLIS_PACKM_8XK_KER ] );
	bli_func_init_default_packm_9xk_1er( &func[ BLIS_PACKM_9XK_KER ] );
	bli_func_init_default_packm_10xk_1er( &func[ BLIS_PACKM_10XK_KER ] );
	bli_func_init_default_packm_11xk_1er( &func[ BLIS_PACKM_11XK_KER ] );
	bli_func_init_default_packm_12xk_1er( &func[ BLIS_PACKM_12XK_KER ] );
	bli_func_init_default_packm_13xk_1er( &func[ BLIS_PACKM_13XK_KER ] );
	bli_func_init_default_packm_14xk_1er( &func[ BLIS_PACKM_14XK_KER ] );
	bli_func_init_default_packm_15xk_1er( &func[ BLIS_PACKM_15XK_KER ] );
	bli_func_init_default_packm_16xk_1er( &func[ BLIS_PACKM_16XK_KER ] );
	bli_func_init_default_packm_17xk_1er( &func[ BLIS_PACKM_17XK_KER ] );
	bli_func_init_default_packm_18xk_1er( &func[ BLIS_PACKM_18XK_KER ] );
	bli_func_init_default_packm_19xk_1er( &func[ BLIS_PACKM_19XK_KER ] );
	bli_func_init_default_packm_20xk_1er( &func[ BLIS_PACKM_20XK_KER ] );
	bli_func_init_default_packm_21xk_1er( &func[ BLIS_PACKM_21XK_KER ] );
	bli_func_init_default_packm_22xk_1er( &func[ BLIS_PACKM_22XK_KER ] );
	bli_func_init_default_packm_23xk_1er( &func[ BLIS_PACKM_23XK_KER ] );
	bli_func_init_default_packm_24xk_1er( &func[ BLIS_PACKM_24XK_KER ] );
	bli_func_init_default_packm_25xk_1er( &func[ BLIS_PACKM_25XK_KER ] );
	bli_func_init_default_packm_26xk_1er( &func[ BLIS_PACKM_26XK_KER ] );
	bli_func_init_default_packm_27xk_1er( &func[ BLIS_PACKM_27XK_KER ] );
	bli_func_init_default_packm_28xk_1er( &func[ BLIS_PACKM_28XK_KER ] );
	bli_func_init_default_packm_29xk_1er( &func[ BLIS_PACKM_29XK_KER ] );
	bli_func_init_default_packm_30xk_1er( &func[ BLIS_PACKM_30XK_KER ] );
	bli_func_init_default_packm_31xk_1er( &func[ BLIS_PACKM_31XK_KER ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_packm_0xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_1xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_2xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_2XK_3MIS_KERNEL_REF, BLIS_DPACKM_2XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_2XK_3MIS_KERNEL_REF, BLIS_ZPACKM_2XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_3xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_4xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_4XK_3MIS_KERNEL_REF, BLIS_DPACKM_4XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_4XK_3MIS_KERNEL_REF, BLIS_ZPACKM_4XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_5xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_6xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_6XK_3MIS_KERNEL_REF, BLIS_DPACKM_6XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_6XK_3MIS_KERNEL_REF, BLIS_ZPACKM_6XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_7xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_8xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_8XK_3MIS_KERNEL_REF, BLIS_DPACKM_8XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_8XK_3MIS_KERNEL_REF, BLIS_ZPACKM_8XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_9xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_10xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_10XK_3MIS_KERNEL_REF, BLIS_DPACKM_10XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_10XK_3MIS_KERNEL_REF, BLIS_ZPACKM_10XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_11xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_12xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_12XK_3MIS_KERNEL_REF, BLIS_DPACKM_12XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_12XK_3MIS_KERNEL_REF, BLIS_ZPACKM_12XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_13xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_14xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_14XK_3MIS_KERNEL_REF, BLIS_DPACKM_14XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_14XK_3MIS_KERNEL_REF, BLIS_ZPACKM_14XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_15xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_16xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_16XK_3MIS_KERNEL_REF, BLIS_DPACKM_16XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_16XK_3MIS_KERNEL_REF, BLIS_ZPACKM_16XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_17xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_18xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_19xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_20xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_21xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_22xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_23xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_24xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_24XK_3MIS_KERNEL_REF, BLIS_DPACKM_24XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_24XK_3MIS_KERNEL_REF, BLIS_ZPACKM_24XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_25xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_26xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_27xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_28xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_29xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_30xk_3mis( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_30XK_3MIS_KERNEL_REF, BLIS_DPACKM_30XK_3MIS_KERNEL_REF,
	  BLIS_CPACKM_30XK_3MIS_KERNEL_REF, BLIS_ZPACKM_30XK_3MIS_KERNEL_REF
	);
}
void bli_func_init_default_packm_31xk_3mis( func_t* func )
{
	bli_func_init_null( func );
}

void bli_cntx_set_default_packm_3mis_kers( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_packm_kers_buf( cntx );

	bli_func_init_default_packm_0xk_3mis( &func[ BLIS_PACKM_0XK_KER ] );
	bli_func_init_default_packm_1xk_3mis( &func[ BLIS_PACKM_1XK_KER ] );
	bli_func_init_default_packm_2xk_3mis( &func[ BLIS_PACKM_2XK_KER ] );
	bli_func_init_default_packm_3xk_3mis( &func[ BLIS_PACKM_3XK_KER ] );
	bli_func_init_default_packm_4xk_3mis( &func[ BLIS_PACKM_4XK_KER ] );
	bli_func_init_default_packm_5xk_3mis( &func[ BLIS_PACKM_5XK_KER ] );
	bli_func_init_default_packm_6xk_3mis( &func[ BLIS_PACKM_6XK_KER ] );
	bli_func_init_default_packm_7xk_3mis( &func[ BLIS_PACKM_7XK_KER ] );
	bli_func_init_default_packm_8xk_3mis( &func[ BLIS_PACKM_8XK_KER ] );
	bli_func_init_default_packm_9xk_3mis( &func[ BLIS_PACKM_9XK_KER ] );
	bli_func_init_default_packm_10xk_3mis( &func[ BLIS_PACKM_10XK_KER ] );
	bli_func_init_default_packm_11xk_3mis( &func[ BLIS_PACKM_11XK_KER ] );
	bli_func_init_default_packm_12xk_3mis( &func[ BLIS_PACKM_12XK_KER ] );
	bli_func_init_default_packm_13xk_3mis( &func[ BLIS_PACKM_13XK_KER ] );
	bli_func_init_default_packm_14xk_3mis( &func[ BLIS_PACKM_14XK_KER ] );
	bli_func_init_default_packm_15xk_3mis( &func[ BLIS_PACKM_15XK_KER ] );
	bli_func_init_default_packm_16xk_3mis( &func[ BLIS_PACKM_16XK_KER ] );
	bli_func_init_default_packm_17xk_3mis( &func[ BLIS_PACKM_17XK_KER ] );
	bli_func_init_default_packm_18xk_3mis( &func[ BLIS_PACKM_18XK_KER ] );
	bli_func_init_default_packm_19xk_3mis( &func[ BLIS_PACKM_19XK_KER ] );
	bli_func_init_default_packm_20xk_3mis( &func[ BLIS_PACKM_20XK_KER ] );
	bli_func_init_default_packm_21xk_3mis( &func[ BLIS_PACKM_21XK_KER ] );
	bli_func_init_default_packm_22xk_3mis( &func[ BLIS_PACKM_22XK_KER ] );
	bli_func_init_default_packm_23xk_3mis( &func[ BLIS_PACKM_23XK_KER ] );
	bli_func_init_default_packm_24xk_3mis( &func[ BLIS_PACKM_24XK_KER ] );
	bli_func_init_default_packm_25xk_3mis( &func[ BLIS_PACKM_25XK_KER ] );
	bli_func_init_default_packm_26xk_3mis( &func[ BLIS_PACKM_26XK_KER ] );
	bli_func_init_default_packm_27xk_3mis( &func[ BLIS_PACKM_27XK_KER ] );
	bli_func_init_default_packm_28xk_3mis( &func[ BLIS_PACKM_28XK_KER ] );
	bli_func_init_default_packm_29xk_3mis( &func[ BLIS_PACKM_29XK_KER ] );
	bli_func_init_default_packm_30xk_3mis( &func[ BLIS_PACKM_30XK_KER ] );
	bli_func_init_default_packm_31xk_3mis( &func[ BLIS_PACKM_31XK_KER ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_packm_0xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_1xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_2xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_2XK_4MI_KERNEL_REF, BLIS_DPACKM_2XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_2XK_4MI_KERNEL_REF, BLIS_ZPACKM_2XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_3xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_4xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_4XK_4MI_KERNEL_REF, BLIS_DPACKM_4XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_4XK_4MI_KERNEL_REF, BLIS_ZPACKM_4XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_5xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_6xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_6XK_4MI_KERNEL_REF, BLIS_DPACKM_6XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_6XK_4MI_KERNEL_REF, BLIS_ZPACKM_6XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_7xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_8xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_8XK_4MI_KERNEL_REF, BLIS_DPACKM_8XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_8XK_4MI_KERNEL_REF, BLIS_ZPACKM_8XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_9xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_10xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_10XK_4MI_KERNEL_REF, BLIS_DPACKM_10XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_10XK_4MI_KERNEL_REF, BLIS_ZPACKM_10XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_11xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_12xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_12XK_4MI_KERNEL_REF, BLIS_DPACKM_12XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_12XK_4MI_KERNEL_REF, BLIS_ZPACKM_12XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_13xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_14xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_14XK_4MI_KERNEL_REF, BLIS_DPACKM_14XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_14XK_4MI_KERNEL_REF, BLIS_ZPACKM_14XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_15xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_16xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_16XK_4MI_KERNEL_REF, BLIS_DPACKM_16XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_16XK_4MI_KERNEL_REF, BLIS_ZPACKM_16XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_17xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_18xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_19xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_20xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_21xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_22xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_23xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_24xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_24XK_4MI_KERNEL_REF, BLIS_DPACKM_24XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_24XK_4MI_KERNEL_REF, BLIS_ZPACKM_24XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_25xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_26xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_27xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_28xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_29xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_30xk_4mi( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_30XK_4MI_KERNEL_REF, BLIS_DPACKM_30XK_4MI_KERNEL_REF,
	  BLIS_CPACKM_30XK_4MI_KERNEL_REF, BLIS_ZPACKM_30XK_4MI_KERNEL_REF
	);
}
void bli_func_init_default_packm_31xk_4mi( func_t* func )
{
	bli_func_init_null( func );
}

void bli_cntx_set_default_packm_4mi_kers( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_packm_kers_buf( cntx );

	bli_func_init_default_packm_0xk_4mi( &func[ BLIS_PACKM_0XK_KER ] );
	bli_func_init_default_packm_1xk_4mi( &func[ BLIS_PACKM_1XK_KER ] );
	bli_func_init_default_packm_2xk_4mi( &func[ BLIS_PACKM_2XK_KER ] );
	bli_func_init_default_packm_3xk_4mi( &func[ BLIS_PACKM_3XK_KER ] );
	bli_func_init_default_packm_4xk_4mi( &func[ BLIS_PACKM_4XK_KER ] );
	bli_func_init_default_packm_5xk_4mi( &func[ BLIS_PACKM_5XK_KER ] );
	bli_func_init_default_packm_6xk_4mi( &func[ BLIS_PACKM_6XK_KER ] );
	bli_func_init_default_packm_7xk_4mi( &func[ BLIS_PACKM_7XK_KER ] );
	bli_func_init_default_packm_8xk_4mi( &func[ BLIS_PACKM_8XK_KER ] );
	bli_func_init_default_packm_9xk_4mi( &func[ BLIS_PACKM_9XK_KER ] );
	bli_func_init_default_packm_10xk_4mi( &func[ BLIS_PACKM_10XK_KER ] );
	bli_func_init_default_packm_11xk_4mi( &func[ BLIS_PACKM_11XK_KER ] );
	bli_func_init_default_packm_12xk_4mi( &func[ BLIS_PACKM_12XK_KER ] );
	bli_func_init_default_packm_13xk_4mi( &func[ BLIS_PACKM_13XK_KER ] );
	bli_func_init_default_packm_14xk_4mi( &func[ BLIS_PACKM_14XK_KER ] );
	bli_func_init_default_packm_15xk_4mi( &func[ BLIS_PACKM_15XK_KER ] );
	bli_func_init_default_packm_16xk_4mi( &func[ BLIS_PACKM_16XK_KER ] );
	bli_func_init_default_packm_17xk_4mi( &func[ BLIS_PACKM_17XK_KER ] );
	bli_func_init_default_packm_18xk_4mi( &func[ BLIS_PACKM_18XK_KER ] );
	bli_func_init_default_packm_19xk_4mi( &func[ BLIS_PACKM_19XK_KER ] );
	bli_func_init_default_packm_20xk_4mi( &func[ BLIS_PACKM_20XK_KER ] );
	bli_func_init_default_packm_21xk_4mi( &func[ BLIS_PACKM_21XK_KER ] );
	bli_func_init_default_packm_22xk_4mi( &func[ BLIS_PACKM_22XK_KER ] );
	bli_func_init_default_packm_23xk_4mi( &func[ BLIS_PACKM_23XK_KER ] );
	bli_func_init_default_packm_24xk_4mi( &func[ BLIS_PACKM_24XK_KER ] );
	bli_func_init_default_packm_25xk_4mi( &func[ BLIS_PACKM_25XK_KER ] );
	bli_func_init_default_packm_26xk_4mi( &func[ BLIS_PACKM_26XK_KER ] );
	bli_func_init_default_packm_27xk_4mi( &func[ BLIS_PACKM_27XK_KER ] );
	bli_func_init_default_packm_28xk_4mi( &func[ BLIS_PACKM_28XK_KER ] );
	bli_func_init_default_packm_29xk_4mi( &func[ BLIS_PACKM_29XK_KER ] );
	bli_func_init_default_packm_30xk_4mi( &func[ BLIS_PACKM_30XK_KER ] );
	bli_func_init_default_packm_31xk_4mi( &func[ BLIS_PACKM_31XK_KER ] );
}

// -----------------------------------------------------------------------------

void bli_func_init_default_packm_0xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_1xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_2xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_2XK_RIH_KERNEL_REF, BLIS_DPACKM_2XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_2XK_RIH_KERNEL_REF, BLIS_ZPACKM_2XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_3xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_4xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_4XK_RIH_KERNEL_REF, BLIS_DPACKM_4XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_4XK_RIH_KERNEL_REF, BLIS_ZPACKM_4XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_5xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_6xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_6XK_RIH_KERNEL_REF, BLIS_DPACKM_6XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_6XK_RIH_KERNEL_REF, BLIS_ZPACKM_6XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_7xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_8xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_8XK_RIH_KERNEL_REF, BLIS_DPACKM_8XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_8XK_RIH_KERNEL_REF, BLIS_ZPACKM_8XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_9xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_10xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_10XK_RIH_KERNEL_REF, BLIS_DPACKM_10XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_10XK_RIH_KERNEL_REF, BLIS_ZPACKM_10XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_11xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_12xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_12XK_RIH_KERNEL_REF, BLIS_DPACKM_12XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_12XK_RIH_KERNEL_REF, BLIS_ZPACKM_12XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_13xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_14xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_14XK_RIH_KERNEL_REF, BLIS_DPACKM_14XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_14XK_RIH_KERNEL_REF, BLIS_ZPACKM_14XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_15xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_16xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_16XK_RIH_KERNEL_REF, BLIS_DPACKM_16XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_16XK_RIH_KERNEL_REF, BLIS_ZPACKM_16XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_17xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_18xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_19xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_20xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_21xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_22xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_23xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_24xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_24XK_RIH_KERNEL_REF, BLIS_DPACKM_24XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_24XK_RIH_KERNEL_REF, BLIS_ZPACKM_24XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_25xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_26xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_27xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_28xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_29xk_rih( func_t* func )
{
	bli_func_init_null( func );
}
void bli_func_init_default_packm_30xk_rih( func_t* func )
{
	bli_func_init
	(
	  func,
	  BLIS_SPACKM_30XK_RIH_KERNEL_REF, BLIS_DPACKM_30XK_RIH_KERNEL_REF,
	  BLIS_CPACKM_30XK_RIH_KERNEL_REF, BLIS_ZPACKM_30XK_RIH_KERNEL_REF
	);
}
void bli_func_init_default_packm_31xk_rih( func_t* func )
{
	bli_func_init_null( func );
}

void bli_cntx_set_default_packm_rih_kers( cntx_t* cntx )
{
	func_t* funcs = bli_cntx_packm_kers_buf( cntx );

	bli_func_init_default_packm_0xk_rih( &func[ BLIS_PACKM_0XK_KER ] );
	bli_func_init_default_packm_1xk_rih( &func[ BLIS_PACKM_1XK_KER ] );
	bli_func_init_default_packm_2xk_rih( &func[ BLIS_PACKM_2XK_KER ] );
	bli_func_init_default_packm_3xk_rih( &func[ BLIS_PACKM_3XK_KER ] );
	bli_func_init_default_packm_4xk_rih( &func[ BLIS_PACKM_4XK_KER ] );
	bli_func_init_default_packm_5xk_rih( &func[ BLIS_PACKM_5XK_KER ] );
	bli_func_init_default_packm_6xk_rih( &func[ BLIS_PACKM_6XK_KER ] );
	bli_func_init_default_packm_7xk_rih( &func[ BLIS_PACKM_7XK_KER ] );
	bli_func_init_default_packm_8xk_rih( &func[ BLIS_PACKM_8XK_KER ] );
	bli_func_init_default_packm_9xk_rih( &func[ BLIS_PACKM_9XK_KER ] );
	bli_func_init_default_packm_10xk_rih( &func[ BLIS_PACKM_10XK_KER ] );
	bli_func_init_default_packm_11xk_rih( &func[ BLIS_PACKM_11XK_KER ] );
	bli_func_init_default_packm_12xk_rih( &func[ BLIS_PACKM_12XK_KER ] );
	bli_func_init_default_packm_13xk_rih( &func[ BLIS_PACKM_13XK_KER ] );
	bli_func_init_default_packm_14xk_rih( &func[ BLIS_PACKM_14XK_KER ] );
	bli_func_init_default_packm_15xk_rih( &func[ BLIS_PACKM_15XK_KER ] );
	bli_func_init_default_packm_16xk_rih( &func[ BLIS_PACKM_16XK_KER ] );
	bli_func_init_default_packm_17xk_rih( &func[ BLIS_PACKM_17XK_KER ] );
	bli_func_init_default_packm_18xk_rih( &func[ BLIS_PACKM_18XK_KER ] );
	bli_func_init_default_packm_19xk_rih( &func[ BLIS_PACKM_19XK_KER ] );
	bli_func_init_default_packm_20xk_rih( &func[ BLIS_PACKM_20XK_KER ] );
	bli_func_init_default_packm_21xk_rih( &func[ BLIS_PACKM_21XK_KER ] );
	bli_func_init_default_packm_22xk_rih( &func[ BLIS_PACKM_22XK_KER ] );
	bli_func_init_default_packm_23xk_rih( &func[ BLIS_PACKM_23XK_KER ] );
	bli_func_init_default_packm_24xk_rih( &func[ BLIS_PACKM_24XK_KER ] );
	bli_func_init_default_packm_25xk_rih( &func[ BLIS_PACKM_25XK_KER ] );
	bli_func_init_default_packm_26xk_rih( &func[ BLIS_PACKM_26XK_KER ] );
	bli_func_init_default_packm_27xk_rih( &func[ BLIS_PACKM_27XK_KER ] );
	bli_func_init_default_packm_28xk_rih( &func[ BLIS_PACKM_28XK_KER ] );
	bli_func_init_default_packm_29xk_rih( &func[ BLIS_PACKM_29XK_KER ] );
	bli_func_init_default_packm_30xk_rih( &func[ BLIS_PACKM_30XK_KER ] );
	bli_func_init_default_packm_31xk_rih( &func[ BLIS_PACKM_31XK_KER ] );
}

