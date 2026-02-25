#ifndef __BLI_GEMM_UKR_H__
#define __BLI_GEMM_UKR_H__

#include "blis.h"

void c_hello();

void c_print(_Float16* c,
             dim_t m,
             dim_t n,
             inc_t rs_c,
             inc_t cs_c
             );

void bli_hgemm_armv8a_asm_h24x8r
    (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha,
        const void*       a,
        const void*       b,
        const void*    beta,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
    );


void bli_hgemm_armv8a_asm_h12x16r
    (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha,
        const void*       a,
        const void*       b,
        const void*    beta,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
    );


void bli_hgemm_armv8a_asm_sh12x8r
    (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha,
        const void*       a,
        const void*       b,
        const void*    beta,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
    );

#endif
