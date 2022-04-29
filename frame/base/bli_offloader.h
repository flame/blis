#ifdef BLIS_ENABLE_AMD_OFFLOAD
#ifndef BLI_OFFLOADER_H
#define BLI_OFFLOADER_H
#include <rocblas.h>
#include <stdint.h>
#include "blis.h"

void bli_offloader_init ( void );

void bli_offloader_init_rntm_from_env ( rntm_t* rntm );

void bli_offloader_finalize ( void );

void bli_offloader_finalize_rntm_from_env ( rntm_t* rntm );

bool bli_do_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );

bool bli_do_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );

err_t bli_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );

err_t bli_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );

#endif // BLI_OFFLOADER_H
#endif // BLIS_ENABLE_AMD_OFFLOAD
