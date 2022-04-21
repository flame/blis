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

bool bli_do_offload_gemmex ( obj_t*  alpha,
                             obj_t*  a,
                             obj_t*  b,
                             obj_t*  beta,
                             obj_t*  c
                           );

bool bli_do_offload_gemmex_rntm_from_env ( rntm_t* rntm,
        obj_t*  alpha,
        obj_t*  a,
        obj_t*  b,
        obj_t*  beta,
        obj_t*  c
                                         );

err_t bli_offload_gemmex ( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           obj_t*  beta,
                           obj_t*  c
                         );

err_t bli_offload_gemmex_rntm_from_env ( rntm_t* rntm,
        obj_t*  alpha,
        obj_t*  a,
        obj_t*  b,
        obj_t*  beta,
        obj_t*  c
                                       );

#endif // BLI_OFFLOADER_H
#endif // BLIS_ENABLE_AMD_OFFLOAD
