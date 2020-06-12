/*===================================================================
 * File Name :  aocldtl_blis.h
 * 
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/


#ifndef __AOCLDTL_BLIS_H
#define __AOCLDTL_BLIS_H

#include "blis.h"

#if AOCL_DTL_LOG_ENABLE
void AOCL_DTL_log_gemm_sizes(int8 loglevel,
							 obj_t* alpha,
							 obj_t* a,
							 obj_t* b,
							 obj_t* beta,
							 obj_t* c,
							 const char* filename,
							 const char* functionn_name,
							 int line);
	
#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)	\
    AOCL_DTL_log_gemm_sizes(loglevel, alpha, a, b, beta, c, __FILE__, __FUNCTION__, __LINE__);
#else
#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)
#endif

#endif

