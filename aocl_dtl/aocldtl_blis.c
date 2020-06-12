/*===================================================================
 * File Name :  aocldtl_blis.c
 * 
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/


#include "blis.h"

#if AOCL_DTL_LOG_ENABLE
void AOCL_DTL_log_gemm_sizes(int8 loglevel,
							 obj_t* alpha,
							 obj_t* a,
							 obj_t* b,
							 obj_t* beta,
							 obj_t* c,
							 const char* filename,
							 const char* function_name,
							 int line)
{
	char buffer[256];
	gint_t m = bli_obj_length( c );
	gint_t n = bli_obj_width( c );
	gint_t k = bli_obj_length( b );
	guint_t csa = bli_obj_col_stride( a );
	guint_t csb = bli_obj_col_stride( b );
	guint_t csc = bli_obj_col_stride( c );
	guint_t rsa = bli_obj_row_stride( a );
	guint_t rsb = bli_obj_row_stride( b );
	guint_t rsc = bli_obj_row_stride( c );
	const num_t dt_exec = bli_obj_dt( c );
	float* alpha_cast = bli_obj_buffer_for_1x1( dt_exec, alpha );
	float* beta_cast  = bli_obj_buffer_for_1x1( dt_exec, beta );

	sprintf(buffer, "%ld %ld %ld %lu %lu %lu %lu %lu %lu %f %f",
				 m, k, n,
				 csa, csb, csc,
				 rsa, rsb, rsc,
				 *alpha_cast, *beta_cast);

	DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}
#endif
