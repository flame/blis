#include "blis.h"
#ifdef BLIS_ENABLE_AMD_OFFLOAD
#include "bli_offloader.h"
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

// The global rntm_t structure. (The definition resides in bli_rntm.c.)
extern rntm_t global_rntm;

// A mutex to allow synchronous access to global_rntm. (The definition
// resides in bli_rntm.c.)
extern bli_pthread_mutex_t global_rntm_mutex;

void bli_offloader_init ( void )
{
	bli_offloader_init_rntm_from_env ( &global_rntm );
}

void bli_offloader_init_rntm_from_env ( rntm_t* rntm )
{
	// allocate struct
	rntm->offloader_state = malloc ( sizeof ( offload_t ) );
	offload_t* config = rntm->offloader_state;
	config->rocblas = NULL;

	char* s_eng = getenv ( "BLIS_OFFLOAD" );
	s_eng = ( s_eng == NULL ) ? "never" : s_eng;
	if ( strcmp ( s_eng, "never" ) == 0 )
	{
		fprintf ( stdout, "Never attempting to offload.\n" );
		config->never_offload_dgemm = true;
		config->never_offload_sgemm = true;
		config->never_offload_zgemm = true;
                config->never_offload_cgemm = true;
		config->offload_sgemm_thresh = LLONG_MAX;
		config->offload_dgemm_thresh = LLONG_MAX;
		config->offload_cgemm_thresh = LLONG_MAX;
                config->offload_zgemm_thresh = LLONG_MAX;
		return;
	}
	else if ( strcmp ( s_eng, "always" ) == 0 )
	{
		fprintf ( stdout, "Always attempting to offload.\n" );
		config->never_offload_dgemm = false;
		config->never_offload_sgemm = false;
		config->never_offload_zgemm = false;
                config->never_offload_cgemm = false;
		config->offload_sgemm_thresh = 0;
		config->offload_dgemm_thresh = 0;
		config->offload_cgemm_thresh = 0;
                config->offload_zgemm_thresh = 0;
		// still initialize rocBLAS handle
	}
	else if ( strcmp ( s_eng, "threshold" ) == 0 )
	{
		const char* s_sgemm = getenv ( "BLIS_OFFLOAD_SGEMM_THRESH" );
		const int64_t offload_after_s = ( s_sgemm == NULL ) ? LLONG_MAX : atol ( s_sgemm );
		config->offload_sgemm_thresh = offload_after_s;

		if ( offload_after_s == LLONG_MAX )
		{
			fprintf ( stdout, "Never offloading sgemms.\n" );
			config->never_offload_sgemm = true;
		}
		else
		{
			fprintf ( stdout, "Offloading all sgemms with at least M*N >= %ld\n", offload_after_s );
			config->never_offload_sgemm = false;
		}

		const char* s_dgemm = getenv ( "BLIS_OFFLOAD_DGEMM_THRESH" );
		const int64_t offload_after_d = ( s_dgemm == NULL ) ? LLONG_MAX : atol ( s_dgemm );
		config->offload_dgemm_thresh = offload_after_d;

		if ( offload_after_d == LLONG_MAX )
		{
			fprintf ( stdout, "Never offloading dgemms.\n" );
			config->never_offload_dgemm = true;
		}
		else
		{
			fprintf ( stdout, "Offloading all dgemms with at least M*N >= %ld\n", offload_after_d );
			config->never_offload_dgemm = false;
		}

		const char* s_cgemm = getenv ( "BLIS_OFFLOAD_CGEMM_THRESH" );
                const int64_t offload_after_c = ( s_sgemm == NULL ) ? LLONG_MAX : atol ( s_cgemm );
                config->offload_cgemm_thresh = offload_after_c;

                if ( offload_after_c == LLONG_MAX )
                {
                        fprintf ( stdout, "Never offloading cgemms.\n" );
                        config->never_offload_cgemm = true;
                }
                else
                {
                        fprintf ( stdout, "Offloading all cgemms with at least M*N >= %ld\n", offload_after_c );
                        config->never_offload_cgemm = false;
                }

                const char* s_zgemm = getenv ( "BLIS_OFFLOAD_ZGEMM_THRESH" );
                const int64_t offload_after_z = ( s_dgemm == NULL ) ? LLONG_MAX : atol ( s_zgemm );
                config->offload_zgemm_thresh = offload_after_z;

                if ( offload_after_z == LLONG_MAX )
                {
                        fprintf ( stdout, "Never offloading zgemms.\n" );
                        config->never_offload_zgemm = true;
                }
                else
                {
                        fprintf ( stdout, "Offloading all zgemms with at least M*N >= %ld\n", offload_after_z );
                        config->never_offload_zgemm = false;
                }

		// still initialize rocBLAS handle
	}
	else
	{
		fprintf ( stderr, "Unknown BLIS_OFFLOAD selection: %s . Offloading never.\n", s_eng );
		config->never_offload_dgemm = true;
		config->never_offload_sgemm = true;
		config->never_offload_zgemm = true;
                config->never_offload_cgemm = true;
		config->offload_sgemm_thresh = LLONG_MAX;
		config->offload_dgemm_thresh = LLONG_MAX;
		config->offload_cgemm_thresh = LLONG_MAX;
                config->offload_zgemm_thresh = LLONG_MAX;
		return;
	}

	const rocblas_status stat = rocblas_create_handle ( & ( config->rocblas ) );
	if ( stat != rocblas_status_success )
	{
		fprintf ( stderr, "Couldn't create rocBLAS handle w/ error %d\n", stat );
	}
	const rocblas_status stat_p = rocblas_set_pointer_mode ( config->rocblas,
	                              rocblas_pointer_mode_host );
	if ( stat_p != rocblas_status_success )
	{
		fprintf ( stderr, "Couldn't set rocBLAS pointer mode to host w/ error %d\n", stat );
	}
}

void bli_offloader_finalize ( void )
{
	bli_offloader_finalize_rntm_from_env ( &global_rntm );
}

void bli_offloader_finalize_rntm_from_env ( rntm_t* rntm )
{
	if ( rntm->offloader_state->rocblas != NULL )
	{
		// just destroy rocblas handle
		const rocblas_status stat = rocblas_destroy_handle ( rntm->offloader_state->rocblas );
		if ( stat != rocblas_status_success )
		{
			fprintf ( stderr, "Couldn't destroy rocBLAS handle w/ error %d\n", stat );
		}
	}

	// free struct itself
	free ( rntm->offloader_state );
}

bool bli_do_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{
	return bli_do_offload_gemmex_rntm_from_env ( &global_rntm, alpha, a, b, beta, c );
}

bool bli_do_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{

	offload_t* config = rntm->offloader_state;

	// never offload anything
	if ( config->never_offload_dgemm && config->never_offload_sgemm )
	{
		return false;
	}

	// figure out if C is integer and reject (for now)
	// NOTE: rocBLAS supports f16, f16 cmpl, f32, f32 cmpl, f64, f64 cmpl, i8, u8, i32,
	//       i32 cmpl, u32 compl, bf16, bf16 cmpl as data type settings
	//       (not in all combinations)
	if ( bli_obj_is_int ( a ) ||  bli_obj_is_int ( b ) || bli_obj_is_int ( c ) )
	{
		return false;
	}

	const inc_t rs_a = bli_obj_row_stride ( a );
	const inc_t rs_b = bli_obj_row_stride ( b );
	const inc_t rs_c = bli_obj_row_stride ( c );
	// do not offload if any row stride is != 1 (as rocBLAS only supports col strides)
	if ( rs_a != 1 || rs_b != 1 || rs_c != 1 )
	{
		return false;
	}

	// figure out if the result matrix C's M*N is above or below the data type specific cutoff
	const bool is_float_c = bli_obj_is_float ( c );
	const bool is_compl_c = bli_obj_is_complex ( c );
	if ( is_float_c && !is_compl_c && config->never_offload_sgemm )
	{
		return false;
	}
	else if ( !is_float_c && !is_compl_c && config->never_offload_dgemm )
	{
		return false;
	}
	else if ( is_float_c && is_compl_c && config->never_offload_cgemm )
        {
                return false;
        }
        else if ( !is_float_c && is_compl_c && config->never_offload_zgemm )
        {
                return false;
        }

	const dim_t m_c = bli_obj_length ( c );
	const dim_t n_c = bli_obj_width ( c );
	const size_t mul = m_c * n_c;

	if ( !is_compl_c )
	{
		return ( is_float_c ) ? ( mul >= config->offload_sgemm_thresh ) : ( mul >= config->offload_dgemm_thresh );
	}
	else
	{
		// make sure we're not conjugate AND not transpose
		if ( bli_obj_has_conj( a ) && !bli_obj_has_trans( a ) ) return false;
		if ( bli_obj_has_conj( b ) && !bli_obj_has_trans( b ) ) return false;

		return ( bli_obj_is_scomplex( c ) ) ? ( mul >= config->offload_cgemm_thresh ) : ( mul >= config->offload_zgemm_thresh );
	}
}


err_t bli_offload_gemmex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{
	return bli_offload_gemmex_rntm_from_env ( &global_rntm, alpha, a, b, beta, c );

}

err_t bli_offload_gemmex_rntm_from_env
     (
       rntm_t* rntm,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     )
{

	offload_t* config = rntm->offloader_state;

	// never offload anything
	if ( config->never_offload_dgemm && config->never_offload_sgemm )
	{
		return BLIS_FAILURE;
	}

	// figure out if C is integer and reject
	if ( bli_obj_is_int ( a ) ||  bli_obj_is_int ( b ) || bli_obj_is_int ( c ) )
	{
		return BLIS_EXPECTED_NONINTEGER_DATATYPE;
	}

	const inc_t rs_a = bli_obj_row_stride ( a );
	const inc_t rs_b = bli_obj_row_stride ( b );
	const inc_t rs_c = bli_obj_row_stride ( c );
	// do not offload if any row stride is != 1 (as rocBLAS only supports col strides)
	if ( rs_a != 1 || rs_b != 1 || rs_c != 1 )
	{
		return BLIS_INVALID_ROW_STRIDE;
	}

	// are any of the matrices complex
        const bool is_compl_a = bli_obj_is_complex ( a );
        const bool is_compl_b = bli_obj_is_complex ( b );
	const bool is_compl_c = bli_obj_is_complex ( c );

	// figure out if the result matrix C's M*N is above or below the data type specific cutoff
	const bool is_float_a = bli_obj_is_float ( a );
	const bool is_float_b = bli_obj_is_float ( b );
	const bool is_float_c = bli_obj_is_float ( c );
	if ( is_float_c && config->never_offload_sgemm )
	{
		return BLIS_FAILURE;
	}
	else if ( !is_float_c && config->never_offload_dgemm )
	{
		return BLIS_FAILURE;
	}

	const inc_t lda = bli_obj_col_stride ( a );
	const inc_t ldb = bli_obj_col_stride ( b );
	const inc_t ldc = bli_obj_col_stride ( c );
	const dim_t m_a = bli_obj_length ( a );
	const dim_t n_a = bli_obj_width ( a );
	// const dim_t m_b = bli_obj_length ( b );
	const dim_t n_b = bli_obj_width ( b );
	const dim_t m_c = bli_obj_length ( c );
	const dim_t n_c = bli_obj_width ( c );
	const size_t mul = m_c * n_c;

	bool should_offload;
	if ( !is_compl_c )
        {
                should_offload = ( is_float_c ) ? ( mul >= config->offload_sgemm_thresh ) : ( mul >= config->offload_dgemm_thresh );
        }
        else
        {
		// make sure we're not conjugate AND not transpose
                if ( bli_obj_has_conj( a ) && !bli_obj_has_trans( a ) ) should_offload = false;
		else if ( bli_obj_has_conj( b ) && !bli_obj_has_trans( b ) ) should_offload = false;
		else should_offload = ( bli_obj_is_scomplex( c ) ) ? ( mul >= config->offload_cgemm_thresh ) : ( mul >= config->offload_zgemm_thresh );
        }
	if ( !should_offload )
        {
                return BLIS_NONCONFORMAL_DIMENSIONS;
        }

	// we should offload: gather some dimensions and pointers
	void *A = bli_obj_buffer_at_off ( a ); // pointer to elements of Matrix A
	void *B = bli_obj_buffer_at_off ( b ); // pointer to elements of Matrix B
	void *C = bli_obj_buffer_at_off ( c ); // pointer to elements of Matrix C

	const bool is_trans_a = bli_obj_has_trans ( a );
	const bool is_trans_b = bli_obj_has_trans ( b );

	const size_t buff_size_a = lda * n_a * bli_obj_elem_size ( a );
	const size_t buff_size_b = ldb * n_b * bli_obj_elem_size ( b );
	const size_t buff_size_c = ldc * n_c * bli_obj_elem_size ( c );

	// inspect pointers for memory location of buffers
	hipPointerAttribute_t attr;
	const hipError_t err_insp_a = hipPointerGetAttributes(&attr, A);
        bool copy_a = true;
	if ( err_insp_a == hipSuccess )
	{
		copy_a = ( attr.memoryType != hipMemoryTypeDevice );
	}
	const hipError_t err_insp_b = hipPointerGetAttributes(&attr, B);
	bool copy_b = true;
	if ( err_insp_b == hipSuccess )
        {
                copy_b = ( attr.memoryType != hipMemoryTypeDevice );
        }
	const hipError_t err_insp_c = hipPointerGetAttributes(&attr, C);
        bool copy_c = true;
        if ( err_insp_c == hipSuccess )
        {
                copy_c = ( attr.memoryType != hipMemoryTypeDevice );
        }

	// if applicable: allocate buffers on device and copy data
	// note: we cannot assume the CPU buffers to be pinned and hence most likely the copies will be synchronous
	void* dev_buff_a;
        void* dev_buff_b;
	void* dev_buff_c;

        hipStream_t stream;
        rocblas_get_stream( config->rocblas, &stream );

	if ( copy_a )
	{
		const hipError_t err_a = hipMalloc ( &dev_buff_a, buff_size_a );
		if ( err_a != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer A of size %ld: %d\n", buff_size_a, err_a );
			return BLIS_FAILURE;
		}
		const hipError_t err_cpa = hipMemcpy ( dev_buff_a, A, buff_size_a, hipMemcpyHostToDevice );
        	if ( err_cpa != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to hipMemcpy A to device: %d\n", err_cpa );
                	return BLIS_FAILURE;
        	}
	}
	else
	{
		dev_buff_a = A;
	}

	if ( copy_b )
	{
		const hipError_t err_b = hipMalloc ( &dev_buff_b, buff_size_b );
		if ( err_b != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer B of size %ld: %d\n", buff_size_b, err_b );
			return BLIS_FAILURE;
		}
		const hipError_t err_cpb = hipMemcpy ( dev_buff_b, B, buff_size_b, hipMemcpyHostToDevice );
	        if ( err_cpb != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to hipMemcpy B to device: %d\n", err_cpb );
			return BLIS_FAILURE;
        	}

	}
	else
	{
		dev_buff_b = B;
	}

	if ( copy_c )
	{
		const hipError_t err_c = hipMalloc ( &dev_buff_c, buff_size_c );
		if ( err_c != hipSuccess )
		{
			fprintf ( stderr, "Failure to allocate device buffer C of size %ld: %d\n", buff_size_c, err_c );
			return BLIS_FAILURE;
		}

		// is beta zero?
		const bool is_beta_non_zero = !bli_obj_equals ( beta, &BLIS_ZERO );

		if ( is_beta_non_zero || ldc != m_c ) // only if the result buffer is m*n sized AND beta == 0.0 we can eschew the copy
		{
			const hipError_t err_cpc = hipMemcpy ( dev_buff_c, C, buff_size_c, hipMemcpyHostToDevice );
			if ( err_cpc != hipSuccess )
			{
				fprintf ( stderr, "Failure to hipMemcpy C to device: %d\n", err_cpc );
				return BLIS_FAILURE;
			}
		}
	}

	// call rocblas
	rocblas_operation trans_a = is_trans_a ? rocblas_operation_transpose : rocblas_operation_none;
	rocblas_operation trans_b = is_trans_b ? rocblas_operation_transpose : rocblas_operation_none;
        if ( is_compl_a && bli_obj_has_conj( a ) )
		trans_a = rocblas_operation_conjugate_transpose;
	if ( is_compl_b && bli_obj_has_conj( b ) )
                trans_b = rocblas_operation_conjugate_transpose;

	rocblas_datatype a_type;
	rocblas_datatype b_type;
	rocblas_datatype c_type;
	if ( is_compl_a )
		a_type = ( bli_obj_is_scomplex( a ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
	else
		a_type = ( is_float_a ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	if ( is_compl_b )
                b_type = ( bli_obj_is_scomplex( b ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
        else
                b_type = ( is_float_b ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	if ( is_compl_c )
                c_type = ( bli_obj_is_scomplex( c ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;
        else
                c_type = ( is_float_c ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;

	rocblas_datatype compute_type;
	if ( !is_compl_a && !is_compl_b && !is_compl_c)
		compute_type = ( is_float_a && is_float_b && is_float_c ) ? rocblas_datatype_f32_r : rocblas_datatype_f64_r;
	else
		compute_type = ( bli_obj_is_scomplex( a ) && bli_obj_is_scomplex( b ) && bli_obj_is_scomplex( c ) ) ? rocblas_datatype_f32_c : rocblas_datatype_f64_c;

	const num_t    dt_exec   = bli_obj_dt ( c );
	void* restrict alpha_f = bli_obj_buffer_for_1x1 ( dt_exec, alpha );
	void* restrict beta_f  = bli_obj_buffer_for_1x1 ( dt_exec, beta );


	const size_t k = is_trans_a ? m_a : n_a;
	const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
	const int32_t solution_index = 0;
	const uint32_t flags = 0;
	const rocblas_status roc_err = rocblas_gemm_ex ( config->rocblas,
	                               trans_a,
	                               trans_b,
	                               m_c,
	                               n_c,
	                               k,
	                               alpha_f,
	                               dev_buff_a,
	                               a_type,
	                               lda,
	                               dev_buff_b,
	                               b_type,
	                               ldb,
	                               beta_f,
	                               dev_buff_c,
	                               c_type,
	                               ldc,
	                               dev_buff_c,
	                               c_type,
	                               ldc,
	                               compute_type,
	                               algo,
	                               solution_index,
	                               flags );
	if ( roc_err != rocblas_status_success )
	{
		fprintf ( stderr, "Failure to call rocblas_dgemm: %d\n", roc_err );
		return BLIS_FAILURE;
	}

	// if applicable: free intermediate buffers
        if ( copy_a )
        {
                const hipError_t err_fa = hipFree ( dev_buff_a );
                if ( err_fa != hipSuccess )
                {
                        fprintf ( stderr, "Failure to free device buffer A: %d\n", err_fa );
                        return BLIS_FAILURE;
                }
        }
        if ( copy_b )
        {
                const hipError_t err_fb = hipFree ( dev_buff_b );
                if ( err_fb != hipSuccess )
                {
                        fprintf ( stderr, "Failure to free device buffer B: %d\n", err_fb );
                        return BLIS_FAILURE;
                }
        }

	if ( copy_c )
	{
		// copy result back synchronously
		const hipError_t err_cpr = hipMemcpy ( C, dev_buff_c, buff_size_c, hipMemcpyDeviceToHost );
		if ( err_cpr != hipSuccess )
		{
			fprintf ( stderr, "Failure to hipMemcpy C from device: %d\n", err_cpr );
			return BLIS_FAILURE;
		}
		// free
		const hipError_t err_fc = hipFree ( dev_buff_c );
	        if ( err_fc != hipSuccess )
        	{
                	fprintf ( stderr, "Failure to free device buffer C: %d\n", err_fc );
                	return BLIS_FAILURE;
        	}
	}
	else
	{
		// only synchronize on the rocBLAS stream to ensure data correctness
		hipStreamSynchronize( stream );
	}

	return BLIS_SUCCESS;
}

#endif
