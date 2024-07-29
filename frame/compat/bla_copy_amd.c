/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
	( \
	const f77_int* n, \
	const ftype*   x, const f77_int* incx, \
	      ftype*   y, const f77_int* incy  \
	) \
{ \
	dim_t  n0; \
	ftype* x0; \
	ftype* y0; \
	inc_t  incx0; \
	inc_t  incy0; \
\
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
	AOCL_DTL_LOG_COPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *n, *incx, *incy) \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv(n0, (ftype*)x, *incx, x0, incx0); \
	bli_convert_blas_incv(n0, (ftype*)y, *incy, y0, incy0); \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(ch, blisname, BLIS_TAPI_EX_SUF) \
	(\
	BLIS_NO_CONJUGATE, \
	n0, \
	x0, incx0, \
	y0, incy0, \
	NULL, \
	NULL  \
	); \
	\
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}\
\
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
	( \
	const f77_int* n, \
	const ftype*   x, const f77_int* incx, \
	      ftype*   y, const f77_int* incy  \
	) \
{ \
	PASTEF77S(ch,blasname)( n, x, incx, y, incy ); \
} \
)

// ---------------------------------------------------------

void scopy_blis_impl
(
	const f77_int* n,
	const float*   x, const f77_int* incx,
	float*   y, const f77_int* incy
)
{
	dim_t  n0;
	float* x0;
	float* y0;
	inc_t  incx0;
	inc_t  incy0;

	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
	AOCL_DTL_LOG_COPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, *incx, *incy)

	/* Initialize BLIS. */

//  bli_init_auto();

	/* Convert/typecast negative values of n to zero. */
	if (*n < 0)
		n0 = (dim_t)0;
	else
		n0 = (dim_t)(*n);

	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */
	if (*incx < 0)
	{
		/* The semantics of negative stride in BLAS are that the vector
		operand be traversed in reverse order. (Another way to think
		of this is that negative strides effectively reverse the order
		of the vector, but without any explicit data movements.) This
		is also how BLIS interprets negative strides. The differences
		is that with BLAS, the caller *always* passes in the 0th (i.e.,
		top-most or left-most) element of the vector, even when the
		stride is negative. By contrast, in BLIS, negative strides are
		used *relative* to the vector address as it is given. Thus, in
		BLIS, if this backwards traversal is desired, the caller *must*
		pass in the address to the (n-1)th (i.e., the bottom-most or
		right-most) element along with a negative stride. */

		x0 = (float*)((x)+(n0 - 1)*(-*incx));
		incx0 = (inc_t)(*incx);

	}
	else
	{
		x0 = (float*)(x);
		incx0 = (inc_t)(*incx);
	}

	if (*incy < 0)
	{
		y0 = (y)+(n0 - 1)*(-*incy);
		incy0 = (inc_t)(*incy);

	}
	else
	{
		y0 = (y);
		incy0 = (inc_t)(*incy);
	}

	cntx_t *cntx = NULL;

	// Query the architecture ID
	arch_t id = bli_arch_query_id();

	// Function pointer declaration for the function
	// that will be used by this API
	scopyv_ker_ft copyv_ker_ptr; // SCOPYV

	// Pick the kernel based on the architecture ID
	switch (id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
			copyv_ker_ptr = bli_scopyv_zen4_asm_avx512;
			break;
#endif
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:
			copyv_ker_ptr = bli_scopyv_zen_int;
			break;
		default:
		// For non-Zen architectures, query the context
		cntx = bli_gks_query_cntx();
		// Query the context for the kernel function pointers for scopyv
		copyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_COPYV_KER, cntx);
	}

	copyv_ker_ptr
	(
		BLIS_NO_CONJUGATE,
		n0,
		x0, incx0,
		y0, incy0,
		cntx
	);

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
	/* Finalize BLIS. */
//    bli_finalize_auto();
}

#ifdef BLIS_ENABLE_BLAS
void scopy_
(
	const f77_int* n,
	const float*   x, const f77_int* incx,
	float*   y, const f77_int* incy
)
{
  scopy_blis_impl( n, x, incx, y, incy );
}
#endif

// --------------------------------------------------------------------

void dcopy_blis_impl
(
	const f77_int* n,
	const double*   x, const f77_int* incx,
	double*   y, const f77_int* incy
)
{
	dim_t  n0;
	double* x0;
	double* y0;
	inc_t  incx0;
	inc_t  incy0;

	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
	AOCL_DTL_LOG_COPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, *incx, *incy)
	/* Initialize BLIS. */
	//  bli_init_auto();

	/* Convert/typecast negative values of n to zero. */
	if (*n < 0)
		n0 = (dim_t)0;
	else
		n0 = (dim_t)(*n);

	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */
	if (*incx < 0)
	{
		/* The semantics of negative stride in BLAS are that the vector
		operand be traversed in reverse order. (Another way to think
		of this is that negative strides effectively reverse the order
		of the vector, but without any explicit data movements.) This
		is also how BLIS interprets negative strides. The differences
		is that with BLAS, the caller *always* passes in the 0th (i.e.,
		top-most or left-most) element of the vector, even when the
		stride is negative. By contrast, in BLIS, negative strides are
		used *relative* to the vector address as it is given. Thus, in
		BLIS, if this backwards traversal is desired, the caller *must*
		pass in the address to the (n-1)th (i.e., the bottom-most or
		right-most) element along with a negative stride. */

		x0 = (double*)((x)+(n0 - 1)*(-*incx));
		incx0 = (inc_t)(*incx);

	}
	else
	{
		x0 = (double*)(x);
		incx0 = (inc_t)(*incx);
	}

	if (*incy < 0)
	{
		y0 = (y)+(n0 - 1)*(-*incy);
		incy0 = (inc_t)(*incy);

	}
	else
	{
		y0 = (y);
		incy0 = (inc_t)(*incy);
	}

	cntx_t *cntx = NULL;

	// Query the architecture ID
	arch_t id = bli_arch_query_id();

	// Function pointer declaration for the function
	// that will be used by this API
	dcopyv_ker_ft copyv_ker_ptr; // DCOPYV

	// Pick the kernel based on the architecture ID
	switch (id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
			// For Zen4 and Zen5, kernel implemented in AVX512 is used
			copyv_ker_ptr = bli_dcopyv_zen4_asm_avx512;
			break;
#endif
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:
			// For Zen1, Zen2 and Zen3 architectures, kernel implemented in AVX2 is used.
			copyv_ker_ptr = bli_dcopyv_zen_int;
			break;
		default:
			// For non-Zen architectures, query the context
			cntx = bli_gks_query_cntx();
			// Query the context for the kernel function pointers for dcopyv
			copyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_COPYV_KER, cntx);
	}

#ifdef BLIS_ENABLE_OPENMP
	/*
		Initializing the number of thread to one
		to avoid compiler warnings
	*/
	dim_t nt = 1;

	/*
		For the given problem size and architecture, the function
		returns the optimum number of threads with AOCL dynamic enabled
		else it returns the number of threads requested by the user.
	*/
	bli_nthreads_l1
	(
		BLIS_COPYV_KER,
		BLIS_DOUBLE,
		BLIS_DOUBLE,
		id,
		n0,
		&nt
	);

	/*
		If the number of optimum threads is 1, the OpenMP overhead
		is avoided by calling the function directly
	*/
	if (nt == 1)
	{
#endif

	copyv_ker_ptr
	(
		BLIS_NO_CONJUGATE,
		n0,
		x0, incx0,
		y0, incy0,
		cntx
	);

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
	return;

#ifdef BLIS_ENABLE_OPENMP
	}

	_Pragma("omp parallel num_threads(nt)")
	{
		dim_t start, end, length;
		thrinfo_t thread;

		// The factor by which the size should be a multiple during thread partition. 
		// The main loop of the kernel can handle 32 elements at a time hence 32 is selected for block_size.
		dim_t block_size = 32;

		// Get the thread ID
		bli_thrinfo_set_work_id( omp_get_thread_num(), &thread );

		// Get the actual number of threads spawned
		bli_thrinfo_set_n_way( omp_get_num_threads(), &thread );

		/*
		Calculate the compute range for the current thread
		based on the actual number of threads spawned
		*/

		bli_thread_range_sub
		(
			&thread,
			n0,
			block_size,
			FALSE,
			&start,
			&end
		);

		length = end - start;

		// Adjust the local pointer for computation
		double *x_thread_local = x0 + (start * incx0);
		double *y_thread_local = y0 + (start * incy0);

		// Invoke the function based on the kernel function pointer
		copyv_ker_ptr
		(
			BLIS_NO_CONJUGATE,
			length,
			x_thread_local, incx0,
			y_thread_local, incy0,
			cntx
		);
	}

#endif // BLIS_ENABLE_OPENMP

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
	/* Finalize BLIS. */
//    bli_finalize_auto();

}
#ifdef BLIS_ENABLE_BLAS

void dcopy_
(
	const f77_int* n,
	const double*   x, const f77_int* incx,
	double*   y, const f77_int* incy
)

{
  dcopy_blis_impl( n, x, incx, y, incy );
}
#endif


// ---------------------------------------------------------------

void zcopy_blis_impl
(
	const f77_int* n,
	const dcomplex*   x, const f77_int* incx,
	dcomplex*   y, const f77_int* incy
)
{
	dim_t  n0;
	dcomplex* x0;
	dcomplex* y0;
	inc_t  incx0;
	inc_t  incy0;

	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
	AOCL_DTL_LOG_COPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *n, *incx, *incy)

	/* Initialize BLIS. */

//  bli_init_auto();

	/* Convert/typecast negative values of n to zero. */
	if (*n < 0)
		n0 = (dim_t)0;
	else
		n0 = (dim_t)(*n);

	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */
	if (*incx < 0)
	{
		/* The semantics of negative stride in BLAS are that the vector
		operand be traversed in reverse order. (Another way to think
		of this is that negative strides effectively reverse the order
		of the vector, but without any explicit data movements.) This
		is also how BLIS interprets negative strides. The differences
		is that with BLAS, the caller *always* passes in the 0th (i.e.,
		top-most or left-most) element of the vector, even when the
		stride is negative. By contrast, in BLIS, negative strides are
		used *relative* to the vector address as it is given. Thus, in
		BLIS, if this backwards traversal is desired, the caller *must*
		pass in the address to the (n-1)th (i.e., the bottom-most or
		right-most) element along with a negative stride. */

		x0 = (dcomplex*)((x)+(n0 - 1)*(-*incx));
		incx0 = (inc_t)(*incx);

	}
	else
	{
		x0 = (dcomplex*)(x);
		incx0 = (inc_t)(*incx);
	}

	if (*incy < 0)
	{
		y0 = (y)+(n0 - 1)*(-*incy);
		incy0 = (inc_t)(*incy);

	}
	else
	{
		y0 = (y);
		incy0 = (inc_t)(*incy);
	}

	cntx_t *cntx = NULL;

	// Query the architecture ID
	arch_t id = bli_arch_query_id();

	// Function pointer declaration for the function
	// that will be used by this API
	zcopyv_ker_ft copyv_ker_ptr; // ZCOPYV

	// Pick the kernel based on the architecture ID
	switch (id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
			// For Zen4 and Zen5 architecture, kernel implemented in AVX512 is used
			copyv_ker_ptr = bli_zcopyv_zen4_asm_avx512;
			break;
#endif
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:
			// For Zen1, Zen2 and Zen3 architectures, kernel implemented in AVX2 is used.
			copyv_ker_ptr = bli_zcopyv_zen_int;
			break;
		default:
			// For non-Zen architectures, query the context
			cntx = bli_gks_query_cntx();
			// Query the context for the kernel function pointers for zcopyv
			copyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_COPYV_KER, cntx);
		}

#ifdef BLIS_ENABLE_OPENMP
	/*
		Initializing the number of thread to one
		to avoid compiler warnings
	*/
	dim_t nt = 1;

	/*
		For the given problem size and architecture, the function
		returns the optimum number of threads with AOCL dynamic enabled
		else it returns the number of threads requested by the user.
	*/
	bli_nthreads_l1
	(
		BLIS_COPYV_KER,
		BLIS_DCOMPLEX,
		BLIS_DCOMPLEX,
		id,
		n0,
		&nt
	);

	/*
		If the number of optimum threads is 1, the OpenMP overhead
		is avoided by calling the function directly
	*/
	if (nt == 1)
	{
#endif

	copyv_ker_ptr
	(
		BLIS_NO_CONJUGATE,
		n0,
		x0, incx0,
		y0, incy0,
		cntx
	);

#ifdef BLIS_ENABLE_OPENMP
	}

	else
	{
		_Pragma("omp parallel num_threads(nt)")
		{
			dim_t start, length;

			// Get the thread ID
			dim_t thread_id = omp_get_thread_num();

			// Get the actual number of threads spawned
			dim_t nt_use = omp_get_num_threads();
			/*
				Calculate the compute range for the current thread
				based on the actual number of threads spawned
			*/
			bli_thread_vector_partition
			(
				n0,
				nt_use,
				&start, &length,
				thread_id
			);

			// Adjust the local pointer for computation
			dcomplex *x_thread_local = x0 + (start * incx0);
			dcomplex *y_thread_local = y0 + (start * incy0);

			// Invoke the function based on the kernel function pointer
			copyv_ker_ptr
			(
				BLIS_NO_CONJUGATE,
				length,
				x_thread_local, incx0,
				y_thread_local, incy0,
				cntx
			);
		}
	}

#endif

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
	/* Finalize BLIS. */
//    bli_finalize_auto();
}

#ifdef BLIS_ENABLE_BLAS
void zcopy_
(
	const f77_int* n,
	const dcomplex*   x, const f77_int* incx,
	dcomplex*   y, const f77_int* incy
)
{
  zcopy_blis_impl( n, x, incx, y, incy );
}
#endif

INSERT_GENTFUNC_BLAS_C(copy, copyv)
