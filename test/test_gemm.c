/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name of The University of Texas nor the names of its
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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

//#define FILE_IN_OUT
//#define PRINT
#define MATRIX_INITIALISATION

// uncomment to enable cblas interface
//#define CBLAS

// Uncomment to enable progress printing.
#define PROGRESS_ENABLED

#ifdef PROGRESS_ENABLED
dim_t AOCL_progress( const char* const api,
		     const dim_t lapi,
		     const dim_t progress,
		     const dim_t current_thread,
		     const dim_t total_threads )
{
	printf("\n%s, len = %ld, nt = %ld, tid = %ld, Processed %ld Elements",
		   api, lapi, total_threads, current_thread, progress);

	return 0;
}
#endif

int main(int argc, char **argv)
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	inc_t lda, ldb, ldc;
	num_t dt, dt_a;
	inc_t r, n_repeats;
	trans_t transa;
	trans_t transb;
	f77_char f77_transa;
	f77_char f77_transb;

	double dtime;
	double dtime_save;
	double gflops;

#ifdef PROGRESS_ENABLED
	AOCL_BLIS_set_progress(AOCL_progress);
#endif

	// bli_init();
	// bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 300;

	// dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
	// dt = BLIS_SCOMPLEX;
	// dt = BLIS_DCOMPLEX;

	if (bli_is_real(dt) || bli_is_scomplex(dt))
		dt_a = dt;
	else
	{
		dt_a = dt;
		// Enable the following to call
		// dzgemm
		// dt_a = BLIS_DOUBLE;
	}
	const char stor_scheme = 'C';

	transa = BLIS_NO_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;

	bli_param_map_blis_to_netlib_trans(transa, &f77_transa);
	bli_param_map_blis_to_netlib_trans(transb, &f77_transb);

	printf("BLIS Library version is : %s\n", bli_info_get_version_str());

#ifdef FILE_IN_OUT
	FILE *fin = NULL;
	FILE *fout = NULL;
	if (argc < 3)
	{
		printf("Usage: ./test_gemm_XX.x input.csv output.csv\n");
		exit(1);
	}
	fin = fopen(argv[1], "r");
	if (fin == NULL)
	{
		printf("Error opening the file %s\n", argv[1]);
		exit(1);
	}
	fout = fopen(argv[2], "w");
	if (fout == NULL)
	{
		printf("Error opening output file %s\n", argv[2]);
		exit(1);
	}
	fprintf(fout, "m\t k\t n\t cs_a\t cs_b\t cs_c\t gflops\n");
	printf("~~~~~~~~~~_BLAS\t m\t k\t n\t cs_a\t cs_b\t cs_c \t gflops\n");

	while (fscanf(fin, "%ld %ld %ld %ld %ld %ld\n", &m, &k, &n, &lda, &ldb, &ldc) == 6)
	{
		// dimensions should not be greater than leading dimensions
		// These are valid only when Op(A) = n and op(B) = n
		if ((stor_scheme == 'C') || (stor_scheme == 'c'))
		{
			if ((m > lda) || (k > ldb) || (m > ldc))
				continue;
		}
		else if ((stor_scheme == 'R') || (stor_scheme == 'r'))
		{
			// leading dimension should be greater than number of cols
			if ((k > lda) || (n > ldb) || (n > ldc))
				continue;
		}
		else
		{
			printf("Invalid Storage type\n");
			continue;
		}
#else
	dim_t p, p_begin, p_end, p_inc;
	dim_t m_input, n_input, k_input;
	p_begin = 200;
	p_end   = 2000;
	p_inc   = 200;

	m_input = n_input = k_input = -1;
	for (p = p_begin; p <= p_end; p += p_inc)
	{
		if (m_input < 0)
			m = p * (dim_t)labs(m_input);
		else
			m = (dim_t)m_input;
		if (n_input < 0)
			n = p * (dim_t)labs(n_input);
		else
			n = (dim_t)n_input;
		if (k_input < 0)
			k = p * (dim_t)labs(k_input);
		else
			k = (dim_t)k_input;

		if ((stor_scheme == 'C') || (stor_scheme == 'c'))
		{
			lda = m;
			ldb = k, ldc = m;
		}
		else if ((stor_scheme == 'R') || (stor_scheme == 'r'))
		{
			lda = k;
			ldb = n, ldc = n;
		}
#endif

		bli_obj_create(dt, 1, 1, 0, 0, &alpha);
		bli_obj_create(dt, 1, 1, 0, 0, &beta);

		siz_t elem_size = bli_dt_size(dt);

		lda = bli_align_dim_to_size(lda, elem_size, BLIS_HEAP_STRIDE_ALIGN_SIZE);
		ldb = bli_align_dim_to_size(ldb, elem_size, BLIS_HEAP_STRIDE_ALIGN_SIZE);
		ldc = bli_align_dim_to_size(ldc, elem_size, BLIS_HEAP_STRIDE_ALIGN_SIZE);

		// Will verify the leading dimension is powers of 2 and add 64bytes.
		inc_t n_bytes = lda * sizeof(dt_a);

		if ((n_bytes != 0) && !(n_bytes & (n_bytes - 1))) // check whether n_bytes is power of 2.
			lda += BLIS_SIMD_ALIGN_SIZE / sizeof(dt_a);

		n_bytes = ldb * sizeof(dt);
		if ((n_bytes != 0) && !(n_bytes & (n_bytes - 1))) // check whether n_bytes is power of 2.
			ldb += BLIS_SIMD_ALIGN_SIZE / sizeof(dt);

		n_bytes = ldc * sizeof(dt);
		if ((n_bytes != 0) && !(n_bytes & (n_bytes - 1))) // check whether n_bytes is power of 2.
			ldc += BLIS_SIMD_ALIGN_SIZE / sizeof(dt);

		if ((stor_scheme == 'C') || (stor_scheme == 'c'))
		{
			// Col-major Order
			bli_obj_create(dt_a, m, k, 1, lda, &a);
			bli_obj_create(dt, k, n, 1, ldb, &b);
			bli_obj_create(dt, m, n, 1, ldc, &c);
			bli_obj_create(dt, m, n, 1, ldc, &c_save);
		}
		else if ((stor_scheme == 'R') || (stor_scheme == 'r'))
		{
			// Row-major Order
			bli_obj_create(dt_a, m, k, lda, 1, &a);
			bli_obj_create(dt, k, n, ldb, 1, &b);
			bli_obj_create(dt, m, n, ldc, 1, &c);
			bli_obj_create(dt, m, n, ldc, 1, &c_save);
		}
		else
		{
			printf("Invalid Storage type\n");
			continue;
		}

#ifdef MATRIX_INITIALISATION
		bli_randm(&a);
		bli_randm(&b);
		bli_randm(&c);
#endif
		bli_obj_set_conjtrans(transa, &a);
		bli_obj_set_conjtrans(transb, &b);
		bli_setsc((0.9 / 1.0), 0.2, &alpha);
		bli_setsc(-(1.1 / 1.0), 0.3, &beta);

		bli_copym(&c, &c_save);
		dtime_save = DBL_MAX;
		for (r = 0; r < n_repeats; ++r)
		{
			bli_copym(&c_save, &c);
			dtime = bli_clock();

#ifdef BLIS
			bli_gemm(&alpha,
					 &a,
					 &b,
					 &beta,
					 &c);
#else
			f77_int lda, ldb, ldc;
			f77_int mm = bli_obj_length(&c);
			f77_int kk = bli_obj_width_after_trans(&a);
			f77_int nn = bli_obj_width(&c);
#ifdef CBLAS
			enum CBLAS_ORDER cblas_order;
			enum CBLAS_TRANSPOSE cblas_transa;
			enum CBLAS_TRANSPOSE cblas_transb;

			if (bli_obj_row_stride(&c) == 1)
			{
				cblas_order = CblasColMajor;
			}
			else
			{
				cblas_order = CblasRowMajor;
			}

			if (bli_is_trans(transa))
				cblas_transa = CblasTrans;
			else if (bli_is_conjtrans(transa))
				cblas_transa = CblasConjTrans;
			else
				cblas_transa = CblasNoTrans;

			if (bli_is_trans(transb))
				cblas_transb = CblasTrans;
			else if (bli_is_conjtrans(transb))
				cblas_transb = CblasConjTrans;
			else
				cblas_transb = CblasNoTrans;
#else
			f77_char f77_transa;
			f77_char f77_transb;
			bli_param_map_blis_to_netlib_trans(transa, &f77_transa);
			bli_param_map_blis_to_netlib_trans(transb, &f77_transb);
#endif
			if ((stor_scheme == 'C') || (stor_scheme == 'c'))
			{
				lda = bli_obj_col_stride(&a);
				ldb = bli_obj_col_stride(&b);
				ldc = bli_obj_col_stride(&c);
			}
			else
			{
				lda = bli_obj_row_stride(&a);
				ldb = bli_obj_row_stride(&b);
				ldc = bli_obj_row_stride(&c);
			}

			if (bli_is_float(dt))
			{
				float *alphap = bli_obj_buffer(&alpha);
				float *ap = bli_obj_buffer(&a);
				float *bp = bli_obj_buffer(&b);
				float *betap = bli_obj_buffer(&beta);
				float *cp = bli_obj_buffer(&c);
#ifdef CBLAS
				cblas_sgemm(cblas_order,
							cblas_transa,
							cblas_transb,
							mm,
							nn,
							kk,
							*alphap,
							ap, lda,
							bp, ldb,
							*betap,
							cp, ldc);
#else
				sgemm_(&f77_transa,
					   &f77_transb,
					   &mm,
					   &nn,
					   &kk,
					   alphap,
					   ap, (f77_int *)&lda,
					   bp, (f77_int *)&ldb,
					   betap,
					   cp, (f77_int *)&ldc);
#endif
			}
			else if (bli_is_double(dt))
			{
				double *alphap = bli_obj_buffer(&alpha);
				double *ap = bli_obj_buffer(&a);
				double *bp = bli_obj_buffer(&b);
				double *betap = bli_obj_buffer(&beta);
				double *cp = bli_obj_buffer(&c);
#ifdef CBLAS
				cblas_dgemm(cblas_order,
							cblas_transa,
							cblas_transb,
							mm,
							nn,
							kk,
							*alphap,
							ap, lda,
							bp, ldb,
							*betap,
							cp, ldc);
#else
				dgemm_(&f77_transa,
					   &f77_transb,
					   &mm,
					   &nn,
					   &kk,
					   alphap,
					   ap, (f77_int *)&lda,
					   bp, (f77_int *)&ldb,
					   betap,
					   cp, (f77_int *)&ldc);
#endif
			}
			else if (bli_is_scomplex(dt))
			{
				scomplex *alphap = bli_obj_buffer(&alpha);
				scomplex *ap = bli_obj_buffer(&a);
				scomplex *bp = bli_obj_buffer(&b);
				scomplex *betap = bli_obj_buffer(&beta);
				scomplex *cp = bli_obj_buffer(&c);
#ifdef CBLAS
				cblas_cgemm(cblas_order,
							cblas_transa,
							cblas_transb,
							mm,
							nn,
							kk,
							alphap,
							ap, lda,
							bp, ldb,
							betap,
							cp, ldc);
#else
				cgemm_(&f77_transa,
					   &f77_transb,
					   &mm,
					   &nn,
					   &kk,
					   alphap,
					   ap, (f77_int *)&lda,
					   bp, (f77_int *)&ldb,
					   betap,
					   cp, (f77_int *)&ldc);
#endif
			}
			else if (bli_is_dcomplex(dt))
			{
				dcomplex *alphap = bli_obj_buffer(&alpha);
				dcomplex *ap = bli_obj_buffer(&a);
				dcomplex *bp = bli_obj_buffer(&b);
				dcomplex *betap = bli_obj_buffer(&beta);
				dcomplex *cp = bli_obj_buffer(&c);
#ifdef CBLAS
				cblas_zgemm(cblas_order,
							cblas_transa,
							cblas_transb,
							mm,
							nn,
							kk,
							alphap,
							ap, lda,
							bp, ldb,
							betap,
							cp, ldc);
#else
#if 1
				if (bli_is_double(dt_a))
				{
					dzgemm_(
						&f77_transa,
						&f77_transb,
						&mm,
						&nn,
						&kk,
						alphap,
						(double *)ap, (f77_int *)&lda,
						bp, (f77_int *)&ldb,
						betap,
						cp, (f77_int *)&ldc);
				}
				else
				{
					zgemm_(&f77_transa,
						   &f77_transb,
						   &mm,
						   &nn,
						   &kk,
						   alphap,
						   ap, (f77_int *)&lda,
						   bp, (f77_int *)&ldb,
						   betap,
						   cp, (f77_int *)&ldc);
				}
#endif
#endif
			}
#endif

#ifdef PRINT
			bli_printm("a", &a, "%4.1f", "");
			bli_printm("b", &b, "%4.1f", "");
			bli_printm("c", &c, "%4.1f", "");
			bli_printm("c after", &c, "%4.1f", "");
			exit(1);
#endif
			dtime_save = bli_clock_min_diff(dtime_save, dtime);
		} // nrepeats

		gflops = (2.0 * m * k * n) / (dtime_save * 1.0e9);
		if (bli_is_dcomplex(dt) && (bli_is_double(dt_a)))
			gflops *= 2.0;
		else if (bli_is_complex(dt))
			gflops *= 4.0;

#ifdef BLIS
		printf("data_gemm_blis");
#else
		printf("data_gemm_%s", BLAS);
#endif

#ifdef FILE_IN_OUT

		printf("%6lu \t %4lu \t %4lu \t %4lu \t %4lu \t %4lu \t %6.3f\n",
			   (unsigned long)m, (unsigned long)k, (unsigned long)n,
			   (unsigned long)lda, (unsigned long)ldb, (unsigned long)ldc, gflops);

		fprintf(fout, "%6lu \t %4lu \t %4lu \t %4lu \t %4lu \t %4lu \t %6.3f\n",
				(unsigned long)m, (unsigned long)k, (unsigned long)n,
				(unsigned long)lda, (unsigned long)ldb, (unsigned long)ldc, gflops);
		fflush(fout);
#else
		printf("( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
			   (unsigned long)(p - p_begin) / p_inc + 1,
			   (unsigned long)m, (unsigned long)k,
			   (unsigned long)n, gflops);
#endif
		bli_obj_free(&alpha);
		bli_obj_free(&beta);

		bli_obj_free(&a);
		bli_obj_free(&b);
		bli_obj_free(&c);
		bli_obj_free(&c_save);
	} // while

	// bli_finalize();
#ifdef FILE_IN_OUT
	fclose(fin);
	fclose(fout);
#endif
	return 0;
}
