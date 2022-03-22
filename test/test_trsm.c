/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-2022, Advanced Micro Devices, Inc.

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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

//#define FILE_IN_OUT
#ifdef FILE_IN_OUT
//#define READ_ALL_PARAMS_FROM_FILE
#endif

// uncomment to enable cblas interface
//#define CBLAS

#define CACHE_LINE_SIZE 64

// Uncomment to enable progress printing.
//#define PROGRESS_ENABLED

#ifdef PROGRESS_ENABLED
dim_t AOCL_progress(char *api,
					dim_t lapi,
					dim_t progress,
					dim_t current_thread,
					dim_t total_threads)
{
	printf("\n%s, len = %ld, nt = %ld, tid = %ld, Processed %ld Elements",
		   api, lapi, total_threads, current_thread, progress);

	return 0;
}
#endif

int main(int argc, char **argv)
{
	obj_t a, c;
	obj_t c_save;
	obj_t alpha;
	dim_t m, n;
	num_t dt;
	int r, n_repeats;
	side_t side;
	uplo_t uploa;
	trans_t transa;
	diag_t diaga;
	f77_char f77_side;
	f77_char f77_uploa;
	f77_char f77_transa;
	f77_char f77_diaga;

	double dtime;
	double dtime_save;
	double gflops;

#ifdef FILE_IN_OUT
	FILE *fin = NULL;
	FILE *fout = NULL;
#else
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int m_input, n_input;

#ifdef PROGRESS_ENABLED
	AOCL_BLIS_set_progress(AOCL_progress);
#endif

	// bli_init();

	// bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

#ifndef PRINT
	p_begin = 200;
	p_end   = 2000;
	p_inc   = 200;

	m_input = -1;
	n_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 4;
	n_input = 4;
#endif
#endif

	n_repeats = 3;

	// dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
	// dt = BLIS_SCOMPLEX;
	// dt = BLIS_DCOMPLEX;

#ifdef FILE_IN_OUT
	if (argc < 3)
	{
		printf("Usage: ./test_trsm_XX.x input.csv output.csv\n");
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
		printf("Error opening the file %s\n", argv[2]);
		exit(1);
	}
	inc_t cs_a;
	inc_t cs_b;
#ifdef READ_ALL_PARAMS_FROM_FILE
	char side_c, uploa_c, transa_c, diaga_c;

	fprintf(fout, "side, uploa, transa, diaga, m\t n\t cs_a\t cs_b\t gflops\n");

	printf("~~~~~~~_BLAS\t side, uploa, transa, diaga, m\t n\t cs_a\t cs_b\t gflops\n");

	while (fscanf(fin, "%c %c %c %c %ld %ld %ld %ld\n", &side_c, &uploa_c, &transa_c, &diaga_c, &m, &n, &cs_a, &cs_b) == 8)
	{

		if ('l' == side_c || 'L' == side_c)
			side = BLIS_LEFT;
		else if ('r' == side_c || 'R' == side_c)
			side = BLIS_RIGHT;
		else
		{
			printf("Invalid entry for the argument 'side':%c\n", side_c);
			continue;
		}

		if ('l' == uploa_c || 'L' == uploa_c)
			uploa = BLIS_LOWER;
		else if ('u' == uploa_c || 'U' == uploa_c)
			uploa = BLIS_UPPER;
		else
		{
			printf("Invalid entry for the argument 'uplo':%c\n", uploa_c);
			continue;
		}

		if ('t' == transa_c || 'T' == transa_c)
			transa = BLIS_TRANSPOSE;
		else if ('n' == transa_c || 'N' == transa_c)
			transa = BLIS_NO_TRANSPOSE;
		else
		{
			printf("Invalid entry for the argument 'transa':%c\n", transa_c);
			continue;
		}

		if ('u' == diaga_c || 'U' == diaga_c)
			diaga = BLIS_UNIT_DIAG;
		else if ('n' == diaga_c || 'N' == diaga_c)
			diaga = BLIS_NONUNIT_DIAG;
		else
		{
			printf("Invalid entry for the argument 'diaga':%c\n", diaga_c);
			continue;
		}
#else

	fprintf(fout, "m\t n\t cs_a\t cs_b\t gflops\n");

	printf("~~~~~~~_BLAS\t m\t n\t cs_a\t cs_b\t gflops\n");

	while (fscanf(fin, "%ld %ld %ld %ld\n", &m, &n, &cs_a, &cs_b) == 4)
	{

		side = BLIS_LEFT;
		// side = BLIS_RIGHT;

		uploa = BLIS_LOWER;
		// uploa = BLIS_UPPER;

		transa = BLIS_NO_TRANSPOSE;

		diaga = BLIS_NONUNIT_DIAG;

#endif

		bli_param_map_blis_to_netlib_side(side, &f77_side);
		bli_param_map_blis_to_netlib_uplo(uploa, &f77_uploa);
		bli_param_map_blis_to_netlib_trans(transa, &f77_transa);
		bli_param_map_blis_to_netlib_diag(diaga, &f77_diaga);

		siz_t elem_size = bli_dt_size(dt);

		cs_a = bli_align_dim_to_size(cs_a, elem_size, BLIS_HEAP_STRIDE_ALIGN_SIZE);
		cs_b = bli_align_dim_to_size(cs_b, elem_size, BLIS_HEAP_STRIDE_ALIGN_SIZE);

		// Will verify the leading dimension is powers of 2 and add 64bytes.
		inc_t n_bytes = cs_a * sizeof(dt);

		if ((n_bytes != 0) && !(n_bytes & (n_bytes - 1))) // check whether n_bytes is power of 2.
			cs_a += CACHE_LINE_SIZE / sizeof(dt);

		n_bytes = cs_b * sizeof(dt);
		if ((n_bytes != 0) && !(n_bytes & (n_bytes - 1))) // check whether n_bytes is power of 2.
			cs_b += CACHE_LINE_SIZE / sizeof(dt);

		if (bli_is_left(side) && ((m > cs_a) || (m > cs_b)))
			continue; // leading dimension should be greater than number of rows

		if (bli_is_right(side) && ((n > cs_a) || (m > cs_b)))
			continue; // leading dimension should be greater than number of rows

		if (bli_is_left(side))
			bli_obj_create(dt, m, m, 1, m, &a);
		else
			bli_obj_create(dt, n, n, 1, n, &a);
		bli_obj_create(dt, m, n, 1, m, &c);
		bli_obj_create(dt, m, n, 1, m, &c_save);

#else

	for (p = p_end; p >= p_begin; p -= p_inc)
	{
		if (m_input < 0)
			m = p * (dim_t)abs(m_input);
		else
			m = (dim_t)m_input;
		if (n_input < 0)
			n = p * (dim_t)abs(n_input);
		else
			n = (dim_t)n_input;

		side = BLIS_LEFT;
		// side = BLIS_RIGHT;

		uploa = BLIS_LOWER;
		// uploa = BLIS_UPPER;

		transa = BLIS_NO_TRANSPOSE;

		diaga = BLIS_NONUNIT_DIAG;

		bli_param_map_blis_to_netlib_side(side, &f77_side);
		bli_param_map_blis_to_netlib_uplo(uploa, &f77_uploa);
		bli_param_map_blis_to_netlib_trans(transa, &f77_transa);
		bli_param_map_blis_to_netlib_diag(diaga, &f77_diaga);

		if (bli_is_left(side))
			bli_obj_create(dt, m, m, 0, 0, &a);
		else
			bli_obj_create(dt, n, n, 0, 0, &a);
		bli_obj_create(dt, m, n, 0, 0, &c);
		bli_obj_create(dt, m, n, 0, 0, &c_save);
#endif

		bli_randm(&a);
		bli_randm(&c);

		bli_obj_set_struc(BLIS_TRIANGULAR, &a);
		bli_obj_set_uplo(uploa, &a);
		bli_obj_set_conjtrans(transa, &a);
		bli_obj_set_diag(diaga, &a);

		// Randomize A and zero the unstored triangle to ensure the
		// implementation reads only from the stored region.
		bli_randm(&a);
		bli_mktrim(&a);

		// Load the diagonal of A to make it more likely to be invertible.
		bli_shiftd(&BLIS_TWO, &a);

		bli_obj_create(dt, 1, 1, 0, 0, &alpha);
		bli_setsc((2.0 / 1.0), 1.0, &alpha);

		bli_copym(&c, &c_save);

		dtime_save = DBL_MAX;

		for (r = 0; r < n_repeats; ++r)
		{
			bli_copym(&c_save, &c);

			dtime = bli_clock();

#ifdef PRINT
			bli_invertd(&a);
			bli_printm("a", &a, "%4.1f", "");
			bli_invertd(&a);
			bli_printm("c", &c, "%4.1f", "");
#endif

#ifdef BLIS

			bli_trsm(side,
					 &alpha,
					 &a,
					 &c);
#else

#ifdef CBLAS
			enum CBLAS_ORDER cblas_order;
			enum CBLAS_TRANSPOSE cblas_transa;
			enum CBLAS_UPLO cblas_uplo;
			enum CBLAS_SIDE cblas_side;
			enum CBLAS_DIAG cblas_diag;

			if (bli_obj_row_stride(&c) == 1)
				cblas_order = CblasColMajor;
			else
				cblas_order = CblasRowMajor;

			if (bli_is_trans(transa))
				cblas_transa = CblasTrans;
			else if (bli_is_conjtrans(transa))
				cblas_transa = CblasConjTrans;
			else
				cblas_transa = CblasNoTrans;

			if (bli_is_upper(uploa))
				cblas_uplo = CblasUpper;
			else
				cblas_uplo = CblasLower;

			if (bli_is_left(side))
				cblas_side = CblasLeft;
			else
				cblas_side = CblasRight;

			if (bli_is_unit_diag(diaga))
				cblas_diag = CblasUnit;
			else
				cblas_diag = CblasNonUnit;

#else
			f77_char f77_transa;
			bli_param_map_blis_to_netlib_trans(transa, &f77_transa);
#endif
			if (bli_is_float(dt))
			{
				f77_int mm = bli_obj_length(&c);
				f77_int nn = bli_obj_width(&c);
				f77_int lda = bli_obj_col_stride(&a);
				f77_int ldc = bli_obj_col_stride(&c);

				float *alphap = bli_obj_buffer(&alpha);
				float *ap = bli_obj_buffer(&a);
				float *cp = bli_obj_buffer(&c);

#ifdef CBLAS
				cblas_strsm(cblas_order,
							cblas_side,
							cblas_uplo,
							cblas_transa,
							cblas_diag,
							mm,
							nn,
							*alphap,
							ap, lda,
							cp, ldc);
#else
				strsm_(&f77_side,
					   &f77_uploa,
					   &f77_transa,
					   &f77_diaga,
					   &mm,
					   &nn,
					   alphap,
					   ap, &lda,
					   cp, &ldc);
#endif
			}
			else if (bli_is_double(dt))
			{
				f77_int mm = bli_obj_length(&c);
				f77_int nn = bli_obj_width(&c);
				f77_int lda = bli_obj_col_stride(&a);
				f77_int ldc = bli_obj_col_stride(&c);
				double *alphap = bli_obj_buffer(&alpha);
				double *ap = bli_obj_buffer(&a);
				double *cp = bli_obj_buffer(&c);

#ifdef CBLAS
				cblas_dtrsm(cblas_order,
							cblas_side,
							cblas_uplo,
							cblas_transa,
							cblas_diag,
							mm,
							nn,
							*alphap,
							ap, lda,
							cp, ldc);
#else
				dtrsm_(&f77_side,
					   &f77_uploa,
					   &f77_transa,
					   &f77_diaga,
					   &mm,
					   &nn,
					   alphap,
					   ap, &lda,
					   cp, &ldc);
#endif
			}
			else if (bli_is_scomplex(dt))
			{
				f77_int mm = bli_obj_length(&c);
				f77_int nn = bli_obj_width(&c);
				f77_int lda = bli_obj_col_stride(&a);
				f77_int ldc = bli_obj_col_stride(&c);
				scomplex *alphap = bli_obj_buffer(&alpha);
				scomplex *ap = bli_obj_buffer(&a);
				scomplex *cp = bli_obj_buffer(&c);

#ifdef CBLAS
				cblas_ctrsm(cblas_order,
							cblas_side,
							cblas_uplo,
							cblas_transa,
							cblas_diag,
							mm,
							nn,
							alphap,
							ap, lda,
							cp, ldc);
#else
				ctrsm_(&f77_side,
					   &f77_uploa,
					   &f77_transa,
					   &f77_diaga,
					   &mm,
					   &nn,
					   alphap,
					   ap, &lda,
					   cp, &ldc);
#endif
			}
			else if (bli_is_dcomplex(dt))
			{
				f77_int mm = bli_obj_length(&c);
				f77_int nn = bli_obj_width(&c);
				f77_int lda = bli_obj_col_stride(&a);
				f77_int ldc = bli_obj_col_stride(&c);
				dcomplex *alphap = bli_obj_buffer(&alpha);
				dcomplex *ap = bli_obj_buffer(&a);
				dcomplex *cp = bli_obj_buffer(&c);
#ifdef CBLAS
				cblas_ztrsm(cblas_order,
							cblas_side,
							cblas_uplo,
							cblas_transa,
							cblas_diag,
							mm,
							nn,
							alphap,
							ap, lda,
							cp, ldc);
#else
				ztrsm_(&f77_side,
					   &f77_uploa,
					   &f77_transa,
					   &f77_diaga,
					   &mm,
					   &nn,
					   alphap,
					   ap, &lda,
					   cp, &ldc);
#endif
			}
			else
			{
				printf("Invalid data type! Exiting!\n");
				exit(1);
			}
#endif

			dtime_save = bli_clock_min_diff(dtime_save, dtime);
		}

		if (bli_is_left(side))
			gflops = (1.0 * m * m * n) / (dtime_save * 1.0e9);
		else
			gflops = (1.0 * m * n * n) / (dtime_save * 1.0e9);

		if (bli_is_complex(dt))
			gflops *= 4.0;

#ifdef BLIS
		printf("data_trsm_blis");
#else
		printf("data_trsm_%s", BLAS);
#endif

#ifdef FILE_IN_OUT
#ifdef READ_ALL_PARAMS_FROM_FILE

		printf("%c\t %c\t %c\t %c\t %4lu\t %4lu\t %4lu\t %4lu\t %6.3f\n", side_c, uploa_c, transa_c, diaga_c,
			   (unsigned long)m, (unsigned long)n,
			   (unsigned long)cs_a, (unsigned long)cs_b,
			   gflops);

		fprintf(fout, "%c\t %c\t %c\t %c\t %4lu\t %4lu\t %4lu\t %4lu\t %6.3f\n", side_c, uploa_c, transa_c, diaga_c,
				(unsigned long)m, (unsigned long)n,
				(unsigned long)cs_a, (unsigned long)cs_b,
				gflops);
#else
		printf("%4lu\t %4lu\t %4lu\t %4lu\t %6.3f\n", (unsigned long)m, (unsigned long)n,
			   (unsigned long)cs_a, (unsigned long)cs_b,
			   gflops);
		fprintf(fout, "%4lu\t %4lu\t %4lu\t %4lu\t %6.3f\n", (unsigned long)m, (unsigned long)n,
				(unsigned long)cs_a, (unsigned long)cs_b,
				gflops);
#endif
		fflush(fout);

#else
		printf("( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
			   (unsigned long)(p - p_begin) / p_inc + 1,
			   (unsigned long)m,
			   (unsigned long)n, gflops);
#endif
		bli_obj_free(&alpha);

		bli_obj_free(&a);
		bli_obj_free(&c);
		bli_obj_free(&c_save);
	}

#ifdef FILE_IN_OUT
	fclose(fin);
	fclose(fout);
#endif
	// bli_finalize();

	return 0;
}
