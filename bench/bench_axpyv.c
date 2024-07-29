/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef DT
#define DT BLIS_DOUBLE
#endif
#define AOCL_MATRIX_INITIALISATION

int main( int argc, char** argv )
{
	obj_t x, y, y_save, alpha;  // BLIS objects
	dim_t p_inc = 0;            // To keep track of number of inputs
	num_t dt;                   // BLIS datatype
	char  dt_ch;                // {S, D, Z, C} from input
	int   r, n_repeats;         // repetition counter; number of repeats

	double dtime;
	double dtime_save;
	double gflops;

	FILE* fin  = NULL;      // Input FILE*
	FILE* fout = NULL;      // Output FILE*

	n_repeats = N_REPEAT;   // Fetched from Makefile

	dt = DT;                // Set datatype as BLIS_DOUBLE

	if ( argc < 3 )
	{
		printf( "Usage: ./bench_axpyv_XX.x input.txt output.txt\n" );
		exit( 1 );
	}

	fin = fopen( argv[1], "r" );        // Open input file in read mode
	if ( fin == NULL )
	{
		printf( "Error opening input file %s\n", argv[1] );
		exit( 1 );
	}

	fout = fopen( argv[2], "w" );       // Open output file in write mode
	if ( fout == NULL )
	{
		printf( "Error opening output file %s\n", argv[2] );
		exit( 1 );
	}

#ifdef DEBUG
	fprintf( fout, "gflops\n" );
#else
	fprintf(fout, "Func Dt n alphaR  alphaI incx incy gflops\n" );
#endif

	dim_t n;        // dimension
	inc_t incx;     // stride x
	inc_t incy;     // stride y
	char tmp[256];  // to store function name, line not present in logs
	double alpha_r, alpha_i;

	// {function name} {S, D, C, Z} {n}
	// {alpha_r} {alpha_i} {incx} {incy}
	while ( fscanf( fin, "%s %c " INT_FS " %lf %lf " INT_FS INT_FS "\n",
			tmp, &dt_ch, &n,
			&alpha_r, &alpha_i, &incx, &incy ) == 7 )
	{
		if ( dt_ch == 'D' || dt_ch == 'd' ) dt = BLIS_DOUBLE;
		else if ( dt_ch == 'Z' || dt_ch == 'z' ) dt = BLIS_DCOMPLEX;
		else if ( dt_ch == 'S' || dt_ch == 's' ) dt = BLIS_FLOAT;
		else if ( dt_ch == 'C' || dt_ch == 'c' ) dt = BLIS_SCOMPLEX;
		else
		{
			printf( "Invalid data type %c\n", dt_ch );
			continue;
		}

		// Creating BLIS objects
		bli_obj_create( dt, n, 1, incx, 1, &x );      // For input vector x
		bli_obj_create( dt, n, 1, incy, 1, &y );      // For output vector y
		bli_obj_create( dt, n, 1, incy, 1, &y_save ); // For vector y_save
		bli_obj_create( dt, 1, 1, 0, 0, &alpha);      // For input scalar alpha

		#ifdef AOCL_MATRIX_INITIALISATION
			bli_randm( &x );
			bli_randm( &y );
		#endif

		// Copying contents of y to y_save
		bli_copyv( &y, &y_save );

		bli_setsc( alpha_r, alpha_i, &alpha );

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			// Copying contents of y_save to y
			bli_copyv( &y_save, &y );

			dtime = bli_clock();

#ifdef BLIS
			bli_axpyv( &alpha, &x, &y );
#else
			f77_int nn = bli_obj_length( &x );
			f77_int blas_incx = bli_obj_vector_inc( &x );
			f77_int blas_incy = bli_obj_vector_inc( &y );

			if ( bli_is_float( dt ) )
			{
				float* alphap = bli_obj_buffer( &alpha );
				float* xp = bli_obj_buffer( &x );
				float* yp = bli_obj_buffer( &y );

#ifdef CBLAS
				cblas_saxpy( nn,
							  *alphap,
							  xp,
							  blas_incx,
							  yp,
							  blas_incy );
#else
				saxpy_( &nn,
						 alphap,
						 xp,
						 &blas_incx,
						 yp,
						 &blas_incy );
#endif
			}
			else if ( bli_is_double( dt ) )
			{
				double* alphap = bli_obj_buffer( &alpha );
				double* xp = bli_obj_buffer( &x );
				double* yp = bli_obj_buffer( &y );

#ifdef CBLAS
				cblas_daxpy( nn,
							  *alphap,
							  xp,
							  blas_incx,
							  yp,
							  blas_incy );
#else
				daxpy_( &nn,
						 alphap,
						 xp,
						 &blas_incx,
						 yp,
						 &blas_incy );
#endif
			}
			else if ( bli_is_scomplex( dt ) )
			{
				scomplex* alphap = bli_obj_buffer( &alpha );
				scomplex* xp = bli_obj_buffer( &x );
				scomplex* yp = bli_obj_buffer( &y );

#ifdef CBLAS
				cblas_caxpy( nn,
							  *alphap,
							  xp,
							  blas_incx,
							  yp,
							  blas_incy );
#else
				caxpy_( &nn,
						 alphap,
						 xp,
						 &blas_incx,
						 yp,
						 &blas_incy );
#endif
			}
			else if ( bli_is_dcomplex( dt ) )
			{
				dcomplex* alphap = bli_obj_buffer( &alpha );
				dcomplex* xp = bli_obj_buffer( &x );
				dcomplex* yp = bli_obj_buffer( &y );

#ifdef CBLAS
				cblas_zaxpy( nn,
							  *alphap,
							  xp,
							  blas_incx,
							  yp,
							  blas_incy );
#else
				zaxpy_( &nn,
						 alphap,
						 xp,
						 &blas_incx,
						 yp,
						 &blas_incy );
#endif
			}
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}
		gflops = ( 3.0 * n ) / ( dtime_save * 1.0e9 );
		if ( bli_is_complex( dt ) ) gflops *= 4.0;

		printf( "data_axpyv_%s", BLAS );

		p_inc++;
		printf( " %4lu [ %4lu %7.2f ];\n",
				(unsigned long)(p_inc),
				(unsigned long)n,
				gflops );

		fprintf( fout, "%s %c %ld %lf %lf %ld %ld %6.3f\n",
				 tmp, dt_ch, n, alpha_r, alpha_i, incx, incy, gflops );
		fflush( fout );

		bli_obj_free( &x );
		bli_obj_free( &y );
	}

	return 0;
}
