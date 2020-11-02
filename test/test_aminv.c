/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

//#define PRINT

// #define BLIS   // For blis keep this line and comment CBLAS and BLAS
#ifdef BLIS_ENABLE_CBLAS
    #define CHECK_CBLAS // For cblas keep this line and comment BLIS and BLAS
    // #define CHECK_BLAS // For blas keep this line and comment BLIS only.
#endif

#ifdef CHECK_CBLAS
#include "cblas.h"
#endif

/*
 * cblas_i?amin
 * Finds the index of the element with minimum absolute value.
 *
 * Sample prototype for CBLAS interface API for SP is as follows:
 *
 * CBLAS_INDEX cblas_iamin (const int n, const float *x, const int incx);
 */

int main (int argc, char** argv )
{
    obj_t x;
    dim_t n;
    num_t dt;
    obj_t idx;
    num_t dt_idx;
    dim_t p_begin, p_end, p_inc;
    dim_t p;
    int   n_input;
    int   r, n_repeats;

    double dtime;
    double dtime_save;
    double gflops;

    n_repeats = 3;

#ifndef PRINT
    p_begin = 40;
    p_end   = 40000;
    p_inc   = 40;

    n_input = -1;
#else
    p_begin = 16;
    p_end   = 16;
    p_inc   = 1;

    n_input = 16;
#endif

#if 1
    dt = BLIS_FLOAT;
    // dt = BLIS_DOUBLE;
#else
    dt = BLIS_SCOMPLEX;
    // dt = BLIS_DCOMPLEX;
#endif

    dt_idx = BLIS_INT;

    for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
    printf( "data_iaminv_blis" );
#else
    printf( "data_iaminv_%s", BLAS );
#endif

    printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
            ( unsigned long )(p - p_begin)/p_inc + 1,
            ( unsigned long )0, 0.0 );

    for ( p = p_end; p_begin <= p; p -= p_inc )
    {
        if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
            else               n =     ( dim_t )    n_input;

        bli_obj_create( dt, n, 1, 0, 0, &x );
        bli_obj_create( dt_idx, 1, 1, 0, 0, &idx );

        bli_randm( &x );

        dtime_save = 1.0e9;

        f77_int min_index = -1;
#ifdef BLIS
        dim_t* indxp = NULL;
#endif

        for ( r = 0; r < n_repeats; ++r )
        {
            dtime = bli_clock();

#ifdef PRINT
            bli_printm( "x", &x, "%4.3f", "" );
#endif


#ifdef BLIS
            bli_aminv( &x,
                       &idx );
#else

            if ( bli_is_float( dt ) )
            {
                f77_int nn     = bli_obj_length( &x );
                f77_int incx   = bli_obj_vector_inc( &x );
                float*  xp     = bli_obj_buffer( &x );

#ifndef CHECK_BLAS
                min_index = cblas_isamin( nn,
                                          xp,
                                          incx );
#else
                min_index = isamin_( &nn,
                                     xp,
                                     &incx );
#endif
            }
            else if (bli_is_double( dt ) )
            {
                f77_int nn     = bli_obj_length( &x );
                f77_int incx   = bli_obj_vector_inc( &x );
                double*  xp    = bli_obj_buffer( &x );

#ifndef CHECK_BLAS
                min_index = cblas_idamin( nn,
                                          xp,
                                          incx );
#else
                min_index = idamin_( &nn,
                                     xp,
                                     &incx );
#endif
            }
            else if ( bli_is_scomplex( dt ) )
            {
                f77_int nn     = bli_obj_length( &x );
                f77_int incx   = bli_obj_vector_inc( &x );
                scomplex*  xp  = bli_obj_buffer( &x );

#ifndef CHECK_BLAS
                min_index = cblas_icamin( nn,
                                          xp,
                                          incx );
#else
                min_index = icamin_(  &nn,
                                      xp,
                                      &incx );
#endif
            }
            else if ( bli_is_dcomplex( dt ) )
            {
                f77_int nn     = bli_obj_length( &x );
                f77_int incx   = bli_obj_vector_inc( &x );
                dcomplex*  xp  = bli_obj_buffer( &x );

#ifndef CHECK_BLAS
                min_index = cblas_izamin( nn,
                                          xp,
                                          incx );
#else
                min_index = izamin_(  &nn,
                                      xp,
                                      &incx );
#endif
            }

#endif
            dtime_save = bli_clock_min_diff( dtime_save, dtime );
        }

        gflops = ( 1.0 * n ) / dtime_save / 1.0e9;
        if ( bli_obj_is_complex( &x ) ) gflops *= 2.0;

#ifdef BLIS
        printf( "data_iaminv_blis" );
        indxp = (dim_t *)bli_obj_buffer( &idx );
        printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ], Min Index = %d;\n",
                ( unsigned long )(p - p_begin)/p_inc + 1,
                ( unsigned long )n, gflops, indxp );
#else
        printf( "data_iaminv_%s", BLAS );
        printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ], Min Index = %d;\n",
                ( unsigned long )(p - p_begin)/p_inc + 1,
                ( unsigned long )n, gflops, min_index );
#endif

        bli_obj_free( &x );
        bli_obj_free( &idx );
    }

    bli_finalize();
    return 0;
}
