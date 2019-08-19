/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * test_gemm.hh
 *
 *
 * Purpose:
 * this header file contains all function prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#ifndef TEST_GEMM_HH
#define TEST_GEMM_HH

#include <math.h>

#include <stdio.h>
#include <stdlib.h>

// Determine the target operating system
#if defined(_WIN32) || defined(__CYGWIN__)
#define BL_OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define BL_OS_OSX 1
#elif defined(__ANDROID__)
#define BL_OS_ANDROID 1
#elif defined(__linux__)
#define BL_OS_LINUX 1
#elif defined(__bgq__)
#define BL_OS_BGQ 1
#elif defined(__bg__)
#define BL_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__)
#define BL_OS_BSD 1
#else
#error "Cannot determine operating system"
#endif

// gettimeofday() needs this.
#if BL_OS_WINDOWS
  #include <time.h>
#elif BL_OS_OSX
  #include <mach/mach_time.h>
#else
  #include <sys/time.h>
  #include <time.h>
#endif

//#include "bl_config.h"

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define B( i, j )     B[ (j)*ldb + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]
#define GEMM_SIMD_ALIGN_SIZE 32
struct aux_s {
    double *b_next;
    float  *b_next_s;
    int    ldr;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

/*
 *
 *
 */ 
double *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    double *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}



/*
 *
 *
 */
void bl_dgemm_printmatrix(
        double *A,
        int    lda,
        int    m,
        int    n
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            printf("%lf\t", A[j * lda + i]);
        }
        printf("\n");
    }
    printf("\n");
}

/*
 * The timer functions are copied directly from BLIS 0.2.0
 *
 */
static double gtod_ref_time_sec = 0.0;
double bl_clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}


double bl_clock( void )
{
	return bl_clock_helper();
}

#endif
