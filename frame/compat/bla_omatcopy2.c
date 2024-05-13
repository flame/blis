/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef BLIS_ENABLE_BLAS

static dim_t bli_soMatCopy2_cn
       (
         dim_t 		  rows,
         dim_t 		  cols,
         const float  alpha,
         const float* a,
         dim_t 		  lda,
         dim_t 		  stridea,
         float* 	  b,
         dim_t 		  ldb,
         dim_t 		  strideb
       );

static dim_t bli_soMatCopy2_ct
       (
         dim_t 		  rows,
         dim_t 		  cols,
         const float  alpha,
         const float* a,
         dim_t 		  lda,
         dim_t 		  stridea,
         float* 	  b,
         dim_t 		  ldb,
         dim_t 		  strideb
       );

static dim_t bli_doMatCopy2_cn
       (
         dim_t 		   rows,
         dim_t 		   cols,
         const double  alpha,
         const double* a,
         dim_t 		   lda,
         dim_t 		   stridea,
         double* 	   b,
         dim_t 		   ldb,
         dim_t 		   strideb
       );

static dim_t bli_doMatCopy2_ct
       (
         dim_t 		   rows,
         dim_t 		   cols,
         const double  alpha,
         const double* a,
         dim_t 		   lda,
         dim_t 		   stridea,
         double* 	   b,
         dim_t 		   ldb,
         dim_t 		   strideb
       );

static dim_t bli_coMatCopy2_cn
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_coMatCopy2_ct
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_coMatCopy2_cr
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_coMatCopy2_cc
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_zoMatCopy2_cn
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_zoMatCopy2_ct
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_zoMatCopy2_cr
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

static dim_t bli_zoMatCopy2_cc
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       );

void somatcopy2_
     (
       f77_char* 	trans,
       f77_int* 	rows,
       f77_int* 	cols,
       const float* alpha,
       const float* aptr,
       f77_int* 	lda,
       f77_int* 	stridea,
       float* 		 bptr,
       f77_int* 	ldb,
       f77_int* 	strideb
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    //bli_init_once();
    if ( !(*trans == 'n' || *trans == 'N' ||
        *trans == 't' || *trans == 'T' ||
        *trans == 'c' || *trans == 'C' ||
        *trans == 'r' || *trans == 'R'))
    {
        bli_print_msg( " Invalid value of trans in somatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
        return ;
    }
    if ( *rows <= 0 || *cols <= 0 || alpha == NULL ||
         aptr == NULL || bptr == NULL || *lda < 1 ||
         *ldb < 1 || *stridea < 1 || *strideb < 1 )
    {
        bli_print_msg( " Invalid function parameter in somatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
        return ;
    }
    if ( *trans == 'n' || *trans == 'N' )
    {
        bli_soMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 't' || *trans == 'T' )
    {
        bli_soMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'c' || *trans == 'C' )
    {
        bli_soMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'r' || *trans == 'R' )
    {
        bli_soMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else
    {
        // do nothing
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return ;
}

void domatcopy2_
     (
       f77_char* 	 trans,
       f77_int* 	 rows,
       f77_int* 	 cols,
       const double* alpha,
       const double* aptr,
       f77_int* 	 lda,
       f77_int* 	 stridea,
       double* 		 bptr,
       f77_int* 	 ldb,
       f77_int* 	 strideb
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    //bli_init_once();
    if ( !( *trans == 'n' || *trans == 'N' ||
        *trans == 't' || *trans == 'T' ||
        *trans == 'c' || *trans == 'C' ||
        *trans == 'r' || *trans == 'R' ) )
    {
        bli_print_msg( " Invalid value of trans in domatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
        return ;
    }
    if ( *rows <= 0 || *cols <= 0 || alpha == NULL ||
         aptr == NULL || bptr == NULL || *lda < 1 ||
         *ldb < 1 || *stridea < 1 || *strideb < 1 )
    {
        bli_print_msg( " Invalid function parameter in domatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
        return ;
    }
    if ( *trans == 'n' || *trans == 'N' )
    {
        bli_doMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 't' || *trans == 'T' )
    {
        bli_doMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'c' || *trans == 'C' )
    {
        bli_doMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'r' || *trans == 'R' )
    {
        bli_doMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else
    {
        // do nothing
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return ;
}

void comatcopy2_
     (
       f77_char* 	   trans,
       f77_int* 	   rows,
       f77_int* 	   cols,
       const scomplex* alpha,
       const scomplex* aptr,
       f77_int* 	   lda,
       f77_int* 	   stridea,
       scomplex* 	   bptr,
       f77_int* 	   ldb,
       f77_int* 	   strideb
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    //bli_init_once();
    if ( !( *trans == 'n' || *trans == 'N' ||
        *trans == 't' || *trans == 'T' ||
        *trans == 'c' || *trans == 'C' ||
        *trans == 'r' || *trans == 'R' ) )
    {
        bli_print_msg( " Invalid value of trans in comatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
        return ;
    }
    if ( *rows <= 0 || *cols <= 0 || alpha == NULL ||
         aptr == NULL || bptr == NULL || *lda < 1 ||
         *ldb < 1 || *stridea < 1 || *strideb < 1 )
    {
        bli_print_msg( " Invalid function parameter in comatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
        return ;
    }
    if ( *trans == 'n' || *trans == 'N' )
    {
        bli_coMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 't' || *trans == 'T' )
    {
        bli_coMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'c' || *trans == 'C' )
    {
        bli_coMatCopy2_cc
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'r' || *trans == 'R' )
    {
        bli_coMatCopy2_cr
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else
    {
        // do nothing
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return ;
}

void zomatcopy2_
     (
       f77_char* 	   trans,
       f77_int* 	   rows,
       f77_int* 	   cols,
       const dcomplex* alpha,
       const dcomplex* aptr,
       f77_int* 	   lda,
       f77_int* 	   stridea,
       dcomplex* 	   bptr,
       f77_int* 	   ldb,
       f77_int* 	   strideb
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    //bli_init_once();
    if ( !( *trans == 'n' || *trans == 'N' ||
        *trans == 't' || *trans == 'T' ||
        *trans == 'c' || *trans == 'C' ||
        *trans == 'r' || *trans == 'R' ) )
    {
        bli_print_msg( " Invalid value of trans in zomatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
        return ;
    }
    if ( *rows <= 0 || *cols <= 0 || alpha == NULL ||
         aptr == NULL || bptr == NULL || *lda < 1 ||
         *ldb < 1 || *stridea < 1 || *strideb < 1 )
    {
        bli_print_msg( " Invalid function parameter in zomatcopy2_() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
        return ;
    }
    if ( *trans == 'n' || *trans == 'N' )
    {
        bli_zoMatCopy2_cn
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 't' || *trans == 'T' )
    {
        bli_zoMatCopy2_ct
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'c' || *trans == 'C' )
    {
        bli_zoMatCopy2_cc
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else if ( *trans == 'r' || *trans == 'R' )
    {
        bli_zoMatCopy2_cr
        (
          *rows, *cols, *alpha, aptr, *lda,
          *stridea, bptr, *ldb, *strideb
        );
    }
    else
    {
        // do nothing
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return ;
}

// suffix cn means - column major & non-trans
static dim_t bli_soMatCopy2_cn
       (
         dim_t 		  rows,
         dim_t 		  cols,
         const float  alpha,
         const float* a,
         dim_t 		  lda,
         dim_t 		  stridea,
         float* 	  b,
         dim_t 		  ldb,
         dim_t 		  strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const float* aptr;
    float* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
        strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
        ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_soMatCopy2_cn() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = alpha * aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cn means - column major & non-trans
static dim_t bli_doMatCopy2_cn
       (
         dim_t 		   rows,
         dim_t 		   cols,
         const double  alpha,
         const double* a,
         dim_t 		   lda,
         dim_t 		   stridea,
         double* 	   b,
         dim_t 		   ldb,
         dim_t 		   strideb
       )
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
  dim_t i,j;
  const double* aptr;
  double* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
        strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
        ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_doMatCopy2_cn() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )] = alpha * aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cn means - column major & non-trans
static dim_t bli_coMatCopy2_cn
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const scomplex* aptr;
    scomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
        strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
        ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_coMatCopy2_cn() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = 0.0;
                bptr[j + ( ( strideb - 1 ) * j )].imag = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = aptr[ j + ( ( stridea - 1 ) * j )].real - aptr[ j + ( ( stridea - 1 ) * j )].imag;
                bptr[j + ( ( strideb - 1 ) * j )].imag = aptr[ j + ( ( stridea - 1 ) * j )].real + aptr[ j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j + ( ( strideb - 1 ) * j )].imag = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) );
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cn means - column major & non-trans
static dim_t bli_zoMatCopy2_cn
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const dcomplex* aptr;
    dcomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_zoMatCopy2_cn() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = 0.0;
                bptr[j + ( ( strideb - 1 ) * j )].imag = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = aptr[ j + ( ( stridea - 1 ) * j )].real - aptr[ j + ( ( stridea - 1 ) * j )].imag;
                bptr[j + ( ( strideb - 1 ) * j )].imag = aptr[ j + ( ( stridea - 1 ) * j )].real + aptr[ j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j + ( ( strideb - 1 ) * j )].imag = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) );
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix ct means - column major & trans
static dim_t bli_soMatCopy2_ct
       (
         dim_t 		  rows,
         dim_t 		  cols,
         const float  alpha,
         const float* a,
         dim_t 		  lda,
         dim_t 		  stridea,
         float* 	  b,
         dim_t 		  ldb,
         dim_t 		  strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const float* aptr;
    float* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
        strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
        ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_soMatCopy2_ct() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = 0.0;
            }
        }
    }

    else if ( alpha == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = alpha * aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix ct means - column major & trans
static dim_t bli_doMatCopy2_ct
       (
         dim_t 		   rows,
         dim_t 		   cols,
         const double  alpha,
         const double* a,
         dim_t 		   lda,
         dim_t 		   stridea,
         double* 	   b,
         dim_t 		   ldb,
         dim_t 		   strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const double* aptr;
    double* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_doMatCopy2_ct() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = 0.0;
            }
        }
    }

    else if ( alpha == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb] = alpha * aptr[j + ( ( stridea - 1 ) * j )];
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix ct means - column major & trans
static dim_t bli_coMatCopy2_ct
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const scomplex* aptr;
    scomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_coMatCopy2_ct() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = 0.0;
                bptr[j * ldb].imag = 0.0;
            }
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j * ldb].imag = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j * ldb].imag = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) );
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix ct means - column major & trans
static dim_t bli_zoMatCopy2_ct
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const dcomplex* aptr;
    dcomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_zoMatCopy2_ct() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = 0.0;
                bptr[j * ldb].imag = 0.0;
            }
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j * ldb].imag = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j * ldb].imag = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) );
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cr means - column major & conjugate
static dim_t bli_coMatCopy2_cr
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const scomplex* aptr;
    scomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_coMatCopy2_cr() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = 0.0;
                bptr[j + ( ( strideb - 1 ) * j )].imag = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j + ( ( strideb - 1 ) * j )].imag = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) + ( alpha.imag * aptr[j + ( ( stridea - 1) * j )].imag ) );
                bptr[j + ( ( strideb - 1 ) * j )].imag = ( ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.real * aptr[j + ( ( stridea - 1) * j )].imag ) );
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cr means - column major & conjugate
static dim_t bli_zoMatCopy2_cr
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const dcomplex* aptr;
    dcomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( rows + ( rows - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_zoMatCopy2_cr() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;
    bptr = b;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = 0.0;
                bptr[j + ( ( strideb - 1 ) * j )].imag = 0.0;
            }
            bptr += ldb;
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0)
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j + ( ( strideb - 1 ) * j )].imag = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            for ( j = 0; j < rows; j++ )
            {
                bptr[j + ( ( strideb - 1 ) * j )].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) + ( alpha.imag * aptr[j + ( ( stridea - 1) * j )].imag ) );
                bptr[j + ( ( strideb - 1 ) * j )].imag = ( ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.real * aptr[j + ( ( stridea - 1) * j )].imag ) );
            }
            aptr += lda;
            bptr += ldb;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cc means - column major & conjugate-trans
static dim_t bli_coMatCopy2_cc
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const scomplex  alpha,
         const scomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         scomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const scomplex* aptr;
    scomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_coMatCopy2_cc() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = 0.0;
                bptr[j * ldb].imag = 0.0;
            }
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j * ldb].imag = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = ( ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].real ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j * ldb].imag = ( ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].real ) - ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

// suffix cc means - column major & conjugate-trans
static dim_t bli_zoMatCopy2_cc
       (
         dim_t 		     rows,
         dim_t 		     cols,
         const dcomplex  alpha,
         const dcomplex* a,
         dim_t 		     lda,
         dim_t 		     stridea,
         dcomplex* 	     b,
         dim_t 		     ldb,
         dim_t 		     strideb
       )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
    dim_t i,j;
    const dcomplex* aptr;
    dcomplex* bptr;

    if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || stridea < 1 ||
         strideb < 1 || lda < ( rows + ( rows - 1 ) * ( stridea - 1 ) ) ||
         ldb < ( cols + ( cols - 1 ) * ( strideb - 1 ) ) )
    {
        bli_print_msg( " Invalid function parameter in bli_zoMatCopy2_cc() .", __FILE__, __LINE__ );
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
        return ( 0 );
    }

    aptr = a;

    if ( alpha.real == 0.0 && alpha.imag == 0.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = 0.0;
                bptr[j * ldb].imag = 0.0;
            }
        }
    }

    else if ( alpha.real == 1.0 && alpha.imag == 1.0 )
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = aptr[j + ( ( stridea - 1 ) * j )].real + aptr[j + ( ( stridea - 1 ) * j )].imag;
                bptr[j * ldb].imag = aptr[j + ( ( stridea - 1 ) * j )].real - aptr[j + ( ( stridea - 1 ) * j )].imag;
            }
            aptr += lda;
        }
    }

    else
    {
        for ( i = 0; i < cols; i++ )
        {
            bptr = &b[i * strideb];
            for ( j = 0; j < rows; j++ )
            {
                bptr[j * ldb].real = ( ( alpha.real * aptr[j + (  ( stridea - 1 ) * j )].real ) + ( alpha.imag * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
                bptr[j * ldb].imag = ( ( alpha.imag * aptr[j + (  ( stridea - 1 ) * j )].real ) - ( alpha.real * aptr[j + ( ( stridea - 1 ) * j )].imag ) );
            }
            aptr += lda;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ( 0 );
}

#endif
