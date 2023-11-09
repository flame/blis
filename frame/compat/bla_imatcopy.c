/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

static dim_t bli_siMatCopy_cn(dim_t rows,dim_t cols,const float alpha,float* a,dim_t lda, dim_t ldb);

static dim_t bli_diMatCopy_cn(dim_t rows,dim_t cols,const double alpha,double* a,dim_t lda, dim_t ldb);

static dim_t bli_ciMatCopy_cn(dim_t rows,dim_t cols,const scomplex alpha,scomplex* a,dim_t lda, dim_t ldb);

static dim_t bli_ciMatCopy_cr(dim_t rows,dim_t cols,const scomplex alpha,scomplex* a,dim_t lda, dim_t ldb);

static dim_t bli_ziMatCopy_cn(dim_t rows,dim_t cols,const dcomplex alpha,dcomplex* a,dim_t lda, dim_t ldb);

static dim_t bli_ziMatCopy_cr(dim_t rows,dim_t cols,const dcomplex alpha,dcomplex* a,dim_t lda, dim_t ldb);

static void bli_stranspose(float* A,float* B,dim_t cols, dim_t rows);

static void bli_dtranspose(double* A,double* B,dim_t cols, dim_t rows);

static void bli_ctranspose(scomplex* A,scomplex* B,dim_t cols, dim_t rows);

static void bli_ztranspose(dcomplex* A,dcomplex* B,dim_t cols, dim_t rows);

static void bli_stranspose(float* A,float* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_dtranspose(double* A,double* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_ctranspose(scomplex* A,scomplex* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_ztranspose(dcomplex* A,dcomplex* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

void simatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha,float* aptr, f77_int* lda, f77_int* ldb)
{
 //printf("I am from simatcopy_\n");
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid trans  setting simatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }

 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || *lda < 1 || *ldb < 1)
 {
  bli_print_msg( " Invalid function parameters simatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }

 if ( *trans == 'n' || *trans == 'N')
 {
  bli_siMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  //pre transpose
  err_t     r_val;
  float* temp = (float* ) bli_malloc_user((*rows)*(*lda)*sizeof(float), &r_val);
  bli_stranspose(aptr,temp,*lda,*rows);
  
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(float));
 
  bli_siMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);

  //post transpose
  //bli_stranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  //pre transpose
  err_t     r_val;
  float* temp = (float* ) bli_malloc_user((*rows)*(*lda)*sizeof(float), &r_val);
  bli_stranspose(aptr,temp,*lda,*rows);

  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(float));

  //bli_siMatCopy_cn(*cols,*rows,*alpha,temp,*lda,*ldb);
  
  bli_siMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);
  //post transpose
  //bli_stranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_siMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else
 {
 // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void dimatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha,double* aptr, f77_int* lda, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid trans  setting dimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }

 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || *lda < 1 || *ldb < 1)
 {
  bli_print_msg( " Invalid function parameters dimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }

 if ( *trans == 'n' || *trans == 'N')
 {
  bli_diMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  //pre transpose
  err_t     r_val;
  double* temp = (double* ) bli_malloc_user((*rows)*(*lda)*sizeof(double), &r_val);
  bli_dtranspose(aptr,temp,*lda,*rows);
     
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(double));

  bli_diMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);
  
  //post transpose
  //bli_dtranspose(temp,aptr,*rows,*lda);
  //bli_dtranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  //pre transpose
  err_t     r_val;
  double* temp = (double* ) bli_malloc_user((*rows)*(*lda)*sizeof(double), &r_val);
  bli_dtranspose(aptr,temp,*lda,*rows);

  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(double));
  
  bli_diMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);
    
  //post transpose
  //bli_dtranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_diMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else
 {
 // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void cimatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha,scomplex* aptr, f77_int* lda, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid trans  setting cimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }

 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || *lda < 1 || *ldb < 1)
 {
  bli_print_msg( " Invalid function parameters cimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }

 if ( *trans == 'n' || *trans == 'N')
 {
  bli_ciMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  //pre transpose
  err_t     r_val;
  scomplex* temp = (scomplex* ) bli_malloc_user((*rows)*(*lda)*sizeof(scomplex), &r_val);
  bli_ctranspose(aptr,temp,*lda,*rows);

  //bli_ciMatCopy_cn(*cols,*rows,*alpha,temp,*lda,*ldb);
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(scomplex));
  bli_ciMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);

  //post transpose
  //bli_ctranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {

  //pre transpose
  err_t     r_val;
  scomplex* temp = (scomplex* ) bli_malloc_user((*rows)*(*lda)*sizeof(scomplex), &r_val);
  bli_ctranspose(aptr,temp,*lda,*rows);

  //bli_ciMatCopy_cr(*cols,*rows,*alpha,temp,*lda,*ldb);
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(scomplex));
  bli_ciMatCopy_cr(*cols,*rows,*alpha,aptr,*lda,*ldb);

  //post transpose
  //bli_ctranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_ciMatCopy_cr(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void zimatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha,dcomplex* aptr, f77_int* lda, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid trans  setting zimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }

 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || *lda < 1 || *ldb < 1)
 {
  bli_print_msg( " Invalid function parameters dimatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }

 if ( *trans == 'n' || *trans == 'N')
 {
  bli_ziMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {

  //pre transpose
  err_t     r_val;
  dcomplex* temp = (dcomplex *) bli_malloc_user((*rows)*(*lda)*sizeof(dcomplex), &r_val);
  bli_ztranspose(aptr,temp,*lda,*rows);

  //bli_ziMatCopy_cn(*cols,*rows,*alpha,temp,*lda,*ldb);
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(dcomplex));
  bli_ziMatCopy_cn(*cols,*rows,*alpha,aptr,*lda,*ldb);

  //post transpose
  //bli_ztranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  //pre transpose
  err_t     r_val;
  dcomplex* temp = (dcomplex *) bli_malloc_user((*rows)*(*lda)*sizeof(dcomplex), &r_val);
  bli_ztranspose(aptr,temp,*lda,*rows);

  //bli_ziMatCopy_cr(*cols,*rows,*alpha,temp,*lda,*ldb);
  for (dim_t i = 0; i < *cols; i++)
   memcpy(&aptr[i*(*lda)],&temp[i*(*lda)],(*rows)*sizeof(scomplex));
  bli_ziMatCopy_cr(*cols,*rows,*alpha,aptr,*lda,*ldb);

  //post transpose
  //bli_ztranspose(temp,aptr,*lda,*cols);
  bli_free_user(temp);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_ziMatCopy_cr(*rows,*cols,*alpha,aptr,*lda,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

// suffix cn means - column major & non-trans
static dim_t bli_siMatCopy_cn(dim_t rows,dim_t cols,const float alpha,float* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;

 float* s_aptr;
 float* d_aptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_siMatCopy_cn() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_siMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 
 if ( lda == ldb && alpha == 1.0)
 return (0);

 s_aptr = a;
 d_aptr = a;
 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j] = 0.0;
   }
   d_aptr += ldb;
  }
 return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   d_aptr[j] = alpha * s_aptr[j];
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_diMatCopy_cn(dim_t rows,dim_t cols,const double alpha,double* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 double* s_aptr;
 double* d_aptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_diMatcopy_cn() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_diMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }

 if ( lda == ldb && alpha == 1.0)
 return (0);

 ldb = cols;
 s_aptr = a;
 d_aptr = a;
 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j] = 0.0;
   }
   d_aptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   d_aptr[j] = alpha * s_aptr[j];
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_ciMatCopy_cn(dim_t rows,dim_t cols,const scomplex alpha,scomplex* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 scomplex* s_aptr;
 scomplex* d_aptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_ciMatCopy_cn() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_ciMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 s_aptr = a;
 d_aptr = a;
 if ( alpha.real == 0.0 && alpha.imag == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j].real = 0.0;
    d_aptr[j].imag = 0.0;
   }
   d_aptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   scomplex temp = s_aptr[j];
   d_aptr[j].real = ( (alpha.real * temp.real ) - ( alpha.imag * temp.imag ) );
   d_aptr[j].imag = ( (alpha.real * temp.imag ) + ( alpha.imag * temp.real ) );
   //d_aptr[j].real = ( (alpha.real * s_aptr[j].real ) - ( alpha.imag * s_aptr[j].imag ) );
   //d_aptr[j].imag = ( (alpha.real * s_aptr[j].imag ) + ( alpha.imag * s_aptr[j].real ) );
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_ziMatCopy_cn(dim_t rows,dim_t cols,const dcomplex alpha,dcomplex* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 dcomplex* s_aptr;
 dcomplex* d_aptr; 

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_ziMatCopy_cn() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_ziMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 s_aptr = a;
 d_aptr = a;
 if ( alpha.real == 0.0 && alpha.imag == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j].real = 0.0;
    d_aptr[j].imag = 0.0;
   }
   d_aptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   dcomplex temp = s_aptr[j];
   d_aptr[j].real = ( (alpha.real * temp.real ) - ( alpha.imag * temp.imag ) );
   d_aptr[j].imag = ( (alpha.real * temp.imag ) + ( alpha.imag * temp.real ) );
   //d_aptr[j].real = ( (alpha.real * s_aptr[j].real ) - ( alpha.imag * s_aptr[j].imag ) );
   //d_aptr[j].imag = ( (alpha.real * s_aptr[j].imag ) + ( alpha.imag * s_aptr[j].real ) );
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
return(0);
}

// suffix cr means - column major & conjugate
static dim_t bli_ciMatCopy_cr(dim_t rows,dim_t cols,const scomplex alpha,scomplex* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 scomplex* s_aptr;
 scomplex* d_aptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_ciMatCopy_cr() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_ciMatCopy_cr() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 s_aptr = a;
 d_aptr = a;
 if ( alpha.real == 0.0 && alpha.imag == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j].real = 0.0;
    d_aptr[j].imag = 0.0;
   }
   d_aptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   scomplex temp = s_aptr[j];
   d_aptr[j].real = ( (alpha.real * temp.real ) + ( alpha.imag * temp.imag ) );
   d_aptr[j].imag = ( (alpha.imag * temp.real ) - ( alpha.real * temp.imag ) );
   //d_aptr[j].real = ( (alpha.real * s_aptr[j].real ) + ( alpha.imag * s_aptr[j].imag ) );
   //d_aptr[j].imag = ( (alpha.imag * s_aptr[j].real ) - ( alpha.real * s_aptr[j].imag ) );
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cr means - column major & conjugate
static dim_t bli_ziMatCopy_cr(dim_t rows,dim_t cols,const dcomplex alpha,dcomplex* a,dim_t lda, dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 dcomplex* s_aptr;
 dcomplex* d_aptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || lda < cols || ldb < cols)
 {
  fprintf( stderr, " Invalid trans setting bli_ziMatCopy_cr() %ld %ld %ld %ld \n",
          ( long )rows, ( long )cols, ( long )lda, ( long )ldb);
  bli_print_msg( " Invalid function parameters bli_ziMatCopy_cr() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 s_aptr = a;
 d_aptr = a;
 if ( alpha.real == 0.0 && alpha.imag == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    d_aptr[j].real = 0.0;
    d_aptr[j].imag = 0.0;
   }
   d_aptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   dcomplex temp = s_aptr[j];
   d_aptr[j].real = ( (alpha.real * temp.real ) + ( alpha.imag * temp.imag ) );
   d_aptr[j].imag = ( (alpha.imag * temp.real ) - ( alpha.real * temp.imag ) );
   //d_aptr[j].real = ( (alpha.real * s_aptr[j].real ) + ( alpha.imag * s_aptr[j].imag ) );
   //d_aptr[j].imag = ( (alpha.imag * s_aptr[j].real ) - ( alpha.real * s_aptr[j].imag ) );
  }
  s_aptr += lda;
  d_aptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
return(0);
}

#endif
