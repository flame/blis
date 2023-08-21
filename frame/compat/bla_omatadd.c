/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

static dim_t bli_soMatAdd_cn(dim_t m,dim_t n,const float alpha,float* aptr,dim_t lda,const float beta,float* bptr,dim_t ldb,float* C,dim_t ldc);

static dim_t bli_doMatAdd_cn(dim_t m,dim_t n,const double alpha,double* aptr,dim_t lda,const double beta,double* bptr,dim_t ldb,double* C,dim_t ldc);

static dim_t bli_coMatAdd_cn(dim_t m,dim_t n,const scomplex alpha,scomplex* aptr,dim_t lda,const scomplex beta,scomplex* bptr,dim_t ldb,scomplex* C,dim_t ldc);

static dim_t bli_zoMatAdd_cn(dim_t m,dim_t n,const dcomplex alpha,dcomplex* aptr,dim_t lda,const dcomplex beta,dcomplex* bptr,dim_t ldb,dcomplex* C,dim_t ldc);

static void bli_stranspose(const float* A,float* B,dim_t cols, dim_t rows);

static void bli_dtranspose(const double* A,double* B,dim_t cols, dim_t rows);

static void bli_ctranspose(const scomplex* A,scomplex* B,dim_t cols, dim_t rows);

static void bli_ztranspose(const dcomplex* A,dcomplex* B,dim_t cols, dim_t rows);

static void bli_cconjugate(scomplex* A,dim_t cols,dim_t rows);

static void bli_zconjugate(dcomplex* A,dim_t cols,dim_t rows);

static void bli_stranspose(const float* A,float* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_dtranspose(const double* A,double* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_ctranspose(const scomplex* A,scomplex* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_ztranspose(const dcomplex* A,dcomplex* B,dim_t cols, dim_t rows)
{
 for (dim_t i = 0; i < cols; i++)
  for (dim_t j = 0; j < rows; j++)
   B[j*cols + i] = A[i*rows +j];
}

static void bli_cconjugate(scomplex* A,dim_t cols,dim_t rows)
{
 for (dim_t i = 0; i < cols*rows; i++)
  A[i].imag *=(-1);
}

static void bli_zconjugate(dcomplex* A,dim_t cols,dim_t rows)
{
 for (dim_t i = 0; i < cols*rows; i++)
  A[i].imag *=(-1);
}

void somatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();

 if( alpha == NULL || A == NULL || beta == NULL || B == NULL || C == NULL || *lda < 1 || *ldb < 1 || *ldc < 1 || *m < 1 || *n < 1)
 {
  bli_print_msg( " Invalid function parameters somatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( !(*transa == 'n' || *transa == 'N' ||
  *transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R'))
 {
  bli_print_msg( " Invalid value of transa somatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( !(*transb == 'n' || *transb == 'N' ||
  *transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R'))
 {
  bli_print_msg( " Invalid value of transb somatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 float* aptr;
 float* bptr;
 err_t     r_val;

 //pre transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  aptr = (float *) bli_malloc_user((*m)*(*lda)*sizeof(float), &r_val);
  bli_stranspose(A,aptr,*m,*lda);
 }
 else
 {
  aptr = (float *)A;
 }

 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bptr = (float *) bli_malloc_user((*m)*(*ldb)*sizeof(float), &r_val);
  bli_stranspose(B,bptr,*m,*ldb);
 }
 else
 {
  bptr = (float *)B;
 }

 bli_soMatAdd_cn(*m,*n,*alpha,aptr,*lda,*beta,bptr,*ldb,C,*ldc);

 //post transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  bli_free_user(aptr);
 }
 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bli_free_user(bptr);
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void domatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if( alpha == NULL || A == NULL || beta == NULL || B == NULL || C == NULL || *lda < 1 || *ldb < 1 || *ldc < 1 || *m < 1 || *n < 1)
 {
  bli_print_msg( " Invalid function parameters domatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( !(*transa == 'n' || *transa == 'N' ||
  *transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R'))
 {
  bli_print_msg( " Invalid value of transa domatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( !(*transb == 'n' || *transb == 'N' ||
  *transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R'))
 {
  bli_print_msg( " Invalid value of transb domatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 double* aptr;
 double* bptr;
 err_t     r_val;

 //pre transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  aptr = (double *) bli_malloc_user((*m)*(*lda)*sizeof(double), &r_val);
  bli_dtranspose(A,aptr,*m,*lda);
 }
 else
 {
  aptr = (double *)A;
 }

 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bptr = (double *) bli_malloc_user((*m)*(*ldb)*sizeof(double), &r_val);
  bli_dtranspose(B,bptr,*m,*ldb);
 }
 else
 {
  bptr = (double *)B;
 }

 bli_doMatAdd_cn(*m,*n,*alpha,aptr,*lda,*beta,bptr,*ldb,C,*ldc);

 //post transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  bli_free_user(aptr);
 }
 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bli_free_user(bptr);
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void comatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if( alpha == NULL || A == NULL || beta == NULL || B == NULL || C == NULL || *lda < 1 || *ldb < 1 || *ldc < 1 || *m < 1 || *n < 1)
 {
  bli_print_msg( " Invalid function parameters comatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( !(*transa == 'n' || *transa == 'N' ||
  *transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R'))
 {
  bli_print_msg( " Invalid value for transa comatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( !(*transb == 'n' || *transb == 'N' ||
  *transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R'))
 {
  bli_print_msg( " Invalid value of transb domatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 scomplex* aptr;
 scomplex* bptr;
 err_t     r_val;

 //pre transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  aptr = (scomplex *) bli_malloc_user((*m)*(*lda)*sizeof(scomplex), &r_val);
  bli_ctranspose(A,aptr,*m,*lda);
 }
 else
 {
  aptr = (scomplex*)A;
 }

 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bptr = (scomplex *) bli_malloc_user((*m)*(*ldb)*sizeof(scomplex), &r_val);
  bli_ctranspose(B,bptr,*m,*ldb);
 }
 else
 {
  bptr = (scomplex*)B;
 }

 // If Conjugate
 if( *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R')
 {
  bli_cconjugate(aptr,*m,*lda);
 }

 if( *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R')
 {
  bli_cconjugate(bptr,*m,*ldb);
 }

 bli_coMatAdd_cn(*m,*n,*alpha,aptr,*lda,*beta,bptr,*ldb,C,*ldc);

 //post transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  bli_free_user(aptr);
 }
 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bli_free_user(bptr);
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void zomatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if( alpha == NULL || A == NULL || beta == NULL || B == NULL || C == NULL || *lda < 1 || *ldb < 1 || *ldc < 1 || *m < 1 || *n < 1)
 {
  bli_print_msg( " Invalid function parameters zomatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }

 if ( !(*transa == 'n' || *transa == 'N' ||
  *transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R'))
 {
  bli_print_msg( " Invalid value for transa zomatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( !(*transb == 'n' || *transb == 'N' ||
  *transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R'))
 {
  bli_print_msg( " Invalid value for transb zomatadd_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }

 dcomplex* aptr;
 dcomplex* bptr;
 err_t     r_val;

 //pre transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  aptr = (dcomplex *) bli_malloc_user((*m)*(*lda)*sizeof(dcomplex), &r_val);
  bli_ztranspose(A,aptr,*m,*lda);
 }
 else
 {
  aptr = (dcomplex*)A;
 }

 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bptr = (dcomplex *) bli_malloc_user((*m)*(*ldb)*sizeof(dcomplex), &r_val);
  bli_ztranspose(B,bptr,*m,*ldb);
 }
 else
 {
  bptr = (dcomplex*)B;
 }

 // If Conjugate
 if( *transa == 'c' || *transa == 'C' ||
  *transa == 'r' || *transa == 'R')
 {
  bli_zconjugate(aptr,*m,*lda);
 }

 if( *transb == 'c' || *transb == 'C' ||
  *transb == 'r' || *transb == 'R')
 {
  bli_zconjugate(bptr,*m,*ldb);
 }

 bli_zoMatAdd_cn(*m,*n,*alpha,aptr,*lda,*beta,bptr,*ldb,C,*ldc);

 //post transpose
 if(*transa == 't' || *transa == 'T' ||
  *transa == 'c' || *transa == 'C')
 {
  bli_free_user(aptr);
 }
 if(*transb == 't' || *transb == 'T' ||
  *transb == 'c' || *transb == 'C')
 {
  bli_free_user(bptr);
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

static dim_t bli_soMatAdd_cn(dim_t rows,dim_t cols,const float alpha,float* aptr,dim_t lda,const float beta,float* bptr,dim_t ldb,float* C,dim_t ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 if ( rows <= 0 || cols <= 0 || aptr == NULL || lda < rows || bptr == NULL || ldb < rows || C == NULL || ldc < rows )
 {
  bli_print_msg( " Invalid function parameters bli_soMatAdd_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 for ( i=0; i<cols; i++ )
 {
  for(j=0; j<rows; j++)
  {
   C[j] = ((alpha * aptr[j]) + (beta * bptr[j]));
  }
  aptr += lda;
  bptr += ldb;
  C += ldc;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

static dim_t bli_doMatAdd_cn(dim_t rows,dim_t cols,const double alpha,double* aptr,dim_t lda,const double beta,double* bptr,dim_t ldb,double* C,dim_t ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 if ( rows <= 0 || cols <= 0 || aptr == NULL || lda < rows || bptr == NULL || ldb < rows || C == NULL || ldc < rows )
 {
  bli_print_msg( " Invalid function parameters bli_doMatAdd_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 for ( i=0; i<cols; i++ )
 {
  for(j=0; j<rows; j++)
  {
   C[j] = ((alpha * aptr[j]) + (beta * bptr[j]));
  }
  aptr += lda;
  bptr += ldb;
  C += ldc;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

static dim_t bli_coMatAdd_cn(dim_t rows,dim_t cols,const scomplex alpha,scomplex* aptr,dim_t lda,const scomplex beta,scomplex* bptr,dim_t ldb,scomplex* C,dim_t ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 if ( rows <= 0 || cols <= 0 || aptr == NULL || lda < rows || bptr == NULL || ldb < rows || C == NULL || ldc < rows )
 {
  bli_print_msg( " Invalid function parameters bli_coMatAdd_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 for ( i=0; i<cols; i++ )
 {
  for(j=0; j<rows; j++)
  {
   //C[j] = ((alpha * aptr[j]) + (beta * bptr[j]));
   C[j].real = (( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) ) + ( (beta.real * bptr[j].real ) - ( beta.imag * bptr[j].imag ) ));
   C[j].imag = (( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) ) + ( (beta.real * bptr[j].imag ) + ( beta.imag * bptr[j].real ) ));
  }
  aptr += lda;
  bptr += ldb;
  C += ldc;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

static dim_t bli_zoMatAdd_cn(dim_t rows,dim_t cols,const dcomplex alpha,dcomplex* aptr,dim_t lda,const dcomplex beta,dcomplex* bptr,dim_t ldb,dcomplex* C,dim_t ldc)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 if ( rows <= 0 || cols <= 0 || aptr == NULL || lda < rows || bptr == NULL || ldb < rows || C == NULL || ldc < rows )
 {
  bli_print_msg( " Invalid function parameters bli_zoMatAdd_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 for ( i=0; i<cols; i++ )
 {
  for(j=0; j<rows; j++)
  {
   C[j].real = (( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) ) + ( (beta.real * bptr[j].real ) - ( beta.imag * bptr[j].imag ) ));
   C[j].imag = (( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) ) + ( (beta.real * bptr[j].imag ) + ( beta.imag * bptr[j].real ) ));
  }
  aptr += lda;
  bptr += ldb;
  C += ldc;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}
#endif
