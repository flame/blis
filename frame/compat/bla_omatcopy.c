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

static dim_t bli_soMatCopy_cn(dim_t rows,dim_t cols,const float alpha,const float* a,dim_t lda,float* b,dim_t ldb);

static dim_t bli_soMatCopy_ct(dim_t rows,dim_t cols,const float alpha,const float* a,dim_t lda,float* b,dim_t ldb);

static dim_t bli_doMatCopy_cn(dim_t rows,dim_t cols,const double alpha,const double* a,dim_t lda,double* b,dim_t ldb);

static dim_t bli_doMatCopy_ct(dim_t rows,dim_t cols,const double alpha,const double* a,dim_t lda,double* b,dim_t ldb);

static dim_t bli_coMatCopy_cn(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb);

static dim_t bli_coMatCopy_ct(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb);

static dim_t bli_coMatCopy_cr(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb);

static dim_t bli_coMatCopy_cc(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb);

static dim_t bli_zoMatCopy_cn(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb);

static dim_t bli_zoMatCopy_ct(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb);

static dim_t bli_zoMatCopy_cr(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb);

static dim_t bli_zoMatCopy_cc(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb);

void somatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda, float* bptr, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid value of trans parameter in somatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || bptr == NULL || *lda < 1 || *ldb < 1 )
 {
  bli_print_msg( " Invalid function parameter in somatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( *trans == 'n' || *trans == 'N')
 {
  bli_soMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  bli_soMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  bli_soMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_soMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void domatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda, double* bptr, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid value of trans parameter in domatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || bptr == NULL || *lda < 1 || *ldb < 1 )
 {
  bli_print_msg( " Invalid function parameter in domatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( *trans == 'n' || *trans == 'N')
 {
  bli_doMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  bli_doMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  bli_doMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_doMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void comatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda, scomplex* bptr, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid value of trans parameter in comatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || bptr == NULL || *lda < 1 || *ldb < 1 )
 {
  bli_print_msg( " Invalid function parameter in comatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( *trans == 'n' || *trans == 'N')
 {
  bli_coMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  bli_coMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  bli_coMatCopy_cc(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_coMatCopy_cr(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

void zomatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda, dcomplex* bptr, f77_int* ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
 //bli_init_once();
 if ( !(*trans == 'n' || *trans == 'N' ||
  *trans == 't' || *trans == 'T' ||
  *trans == 'c' || *trans == 'C' ||
  *trans == 'r' || *trans == 'R'))
 {
  bli_print_msg( " Invalid value of trans parameter in zomatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid value for trans parameter");
  return ;
 }
 if ( *rows <= 0 || *cols <= 0 || alpha == NULL || aptr == NULL || bptr == NULL || *lda < 1 || *ldb < 1 )
 {
  bli_print_msg( " Invalid function parameter in zomatcopy_() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Invalid function parameters");
  return ;
 }
 if ( *trans == 'n' || *trans == 'N')
 {
  bli_zoMatCopy_cn(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 't' || *trans == 'T')
 {
  bli_zoMatCopy_ct(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'c' || *trans == 'C')
 {
  bli_zoMatCopy_cc(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else if ( *trans == 'r' || *trans == 'R')
 {
  bli_zoMatCopy_cr(*rows,*cols,*alpha,aptr,*lda,bptr,*ldb);
 }
 else
 {
  // do nothing
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
 return ;
}

// suffix cn means - column major & non-trans
static dim_t bli_soMatCopy_cn(dim_t rows,dim_t cols,const float alpha,const float* a,dim_t lda,float* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const float* aptr;
 float* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_soMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }

 aptr = a;
 bptr = b;

 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j] = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha == 1.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j] = aptr[j];
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j] = alpha * aptr[j];
  }
  aptr += lda;
  bptr += ldb;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_doMatCopy_cn(dim_t rows,dim_t cols,const double alpha,const double* a,dim_t lda,double* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const double* aptr;
 double* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_doMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }

 aptr = a;
 bptr = b;

 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j] = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha == 1.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j] = aptr[j];
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j] = alpha * aptr[j];
  }
  aptr += lda;
  bptr += ldb;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_coMatCopy_cn(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const scomplex* aptr;
 scomplex* bptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_coMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;
 bptr = b;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for( i=0; i<cols; i++ )
  {
   for( j=0; j<rows; j++ )
   {
    bptr[j].real = 0.0;
    bptr[j].imag = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = aptr[j].real - aptr[j].imag;
    bptr[j].imag = aptr[j].real + aptr[j].imag;
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j].real = ( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) );
   bptr[j].imag = ( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) );
  }
  aptr += lda;
  bptr += ldb;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cn means - column major & non-trans
static dim_t bli_zoMatCopy_cn(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const dcomplex* aptr;
 dcomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_zoMatCopy_cn() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;
 bptr = b;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = 0.0;
    bptr[j].imag = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = aptr[j].real - aptr[j].imag;
    bptr[j].imag = aptr[j].real + aptr[j].imag;
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j].real = ( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) );
   bptr[j].imag = ( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) );
  }
  aptr += lda;
  bptr += ldb;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix ct means - column major & trans
static dim_t bli_soMatCopy_ct(dim_t rows,dim_t cols,const float alpha,const float* a,dim_t lda,float* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const float* aptr;
 float* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 //if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < cols || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_soMatCopy_ct() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }

 aptr = a;

 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb] = 0.0;
   }
  }
  return(0);
 }

 if ( alpha == 1.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb] = aptr[j];
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb] = alpha * aptr[j];
  }
  aptr += lda;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix ct means - column major & trans
static dim_t bli_doMatCopy_ct(dim_t rows,dim_t cols,const double alpha,const double* a,dim_t lda,double* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const double* aptr;
 double* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_doMatCopy_ct() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }

 aptr = a;

 if ( alpha == 0.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
     bptr[j*ldb] = 0.0;
   }
  }
  return(0);
 }

 if ( alpha == 1.0 )
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb] = aptr[j];
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb] = alpha * aptr[j];
  }
  aptr += lda;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix ct means - column major & trans
static dim_t bli_coMatCopy_ct(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const scomplex* aptr;
 scomplex* bptr;

 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_coMatCopy_ct() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = 0.0;
    bptr[j*ldb].imag = 0.0;
   }
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = aptr[j].real - aptr[j].imag;
    bptr[j*ldb].imag = aptr[j].real + aptr[j].imag;
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb].real = ( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) );
   bptr[j*ldb].imag = ( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) );
  }
  aptr += lda;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix ct means - column major & trans
static dim_t bli_zoMatCopy_ct(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const dcomplex* aptr;
 dcomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_zoMatCopy_ct() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = 0.0;
    bptr[j*ldb].imag = 0.0;
   }
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = aptr[j].real - aptr[j].imag;
    bptr[j*ldb].imag = aptr[j].real + aptr[j].imag;
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb].real = ( (alpha.real * aptr[j].real ) - ( alpha.imag * aptr[j].imag ) );
   bptr[j*ldb].imag = ( (alpha.real * aptr[j].imag ) + ( alpha.imag * aptr[j].real ) );
  }
  aptr += lda;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cr means - column major & conjugate
static dim_t bli_coMatCopy_cr(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const scomplex* aptr;
 scomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_coMatCopy_cr() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;
 bptr = b;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = 0.0;
    bptr[j].imag = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = aptr[j].real + aptr[j].imag;
    bptr[j].imag = aptr[j].real - aptr[j].imag;
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j].real = ( (alpha.real * aptr[j].real ) + ( alpha.imag * aptr[j].imag ) );
   bptr[j].imag = ( (alpha.imag * aptr[j].real ) - ( alpha.real * aptr[j].imag ) );
  }
  aptr += lda;
  bptr += ldb;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cr means - column major & conjugate
static dim_t bli_zoMatCopy_cr(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const dcomplex* aptr;
 dcomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_zoMatCopy_cr() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;
 bptr = b;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = 0.0;
    bptr[j].imag = 0.0;
   }
   bptr += ldb;
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   for(j=0; j<rows; j++)
   {
    bptr[j].real = aptr[j].real + aptr[j].imag;
    bptr[j].imag = aptr[j].real - aptr[j].imag;
   }
   aptr += lda;
   bptr += ldb;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  for(j=0; j<rows; j++)
  {
   bptr[j].real = ( (alpha.real * aptr[j].real ) + ( alpha.imag * aptr[j].imag ) );
   bptr[j].imag = ( (alpha.imag * aptr[j].real ) - ( alpha.real * aptr[j].imag ) );
  }
  aptr += lda;
  bptr += ldb;
 }

 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cc means - column major & conjugate trans
static dim_t bli_coMatCopy_cc(dim_t rows,dim_t cols,const scomplex alpha,const scomplex* a,dim_t lda,scomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const scomplex* aptr;
 scomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_coMatCopy_cc() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = 0.0;
    bptr[j*ldb].imag = 0.0;
   }
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = aptr[j].real + aptr[j].imag;
    bptr[j*ldb].imag = aptr[j].real - aptr[j].imag;
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb].real = ( (alpha.real * aptr[j].real ) + ( alpha.imag * aptr[j].imag ) );
   bptr[j*ldb].imag = ( (alpha.imag * aptr[j].real ) - ( alpha.real * aptr[j].imag ) );
  }
  aptr += lda;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

// suffix cc means - column major & conjugate trans
static dim_t bli_zoMatCopy_cc(dim_t rows,dim_t cols,const dcomplex alpha,const dcomplex* a,dim_t lda,dcomplex* b,dim_t ldb)
{
 AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
 dim_t i,j;
 const dcomplex* aptr;
 dcomplex* bptr;
 if ( rows <= 0 || cols <= 0 || a == NULL || b == NULL || lda < rows || ldb < rows )
 {
  bli_print_msg( " Invalid function parameter in bli_zoMatCopy_cc() .", __FILE__, __LINE__ );
  AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "Invalid function parameters");
  return (0);
 }
 aptr = a;

 if ( alpha.real == 0.0 && alpha.imag == 0.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = 0.0;
    bptr[j*ldb].imag = 0.0;
   }
  }
  return(0);
 }

 if ( alpha.real == 1.0 && alpha.imag == 1.0)
 {
  for ( i=0; i<cols ; i++ )
  {
   bptr = &b[i];
   for(j=0; j<rows; j++)
   {
    bptr[j*ldb].real = aptr[j].real + aptr[j].imag;
    bptr[j*ldb].imag = aptr[j].real - aptr[j].imag;
   }
   aptr += lda;
  }
  return(0);
 }

 for ( i=0; i<cols ; i++ )
 {
  bptr = &b[i];
  for(j=0; j<rows; j++)
  {
   bptr[j*ldb].real = ( (alpha.real * aptr[j].real ) + ( alpha.imag * aptr[j].imag ) );
   bptr[j*ldb].imag = ( (alpha.imag * aptr[j].real ) - ( alpha.real * aptr[j].imag ) );
  }
  aptr += lda;
 }
 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
 return(0);
}

#endif
