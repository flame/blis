/*

BLIS
An object-based framework for developing high-performance BLAS-like
libraries.

Copyright (C) 2018, Advanced Micro Devices, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of The University of Texas at Austin nor the names
of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

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
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
#include "immintrin.h"

#define GEMM_BLK_V1 8
#define GEMM_ACCUM_A 1
#define OPT_CACHE_BLOCKING_L1 1
#define REARRANGE_SHFL 0

static void (*fp_blis_strsm_microkernel)( float *ptr_l,
		                            float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);
static void blis_strsm_microkernel( float *ptr_l,
		                            float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);
static void blis_strsm_microkernel_alpha( float *ptr_l,
		                            float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b,
									float alphaVal
								);
static void blis_strsm_microkernel_unitDiag( float *ptr_l,
		                            float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);
static void blis_strsm_microkernel_alpha_unitDiag( float *ptr_l,
		                            float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b,
									float alphaVal
								);
static void trsm_XAtB_block_allSmallSizedMatrices(float *ptr_l, 
								  float *ptr_b, 
								  int numRows_lb, 
								  int numCols_b, 
								  int rs_l, 
								  int rs_b,
								  int cs_l, 
								  int cs_b);
static void trsm_XAtB_block_allSmallSizedMatrices_alpha(float *ptr_l, 
								  float *ptr_b, 
								  int numRows_lb, 
								  int numCols_b, 
								  int rs_l, 
								  int rs_b,
								  int cs_l, 
								  int cs_b,
								  float alphaVal);
static void trsm_XAtB_block_allSmallSizedMatrices_unitDiag(float *ptr_l, 
								  float *ptr_b, 
								  int numRows_lb, 
								  int numCols_b, 
								  int rs_l, 
								  int rs_b,
								  int cs_l, 
								  int cs_b);
static void trsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag(float *ptr_l, 
								  float *ptr_b, 
								  int numRows_lb, 
								  int numCols_b, 
								  int rs_l, 
								  int rs_b,
								  int cs_l, 
								  int cs_b,
								  float alphaVal);
								  
static void (*fp_blis_dtrsm_microkernel)( double *ptr_l,
                  double *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);

static void blis_dtrsm_microkernel( double *ptr_l,
		                            double *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);

static void blis_dtrsm_microkernel_alpha( double *ptr_l,
		                            double *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b,
									double alphaVal
								);

static void blis_dtrsm_microkernel_unitDiag( double *ptr_l,
		                            double *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b
								);

static void blis_dtrsm_microkernel_alpha_unitDiag( double *ptr_l,
		                            double *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
								  int cs_b,
									double alphaVal
								);

static void dtrsm_XAtB_block_allSmallSizedMatrices(double *ptr_l,
								  double *ptr_b,
								  int numRows_lb,
								  int numCols_b,
								  int rs_l,
								  int rs_b,
								  int cs_l,
								  int cs_b);
static void dtrsm_XAtB_block_allSmallSizedMatrices_alpha(double *ptr_l,
								  double *ptr_b,
								  int numRows_lb,
								  int numCols_b,
								  int rs_l,
								  int rs_b,
								  int cs_l,
								  int cs_b,
								  double alphaVal);
static void dtrsm_XAtB_block_allSmallSizedMatrices_unitDiag(double *ptr_l,
								  double *ptr_b,
								  int numRows_lb,
								  int numCols_b,
								  int rs_l,
								  int rs_b,
								  int cs_l,
								  int cs_b);
static void dtrsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag(double *ptr_l,
								  double *ptr_b,
								  int numRows_lb,
								  int numCols_b,
								  int rs_l,
								  int rs_b,
								  int cs_l,
								  int cs_b,
								  double alphaVal);
static void trsm_AutXB_block_allSmallSizedMatrices(float *ptr_l, 
									float *ptr_b, 
									int numRows_lb, 
									int numCols_b, 
									int rs_l, 
									int rs_b,
									int cs_l, 
									int cs_b);
static void trsm_AutXB_block_allSmallSizedMatrices_alpha(float *ptr_l,
									float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b,
									float alpha);
static void trsm_AutXB_block_allSmallSizedMatrices_unitDiag(float *ptr_l,
									float *ptr_b, 
									int numRows_lb, 
									int numCols_b, 
									int rs_l, 
									int rs_b, 
									int cs_l, 
									int cs_b);
static void trsm_AutXB_block_allSmallSizedMatrices_alpha_unitDiag(float *ptr_l,
									float *ptr_b,
									int numRows_lb,
									int numCols_b,
									int rs_l,
									int rs_b,
									int cs_l,
									int cs_b,
									float alpha);
								  
//AX = B;  A is lower triangular; No transpose; single precision
static err_t bli_strsm_small_AlXB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );
//A.'X = B;  A is upper triangular; A has to be transposed; single precision
static err_t bli_strsm_small_AutXB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );

//XA.' = B;  A is lower triangular; A has to be transposed; single precision
static err_t bli_strsm_small_XAltB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );
//AX = B;  A is lower triangular; No transpose; double precision
static err_t bli_dtrsm_small_AlXB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );


//A.'X = B;  A is upper triangular; A has to be transposed; double precision
static err_t bli_dtrsm_small_AutXB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );


//XA.' = B;  A is lower triangular; A has to be transposed; double precision
static err_t bli_dtrsm_small_XAltB
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     );
	 void trsm_block_c(float *ptr_l, float *ptr_b, int blk_height, int blk_width, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b);
/*
* The bli_trsm_small implements unpacked version of TRSM 
* Currently only column-major is supported, A & B are column-major
* Input: A: MxM (triangular matrix)
*        B: MxN matrix
* Output: X: MxN matrix such that AX = alpha*B or XA = alpha*B or A'X = alpha*B or XA' = alpha*B 
* Here the output X is stored in B
* The custom-kernel will be called only when M*(M+N)* sizeof(Matrix Elements) < L3 cache
*/
err_t bli_trsm_small
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING
    return BLIS_NOT_YET_IMPLEMENTED;
#endif

    // If alpha is zero, B matrix will become zero after scaling & hence solution is also zero matrix 
    if (bli_obj_equals(alpha, &BLIS_ZERO))
    {
        return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
    }
    // We have to call matrix scaling if alpha != 1.0
    
    // if row major format return. Check this again.
    if ((bli_obj_row_stride(a) != 1) ||
        (bli_obj_row_stride(b) != 1))
    {
        return BLIS_INVALID_ROW_STRIDE;
    }

    num_t dt = ((*b).info & (0x7 << 0));

    // only float and double datatypes are supported as of now.
    if (dt != BLIS_DOUBLE && dt != BLIS_FLOAT)
    {
	return BLIS_EXPECTED_REAL_DATATYPE;
    }

    // A is expected to be triangular in trsm
    if (!bli_obj_is_upper_or_lower (a))
    {
	return BLIS_EXPECTED_TRIANGULAR_OBJECT;
    }

    // can use other control structs - even can use array of function pointers,
    // indexed by a number with bits formed by f('side', 'uplo', 'transa', dt).
    // In the below implementation, based on the number of finally implemented
    // cases, can move the checks with more cases higher up.
    if (side == BLIS_LEFT)
    {
	if (bli_obj_has_trans(a))
	{
		if (dt == BLIS_DOUBLE)
		{
			if (bli_obj_is_upper(a))
			{
				//A.'X = B;  A is upper triangular; A has to be transposed; double precision
#if 0 // planning to implement this in this iteration
				return bli_dtrsm_small_AutXB(side, alpha, a, b, cntx, cntl);
#else
				return BLIS_NOT_YET_IMPLEMENTED;
#endif
			}
			else
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
		}
		else if (dt == BLIS_FLOAT)
		{
			if (bli_obj_is_upper(a))
			{
				//A.'X = B;  A is upper triangular; A has to be transposed; single precision
				return bli_strsm_small_AutXB(side, alpha, a, b, cntx, cntl);
				//return BLIS_NOT_YET_IMPLEMENTED;
			}
			else
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
		}
	}
	else
	{
		if (dt == BLIS_DOUBLE)
		{
			if (bli_obj_is_upper(a))
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
			else
			{
				//AX = B;  A is lower triangular; No transpose; double precision
				return bli_dtrsm_small_AlXB(side, alpha, a, b, cntx, cntl);
				//return BLIS_NOT_YET_IMPLEMENTED;
			}
		}
		else if (dt == BLIS_FLOAT)
		{
			if (bli_obj_is_upper(a))
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
			else
			{
				//AX = B;  A is lower triangular; No transpose; single precision
				return bli_strsm_small_AlXB(side, alpha, a, b, cntx, cntl);
				//return BLIS_NOT_YET_IMPLEMENTED;
			}
		}
	}
    }
    else
    {
	if (bli_obj_has_trans(a))
	{
		if (dt == BLIS_DOUBLE)
		{
			if (bli_obj_is_upper(a))
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
			else
			{
				//XA.' = B;  A is lower triangular; A has to be transposed; double precision
        		return bli_dtrsm_small_XAltB(side, alpha, a, b, cntx, cntl);
				//return BLIS_NOT_YET_IMPLEMENTED;
        	}
		}
		else if (dt == BLIS_FLOAT)
		{
			if (bli_obj_is_upper(a))
			{
				return BLIS_NOT_YET_IMPLEMENTED;
			}
			else
			{
				//XA.' = B;  A is lower triangular; A has to be transposed; single precision
				//return BLIS_NOT_YET_IMPLEMENTED;
				return bli_strsm_small_XAltB(side, alpha, a, b, cntx, cntl);
			}
		}
	}
	else
	{
		return BLIS_NOT_YET_IMPLEMENTED;
	}
    }

    return BLIS_NOT_YET_IMPLEMENTED;
};


static void trsm_small_AlXB (
			      float *A,
			      float *B,
			      int M,
			      int N,
			      int lda,
			      int ldb
			    )			                                  
{
  int i;
  int j;
  int k;

  // Need to incorporate alpha

  for (k = 0; k < M; k++)
    {
      float lkk_inv = 1.0/A[k+k*lda];

      for (j = 0; j < N; j++)
	{
	  B[k + j*ldb] *= lkk_inv;
      
	  for (i = k+1; i < M; i++)
	    {
	      B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
	    }
	}
    }// k -loop

}// end of function


// Test code:
void gemm_small( float *ptr_l,
		 float *ptr_b,
		 int blk_m,
		 int blk_n,
		 float *ptr_gemmOut,
		 int cs_l,
		 int cs_b,
		 int rs_l,
		 int rs_b,
		 float alpha,
		 float beta)
{
  int i, j, k;
 
  for (i = 0; i < blk_m; i++)
    {
      for (j = 0; j < blk_n; j++)
	{
	  float t = 0.0;
	  for (k = 0; k < blk_m; k++)
	    {
	      t += (ptr_l[i*rs_l + k* cs_l] * ptr_b[k*rs_b + j*cs_b]);	     
	    }
	  ptr_gemmOut[i*rs_b + j*cs_b] = beta * ptr_gemmOut[i*rs_b + j*cs_b] + alpha * t;
	}
    }
}

/*
 * AX = Alpha*B, Double precision, A:lower triangular
SUPPORTS MATRIX SIZE OF THE FORM 16X4*i, WHERE i IS AN INTEGER
 */

static err_t bli_dtrsm_small_AlXB (
				    side_t side,
				    obj_t* AlphaObj,
				    obj_t* a,
				    obj_t* b,
				    cntx_t* cntx,
				    cntl_t* cntl
				  )
{
  obj_t alpha, beta; // gemm parameters
  obj_t Ga, Gb, Gc;  // for GEMM
  int m = bli_obj_length(b); // number of rows of matrix B
  int n = bli_obj_width(b);  // number of columns of matrix B

  int lda = bli_obj_col_stride(a); // column stride of A
  int ldb = bli_obj_col_stride(b); // column stride of B

  int rsa = bli_obj_row_stride(a); // row stride of A
  int rsb = bli_obj_row_stride(b); // row stride of B

  int i = 0;
  int j;
  int blk_size = 4;
  int isUnitDiag = bli_obj_has_unit_diag(a);

  double alphaVal;
  double *L =  a->buffer;
  double *B =  b->buffer;

  if (m !=16 || (n&3) != 0)
  {
        return BLIS_NOT_YET_IMPLEMENTED;
  }

  alphaVal = *((double *)AlphaObj->buffer);

  /* Small _GEMM preparation code */
  bli_obj_create( BLIS_DOUBLE, 1, 1, 0, 0, &alpha );
  bli_obj_create( BLIS_DOUBLE, 1, 1, 0, 0, &beta );

  /* B = B - A*B */
  bli_setsc(  -(1.0), 0.0, &alpha );
  bli_setsc( (1.0), 0.0, &beta );

  bli_obj_create_with_attached_buffer( BLIS_DOUBLE, blk_size, blk_size, a->buffer, rsa, lda, &Ga);
  bli_obj_create_with_attached_buffer( BLIS_DOUBLE, blk_size, n, b->buffer, rsb, ldb, &Gb);
  bli_obj_create_with_attached_buffer( BLIS_DOUBLE, blk_size, n, b->buffer, rsb, ldb, &Gc);

  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Ga );
  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Gb );
  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Gc );

  //first block of trsm
  Gb.buffer = (void*)(B + i);

 if (alphaVal != 1)
  {
          if (isUnitDiag == 0)
          {
                        blis_dtrsm_microkernel_alpha((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
                        fp_blis_dtrsm_microkernel = blis_dtrsm_microkernel;
          }
          else
          {
                    blis_dtrsm_microkernel_alpha_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
                        fp_blis_dtrsm_microkernel = blis_dtrsm_microkernel_unitDiag;
          }
      bli_setsc( alphaVal, 0.0, &beta );
  }
  else
  {
          if (isUnitDiag == 0)
          {
                        blis_dtrsm_microkernel((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
                        fp_blis_dtrsm_microkernel = blis_dtrsm_microkernel;
          }
          else
          {
                        blis_dtrsm_microkernel_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
                        fp_blis_dtrsm_microkernel = blis_dtrsm_microkernel_unitDiag;
          }
  }


//gemm update
  for (j = i + blk_size; j < m; j += blk_size) // for rows upto multiple of BLOCK_HEIGHT
  {
      Ga.buffer = (void*)(L + j + i*lda);
      Gc.buffer = (void*)(B + j);
      bli_gemm_small(&alpha, &Ga, &Gb, &beta, &Gc, cntx, cntl ); // Gc = beta*Gc + alpha*Ga *Gb
  }
  bli_setsc( (1.0), 0.0, &beta );

  //trsm of remaining blocks
  for (i = blk_size; i < m; i += blk_size)
  {
          Gb.buffer = (void*)(B + i);

          fp_blis_dtrsm_microkernel((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);


          for (j = i + blk_size; j < m; j += blk_size) // for rows upto multiple of BLOCK_HEIGHT
          {
              Ga.buffer = (void*)(L + j + i*lda);
              Gc.buffer = (void*)(B + j);

              bli_gemm_small(&alpha, &Ga, &Gb, &beta, &Gc, cntx, cntl ); // Gc = beta*Gc + alpha*Ga *Gb
          }

  } // End of for loop - i

  return BLIS_SUCCESS;

}


/*
 * AX = Alpha*B, Single precision, A: lower triangular
 */
static err_t bli_strsm_small_AlXB (
                                    side_t  side,
                                    obj_t*  AlphaObj,
                                    obj_t*  a,
                                    obj_t*  b,
                                    cntx_t* cntx,
                                    cntl_t* cntl
				 )
{
  obj_t alpha, beta; // gemm parameters
  obj_t Ga, Gb, Gc;  // for GEMM
  int m = bli_obj_length(b); // number of rows of matrix B
  int n = bli_obj_width(b);  // number of columns of matrix B

  int lda = bli_obj_col_stride(a); // column stride of A
  int ldb = bli_obj_col_stride(b); // column stride of B

  int rsa = bli_obj_row_stride(a); // row stride of A
  int rsb = bli_obj_row_stride(b); // row stride of B

  int i = 0;
  int j;
  int blk_size = 8;
  int isUnitDiag = bli_obj_has_unit_diag(a);

  float alphaVal;
  float *L =  a->buffer;
  float *B =  b->buffer;

  if (m != 16 || (n%8) != 0)
  {
	return BLIS_NOT_YET_IMPLEMENTED;
  }
  if ( (m*(m + n)) > BLIS_SMALL_MATRIX_THRES_TRSM )
  {
  	return BLIS_NOT_YET_IMPLEMENTED;
  }

  alphaVal = *((float *)bli_obj_buffer_for_const(BLIS_FLOAT, AlphaObj));

  /* Small _GEMM preparation code */
  bli_obj_create( BLIS_FLOAT, 1, 1, 0, 0, &alpha );
  bli_obj_create( BLIS_FLOAT, 1, 1, 0, 0, &beta );

  /* B = B - A*B */
  bli_setsc(  -(1.0), 0.0, &alpha );
  bli_setsc( (1.0), 0.0, &beta );

 
  bli_obj_create_with_attached_buffer( BLIS_FLOAT, blk_size, blk_size, a->buffer, rsa, lda, &Ga);
  bli_obj_create_with_attached_buffer( BLIS_FLOAT, blk_size, n, b->buffer, rsb, ldb, &Gb);
  bli_obj_create_with_attached_buffer( BLIS_FLOAT, blk_size, n, b->buffer, rsb, ldb, &Gc);

  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Ga );
  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Gb );
  bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &Gc );

  //first block of trsm
  Gb.buffer = (void*)(B + i);
  
  //trsm of first 8xn block
  if (alphaVal != 1)
  {
	  if (isUnitDiag == 0)
	  {
			blis_strsm_microkernel_alpha((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
			fp_blis_strsm_microkernel = blis_strsm_microkernel;
	  }
	  else
	  {
		    blis_strsm_microkernel_alpha_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
			fp_blis_strsm_microkernel = blis_strsm_microkernel_unitDiag;
	  }
      bli_setsc( alphaVal, 0.0, &beta );
  }
  else
  {
	  if (isUnitDiag == 0)
	  {
			blis_strsm_microkernel((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
			fp_blis_strsm_microkernel = blis_strsm_microkernel;
	  }
	  else
	  {
		   	blis_strsm_microkernel_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
		  	fp_blis_strsm_microkernel = blis_strsm_microkernel_unitDiag;
	  }
  }

  //gemm update
  for (j = i + blk_size; j < m; j += blk_size) // for rows upto multiple of BLOCK_HEIGHT
  {
      Ga.buffer = (void*)(L + j + i*lda);
      Gc.buffer = (void*)(B + j);

      bli_gemm_small(&alpha, &Ga, &Gb, &beta, &Gc, cntx, cntl ); // Gc = beta*Gc + alpha*Ga *Gb
  }

  //trsm of remaining blocks
  for (i = blk_size; i < m; i += blk_size)
  {
	  Gb.buffer = (void*)(B + i);

	  fp_blis_strsm_microkernel((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);

	  for (j = i + blk_size; j < m; j += blk_size) // for rows upto multiple of BLOCK_HEIGHT
	  {
	      Ga.buffer = (void*)(L + j + i*lda);
	      Gc.buffer = (void*)(B + j);

	      bli_gemm_small(&alpha, &Ga, &Gb, &beta, &Gc, cntx, cntl ); // Gc = beta*Gc + alpha*Ga *Gb
	  }

  } // End of for loop - i

  return BLIS_SUCCESS;
}

void trsm_block_c(float *ptr_l, float *ptr_b, int blk_height, int blk_width, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	int i, j, k, l;
	float inv_l;

	inv_l = 1.0 / *ptr_l;

	for (j = 0; j < numCols_b; j += blk_width)
	{
		for (l = j; l < (j+blk_width); l++)
		{
			ptr_b[l*cs_b] = ptr_b[l*cs_b] * inv_l;
		}

		for (i = 1; i < blk_height; i++)
		{
			for (l = j; l < (j+blk_width); l++)
			{
				for (k = 0; k < i; k++)
				{
					ptr_b[i*rs_b + l*cs_b] -= (ptr_b[k*rs_b + l*cs_b] * ptr_l[i*rs_l + k*cs_l]);
				}
				ptr_b[i*rs_b + l*cs_b] = ptr_b[i*rs_b + l*cs_b] / ptr_l[i*rs_l + i*cs_l];
			}
		}
	}
}


/*
 * XA' = Alpha*B, Double precision, A:lower triangular
 */

static err_t bli_dtrsm_small_XAltB(
                                    side_t  side,
                                    obj_t*  AlphaObj,
                                    obj_t*  a,
                                    obj_t*  b,
                                    cntx_t* cntx,
                                    cntl_t* cntl
				 )
{

  int m = bli_obj_length(a); // number of rows of matrix B
  int n = bli_obj_length(b);  // number of columns of matrix B

  int lda = bli_obj_col_stride(a); // column stride of A
  int ldb = bli_obj_col_stride(b); // column stride of B

  int rsa = bli_obj_row_stride(a); // row stride of A
  int rsb = bli_obj_row_stride(b); // row stride of B

  int i = 0;
  int isUnitDiag = bli_obj_has_unit_diag(a);

  double alphaVal;
  double *L =  a->buffer;
  double *B =  b->buffer;

  if ((m%4) != 0 || (n%4) != 0)
  {
	return BLIS_NOT_YET_IMPLEMENTED;
  }
  if ( n > 64 || (m*(m + n)) > BLIS_SMALL_MATRIX_THRES_TRSM )
  {
  	return BLIS_NOT_YET_IMPLEMENTED;
  }
  alphaVal = *((double *)AlphaObj->buffer);
  if (alphaVal != 1)
  {
	  if (isUnitDiag == 0)
	  {
			dtrsm_XAtB_block_allSmallSizedMatrices_alpha((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
	  else
	  {
			dtrsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
  }
  else
  {
	  if (isUnitDiag == 0)
	  {
			dtrsm_XAtB_block_allSmallSizedMatrices((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
	  else
	  {
		  	dtrsm_XAtB_block_allSmallSizedMatrices_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
  }
  return BLIS_SUCCESS;

}


/*
 * XA' = Alpha*B, Single precision, A: lower triangular
 */
static err_t bli_strsm_small_XAltB(
                                    side_t  side,
                                    obj_t*  AlphaObj,
                                    obj_t*  a,
                                    obj_t*  b,
                                    cntx_t* cntx,
                                    cntl_t* cntl
				 )
{
  int m = bli_obj_length(a); // number of rows of matrix B
  int n = bli_obj_length(b);  // number of columns of matrix B

  int lda = bli_obj_col_stride(a); // column stride of A
  int ldb = bli_obj_col_stride(b); // column stride of B

  int rsa = bli_obj_row_stride(a); // row stride of A
  int rsb = bli_obj_row_stride(b); // row stride of B

  int i = 0;
  int isUnitDiag = bli_obj_has_unit_diag(a);

  float alphaVal;
  float *L =  a->buffer;
  float *B =  b->buffer;
 
  if ((m%8) != 0 || (n%8) != 0)
  {
	return BLIS_NOT_YET_IMPLEMENTED;
  }
  if ( n > 128 || (m*(m + n)) > BLIS_SMALL_MATRIX_THRES_TRSM )
  {
  	return BLIS_NOT_YET_IMPLEMENTED;
  }

  alphaVal = *((float *)bli_obj_buffer_for_const(BLIS_FLOAT, AlphaObj));
 
  if (alphaVal != 1)
  {
	  if (isUnitDiag == 0)
	  {
			trsm_XAtB_block_allSmallSizedMatrices_alpha((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
	  else
	  {
			trsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
  }
  else
  {
	  if (isUnitDiag == 0)
	  {
			trsm_XAtB_block_allSmallSizedMatrices((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
	  else
	  {
		  	trsm_XAtB_block_allSmallSizedMatrices_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
  }
  return BLIS_SUCCESS;
}

/*
 * A'X = Alpha*B, Single precision, A: upper triangular
 */
static err_t bli_strsm_small_AutXB(
                                    side_t  side,
                                    obj_t*  AlphaObj,
                                    obj_t*  a,
                                    obj_t*  b,
                                    cntx_t* cntx,
                                    cntl_t* cntl
								  )
{
  int m = bli_obj_width(a);  // number of rows of matrix A (since At, so width is taken)
  int n = bli_obj_width(b);  // number of columns of matrix B

  int lda = bli_obj_col_stride(a); // column stride of A
  int ldb = bli_obj_col_stride(b); // column stride of B

  int rsa = bli_obj_row_stride(a); // row stride of A
  int rsb = bli_obj_row_stride(b); // row stride of B

  int i = 0;
  int isUnitDiag = bli_obj_has_unit_diag(a);

  float alphaVal;
  float *L =  a->buffer;
  float *B =  b->buffer;
 
  if ((m%8) != 0 || (n%8) != 0)
  {
	return BLIS_NOT_YET_IMPLEMENTED;
  }
  if ( m > 64 || n > 128 || (m*(m + n)) > BLIS_SMALL_MATRIX_THRES_TRSM )
  {
  	return BLIS_NOT_YET_IMPLEMENTED;
  }

  alphaVal = *((float *)bli_obj_buffer_for_const(BLIS_FLOAT, AlphaObj));

  if (alphaVal != 1)
  {
	  if (isUnitDiag == 0)
	  {
			trsm_AutXB_block_allSmallSizedMatrices_alpha((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
	  else
	  {
			trsm_AutXB_block_allSmallSizedMatrices_alpha_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb, alphaVal);
	  }
  }
  else
  {
	  if (isUnitDiag == 0)
	  {
			trsm_AutXB_block_allSmallSizedMatrices((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
	  else
	  {
		  	trsm_AutXB_block_allSmallSizedMatrices_unitDiag((L + i * lda + i), (B + i), m, n, rsa, rsb, lda, ldb);
	  }
  }
  return BLIS_SUCCESS;
}
/*
* AX=B A=LOWER TRIANGULAR, NO TRANSPOSE, NON-UNITDIAGONAL
* ALPHA != 1;
*/
static void blis_dtrsm_microkernel_alpha(double *ptr_l,
					 double *ptr_b,
					 int numRows_lb,
					 int numCols_b,
					 int rs_l,
					 int rs_b,
					 int cs_l,
					 int cs_b,
					 double alphaVal
					)
{
	double ones = 1.0;
	int j;
	int cs_b_offset[2];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[4];
	__m256d mat_a_cols[4];
	__m256d mat_a_cols_rearr[10];
	__m256d mat_a_diag_inv[4];
	__m256d reciprocal_diags;
	__m256d alphaReg;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];

	reciprocal_diags = _mm256_broadcast_sd((double const *)&ones);
  alphaReg = _mm256_broadcast_sd((double const *)&alphaVal);

  //read first set of 4x4 block of B into registers
  mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
  mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
  //_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
  mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
  //_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
  mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

  //1st col
  mat_a_cols_rearr[0] = _mm256_broadcast_sd((double const *)(ptr_l+0));
  mat_a_cols_rearr[1] = _mm256_broadcast_sd((double const *)(ptr_l+1));
  mat_a_cols_rearr[3] = _mm256_broadcast_sd((double const *)(ptr_l+2));
  mat_a_cols_rearr[6] = _mm256_broadcast_sd((double const *)(ptr_l+3));

	//2nd col
  ptr_l += cs_l;
  mat_a_cols_rearr[2] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
  mat_a_cols_rearr[4] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[7] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//3rd col
  ptr_l += cs_l;
  mat_a_cols_rearr[5] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[8] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//4th col
  ptr_l += cs_l;
  mat_a_cols_rearr[9] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	numCols_b -= 4; // blk_width = 4

	//compute reciprocals of L(i,i) and broadcast in registers
  mat_a_diag_inv[0] = _mm256_unpacklo_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
  mat_a_diag_inv[1] = _mm256_unpacklo_pd(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);

	mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);
	reciprocal_diags = _mm256_div_pd(reciprocal_diags, mat_a_diag_inv[0]);

	for(j = 0;j < numCols_b; j += 4)
	{
    ptr_b_dup = ptr_b;
    /*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

    ////unpacklow////
    mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange low elements
		mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
		mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);

		mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], alphaReg);
    mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], alphaReg);

		////unpackhigh////
    mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange high elements
		mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
		mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

		mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], alphaReg);
    mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], alphaReg);
		//extract a00
		mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags, 0x00);
    mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
    mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], mat_a_diag_inv[0]);

		//extract diag a11 from a
    mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags, 0x03);
    mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

    //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
    mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
    mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], mat_a_diag_inv[1]);

    //extract diag a22 from a
    mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags, 0x00);
    mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

    //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
    mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], mat_a_diag_inv[2]);

    //extract diag a33 from a
    mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags, 0x0C);
    mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

    //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
    mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], mat_a_diag_inv[3]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
    mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange low elements
		mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
		mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

		 ////unpackhigh////
    mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);

    mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange high elements
		mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
		mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

		//Read next set of B columns
		ptr_b += (cs_b+cs_b_offset[1]);
		mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
    mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
    mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
    mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

		//Store the computed B columns
    _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
    _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

	}
	//Last block trsm processing

	ptr_b_dup = ptr_b;
	/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

  ////unpacklow////
  mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange low elements
	mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
	mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);

	mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], alphaReg);
  mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], alphaReg);

	////unpackhigh////
  mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange high elements
	mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
	mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

	mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], alphaReg);
  mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], alphaReg);
	//extract a00
	mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags, 0x00);
  mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
  mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], mat_a_diag_inv[0]);

	//extract diag a11 from a
  mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags, 0x03);
  mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

  //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
  mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
  mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], mat_a_diag_inv[1]);

  //extract diag a22 from a
  mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags, 0x00);
  mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

  //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
  mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], mat_a_diag_inv[2]);

  //extract diag a33 from a
  mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags, 0x0C);
  mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

  //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
   mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], mat_a_diag_inv[3]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
  mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange low elements
	mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
	mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

	////unpackhigh////
  mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange high elements
	mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
	mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

	 //Store the computed B columns
  _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
  _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

}
/*
*AX=B A=LOWER TRIANGULAR, NO TRANSPOSE, UNITDIAGONAL
*ALPHA != 1;
*/
static void blis_dtrsm_microkernel_alpha_unitDiag(double *ptr_l,
						  double *ptr_b,
						  int numRows_lb,
						  int numCols_b,
						  int rs_l,
						  int rs_b,
						  int cs_l,
						  int cs_b,
						  double alphaVal
						 )
{

	int j;
	int cs_b_offset[2];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[4];
	__m256d mat_a_cols[4];
	__m256d mat_a_cols_rearr[10];
	__m256d alphaReg;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];

  alphaReg = _mm256_broadcast_sd((double const *)&alphaVal);
  // ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

  //read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
  mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
  mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
  //_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
  mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
  //_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
  mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));
  //1st col
  mat_a_cols_rearr[0] = _mm256_broadcast_sd((double const *)(ptr_l+0));
  mat_a_cols_rearr[1] = _mm256_broadcast_sd((double const *)(ptr_l+1));
  mat_a_cols_rearr[3] = _mm256_broadcast_sd((double const *)(ptr_l+2));
  mat_a_cols_rearr[6] = _mm256_broadcast_sd((double const *)(ptr_l+3));

	//2nd col
  ptr_l += cs_l;
  mat_a_cols_rearr[2] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
  mat_a_cols_rearr[4] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[7] = _mm256_broadcast_sd((double const *)(ptr_l + 3));
  //3rd col
  ptr_l += cs_l;
  mat_a_cols_rearr[5] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[8] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//4th col
  ptr_l += cs_l;
  mat_a_cols_rearr[9] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	numCols_b -= 4; // blk_width = 4

	for(j = 0;j < numCols_b; j += 4)
	{
    ptr_b_dup = ptr_b;
		/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

    ////unpacklow////
    mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange low elements
		mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
		mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);

		mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], alphaReg);
    mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], alphaReg);

		////unpackhigh////
    mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange high elements
		mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
		mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

		mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], alphaReg);
    mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], alphaReg);

    //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
    mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

    //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

    //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)


		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
    mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange low elements
		mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
		mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

		////unpackhigh////
    mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange high elements
		mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
		mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

		//Read next set of B columns
		ptr_b += (cs_b+cs_b_offset[1]);
		mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
    mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
    mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
    mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

		//Store the computed B columns
    _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
    _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

	}
	 //Last block trsm processing

		ptr_b_dup = ptr_b;
	/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

   ////unpacklow////
  mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange low elements
	mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
	mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);

	mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], alphaReg);
  mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], alphaReg);

	///unpackhigh////
  mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange high elements
	mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
	mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

	mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], alphaReg);
  mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], alphaReg);

  //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
  mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

  //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

  //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
  mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange low elements
	mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
	mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

	////unpackhigh////
  mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange high elements
	mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
	mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

	//Store the computed B columns
  _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
  _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

}
/*
*AX = B A= LOWERTRIANGULAR, NO TRANSPOSE, NON-UNITDIAGONAL
*ALPHA = 1
*/
static void blis_dtrsm_microkernel(double *ptr_l,
				   double *ptr_b,
				   int numRows_lb,
				   int numCols_b,
				   int rs_l,
				   int rs_b,
				   int cs_l,
				   int cs_b
				  )
{
	double ones = 1.0;
	int j;
	int cs_b_offset[2];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[4];
	__m256d mat_a_cols[4];
	__m256d mat_a_cols_rearr[10];
	__m256d mat_a_diag_inv[4];
	__m256d reciprocal_diags;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];

  reciprocal_diags = _mm256_broadcast_sd((double const *)&ones);

  // ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

   //read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
   mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
   //_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
   //row2 = (cs_l << 1);
   //row4 = (cs_l << 2);
   mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
   //_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
   mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
   //_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
   mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

	 //1st col
   mat_a_cols_rearr[0] = _mm256_broadcast_sd((double const *)(ptr_l+0));
   mat_a_cols_rearr[1] = _mm256_broadcast_sd((double const *)(ptr_l+1));
   mat_a_cols_rearr[3] = _mm256_broadcast_sd((double const *)(ptr_l+2));
   mat_a_cols_rearr[6] = _mm256_broadcast_sd((double const *)(ptr_l+3));

	 //2nd col
   ptr_l += cs_l;
   mat_a_cols_rearr[2] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
   mat_a_cols_rearr[4] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
   mat_a_cols_rearr[7] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//3rd col
   ptr_l += cs_l;
   mat_a_cols_rearr[5] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
   mat_a_cols_rearr[8] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	 //4th col
   ptr_l += cs_l;
   mat_a_cols_rearr[9] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	 numCols_b -= 4; // blk_width = 4

	 //compute reciprocals of L(i,i) and broadcast in registers
   mat_a_diag_inv[0] = _mm256_unpacklo_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
   mat_a_diag_inv[1] = _mm256_unpacklo_pd(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);

  mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);
	reciprocal_diags = _mm256_div_pd(reciprocal_diags, mat_a_diag_inv[0]);

	for(j = 0;j < numCols_b; j += 4)
	{
    ptr_b_dup = ptr_b;
		/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

    ////unpacklow////
    mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange low elements
		mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
		mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);


		////unpackhigh////
    mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange high elements
		mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
		mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

		//extract a00
		mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags, 0x00);
    mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
    mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], mat_a_diag_inv[0]);

		//extract diag a11 from a
    mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags, 0x03);
    mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

    //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
    mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
    mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], mat_a_diag_inv[1]);

    //extract diag a22 from a
    mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags, 0x00);
    mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

    //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

		 //Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
     mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], mat_a_diag_inv[2]);

     //extract diag a33 from a
     mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags, 0x0C);
     mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

     //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
     mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

		 //Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
     mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], mat_a_diag_inv[3]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
    mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange low elements
		mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
		mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

		////unpackhigh////
    mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange high elements
		mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
		mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

		//Read next set of B columns
		ptr_b += (cs_b+cs_b_offset[1]);
		mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
    mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
    mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
    mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

		//Store the computed B columns
    _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
    _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

	}
  //Last block trsm processing

  ptr_b_dup = ptr_b;
	/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

  ////unpacklow////
  mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange low elements
	mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
	mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);

	////unpackhigh////
  mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange high elements
	mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
	mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);

	//extract a00
	mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags, 0x00);
  mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
  mat_b_rearr[0] = _mm256_mul_pd(mat_b_rearr[0], mat_a_diag_inv[0]);

	//extract diag a11 from a
  mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags, 0x03);
  mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

  //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
  mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
  mat_b_rearr[1] = _mm256_mul_pd(mat_b_rearr[1], mat_a_diag_inv[1]);

  //extract diag a22 from a
  mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags, 0x00);
  mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

  //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
  mat_b_rearr[2] = _mm256_mul_pd(mat_b_rearr[2], mat_a_diag_inv[2]);

  //extract diag a33 from a
  mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags, 0x0C);
  mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

  //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
  mat_b_rearr[3] = _mm256_mul_pd(mat_b_rearr[3], mat_a_diag_inv[3]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
  mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange low elements
	mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
	mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

	////unpackhigh////
  mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange high elements
	mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
	mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

	//Store the computed B columns
  _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
  _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);



}
/*
*AX = B A=LOWER TRIANGULAR, NO TRANSPOSE, UNITDIAGONAL
*ALPHA = 1
*/
static void blis_dtrsm_microkernel_unitDiag(double *ptr_l,
					    double *ptr_b,
					    int numRows_lb,
					    int numCols_b,
					    int rs_l,
					    int rs_b,
					    int cs_l,
					    int cs_b
					   )
{


	//double ones = 1.0;
	int j;
	int cs_b_offset[2];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[4];
	__m256d mat_a_cols[4];
	__m256d mat_a_cols_rearr[10];

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];

  // ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

  //read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
  mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
  //_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
  //row2 = (cs_l << 1);
  //row4 = (cs_l << 2);
  mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
  //_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
  mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
  //_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
  mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

	//1st col
  mat_a_cols_rearr[0] = _mm256_broadcast_sd((double const *)(ptr_l+0));
  mat_a_cols_rearr[1] = _mm256_broadcast_sd((double const *)(ptr_l+1));
  mat_a_cols_rearr[3] = _mm256_broadcast_sd((double const *)(ptr_l+2));
        mat_a_cols_rearr[6] = _mm256_broadcast_sd((double const *)(ptr_l+3));

	//2nd col
  ptr_l += cs_l;
  mat_a_cols_rearr[2] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
  mat_a_cols_rearr[4] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[7] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//3rd col
  ptr_l += cs_l;
  mat_a_cols_rearr[5] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
  mat_a_cols_rearr[8] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//4th col
  ptr_l += cs_l;
  mat_a_cols_rearr[9] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	numCols_b -= 4; // blk_width = 4


	for(j = 0;j < numCols_b; j += 4)
	{
    ptr_b_dup = ptr_b;
		/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

    ////unpacklow////
    mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange low elements
		mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
		mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);


		////unpackhigh////
    mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
    mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

		//rearrange high elements
		mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
		mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);


    //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
    mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

    //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
    mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)


    //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
    mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)


		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
    mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange low elements
		mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
		mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

		////unpackhigh////
    mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
    mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

		//rearrange high elements
		mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
		mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

		//Read next set of B columns
		ptr_b += (cs_b+cs_b_offset[1]);
		mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b);
    mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + (cs_b)));
    mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0]));
    mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1]));

		//Store the computed B columns
    _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
    _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
    _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);

	}
  //Last block trsm processing

	ptr_b_dup = ptr_b;
	/*Shuffle to rearrange/transpose 8x4 block of B into contiguous row-wise registers*/

  ////unpacklow////
  mat_b_rearr[1] = _mm256_unpacklo_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_rearr[3] = _mm256_unpacklo_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange low elements
	mat_b_rearr[0] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x20);
	mat_b_rearr[2] = _mm256_permute2f128_pd(mat_b_rearr[1],mat_b_rearr[3],0x31);


	////unpackhigh////
  mat_b_col[0] = _mm256_unpackhi_pd(mat_b_col[0], mat_b_col[1]);
  mat_b_col[1] = _mm256_unpackhi_pd(mat_b_col[2], mat_b_col[3]);

	//rearrange high elements
	mat_b_rearr[1] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x20);
	mat_b_rearr[3] = _mm256_permute2f128_pd(mat_b_col[0],mat_b_col[1],0x31);


  //(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
  mat_b_rearr[1] = _mm256_fnmadd_pd(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)

  //(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
  mat_b_rearr[2] = _mm256_fnmadd_pd(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)

  //(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
  mat_b_rearr[3] = _mm256_fnmadd_pd(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
  mat_a_cols[1] = _mm256_unpacklo_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_a_cols[3] = _mm256_unpacklo_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange low elements
	mat_a_cols[0] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x20);
	mat_a_cols[2] = _mm256_permute2f128_pd(mat_a_cols[1],mat_a_cols[3],0x31);

	////unpackhigh////
  mat_b_rearr[0] = _mm256_unpackhi_pd(mat_b_rearr[0], mat_b_rearr[1]);
  mat_b_rearr[1] = _mm256_unpackhi_pd(mat_b_rearr[2], mat_b_rearr[3]);

	//rearrange high elements

	mat_a_cols[1] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x20);
	mat_a_cols[3] = _mm256_permute2f128_pd(mat_b_rearr[0],mat_b_rearr[1],0x31);

	//Store the computed B columns
  _mm256_storeu_pd((double *)ptr_b_dup, mat_a_cols[0]);
  _mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
  _mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);


}
///////////////////////////// AX=B ///////////////////////////////
static void blis_strsm_microkernel_alpha(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alphaVal)
{
	float ones = 1.0;
	int j;
	int cs_b_offset[6];
	//int row2, row4, row6;
	float *ptr_b_dup;

	//70 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_cols[8];
	__m256 mat_a_cols_rearr[36];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags;
	__m256 alphaReg;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);
	reciprocal_diags = _mm256_broadcast_ss((float const *)&ones);
	alphaReg = _mm256_broadcast_ss((float const *)&alphaVal);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
	mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
	//_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
	//row2 = (cs_l << 1);
	//row4 = (cs_l << 2);
	mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
	//_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
	mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
	//_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
	mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
	//_mm_prefetch((char*)(ptr_l + row2 + cs_l), _MM_HINT_T0);
	//row6 = row2 + row4;
	mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
	//_mm_prefetch((char*)(ptr_l + row4), _MM_HINT_T0);
	mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
	//_mm_prefetch((char*)(ptr_l + row4 + cs_l), _MM_HINT_T0);
	mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
	//_mm_prefetch((char*)(ptr_l + row6), _MM_HINT_T0);
	mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));
	//_mm_prefetch((char*)(ptr_l + row6 + cs_l), _MM_HINT_T0);

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);

	//read first set of 16x16 block of L, where 16 is the blk_height and 16 is the blk_width  for L
	/*mat_a_cols[0] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[1] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[2] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[3] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[4] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[5] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[6] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[7] = _mm256_loadu_ps((float const *)ptr_l);*/

	//Shuffle to rearrange/transpose 16x16 block of L into contiguous row-wise registers
	//tmpRegs[0] = _mm256_castps256_ps128(mat_a_cols[0]); //zero latency, no instruction added actually.
	//mat_a_cols_rearr[0] = _mm256_broadcastss_ps(tmpRegs[0]);
	//1st col
	mat_a_cols_rearr[0] = _mm256_broadcast_ss((float const *)(ptr_l+0));
	mat_a_cols_rearr[1] = _mm256_broadcast_ss((float const *)(ptr_l+1));
	mat_a_cols_rearr[3] = _mm256_broadcast_ss((float const *)(ptr_l+2));
	mat_a_cols_rearr[6] = _mm256_broadcast_ss((float const *)(ptr_l+3));
	mat_a_cols_rearr[10] = _mm256_broadcast_ss((float const *)(ptr_l+4));
	mat_a_cols_rearr[15] = _mm256_broadcast_ss((float const *)(ptr_l+5));
	mat_a_cols_rearr[21] = _mm256_broadcast_ss((float const *)(ptr_l+6));
	mat_a_cols_rearr[28] = _mm256_broadcast_ss((float const *)(ptr_l+7));
	//2nd col
	ptr_l += cs_l;
	mat_a_cols_rearr[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_cols_rearr[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[7] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[11] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[16] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[22] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[29] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//3rd col
	ptr_l += cs_l;
	mat_a_cols_rearr[5] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[8] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[12] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[17] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[23] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[30] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//4rth col
	ptr_l += cs_l;
	mat_a_cols_rearr[9] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[13] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[18] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[24] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[31] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//5th col
	ptr_l += cs_l;
	mat_a_cols_rearr[14] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[19] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[25] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[32] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//6th col
	ptr_l += cs_l;
	mat_a_cols_rearr[20] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[26] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[33] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[27] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[34] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[35] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	numCols_b -= 8; // blk_width = 8

	//compute reciprocals of L(i,i) and broadcast in registers
	mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
	mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);
	mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_cols_rearr[14], mat_a_cols_rearr[20]);
	mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_cols_rearr[27], mat_a_cols_rearr[35]);

	//mat_a_diag_inv[1] = _mm256_permute_ps(mat_a_diag_inv[1], 0x55);
	//mat_a_diag_inv[3] = _mm256_permute_ps(mat_a_diag_inv[3], 0x55);
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x20);

	//reciprocal of diagnol elements
	reciprocal_diags = _mm256_div_ps(reciprocal_diags, mat_a_diag_inv[0]);

	//Start loop for cols of B to be processed in size of blk_width
	for (j = 0; j < numCols_b; j += 8)
	{
		ptr_b_dup = ptr_b;

		/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
		mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
		
		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
		mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
		mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
		mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
		mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
		mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
		mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
		mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
		mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
		mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		//Read next set of B columns
		ptr_b += (cs_b + cs_b_offset[5]);
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);

	//end loop of cols
	}

	//Last block trsm processing
	ptr_b_dup = ptr_b;

	/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

	////unpacklow////
	mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
	
	mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
	mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
	mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
	
	////unpackhigh////
	mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
	mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
	mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
	mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
	mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
	mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
	mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

	//Merge rearranged high elements into complete rows
	mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
	mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
	mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
	mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

	mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
	mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
	mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
	mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);

	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
	mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

	//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
	mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

	//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
	mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

	//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
	mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

	//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
	mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

	//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
	mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

	//(Row7): FMA operations of b7 with elements of index (7, 0)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
	mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
	mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
	mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
	mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
	mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
	mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
	mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
	mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
	mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
	mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

	////unpackhigh////
	mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

	//Merge rearranged high elements into complete rows
	mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

	//Store the computed B columns
	_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
	_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);

	//end loop of cols
}

static void blis_strsm_microkernel_alpha_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alphaVal)
{
	//float ones = 1.0;
	int j;
	int cs_b_offset[6];
	//int row2, row4, row6;
	float *ptr_b_dup;

	//70 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_cols[8];
	__m256 mat_a_cols_rearr[36];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags;
	__m256 alphaReg;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);
	//reciprocal_diags = _mm256_broadcast_ss((float const *)&ones);
	alphaReg = _mm256_broadcast_ss((float const *)&alphaVal);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
	mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
	//_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
	//row2 = (cs_l << 1);
	//row4 = (cs_l << 2);
	mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
	//_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
	mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
	//_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
	mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
	//_mm_prefetch((char*)(ptr_l + row2 + cs_l), _MM_HINT_T0);
	//row6 = row2 + row4;
	mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
	//_mm_prefetch((char*)(ptr_l + row4), _MM_HINT_T0);
	mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
	//_mm_prefetch((char*)(ptr_l + row4 + cs_l), _MM_HINT_T0);
	mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
	//_mm_prefetch((char*)(ptr_l + row6), _MM_HINT_T0);
	mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));
	//_mm_prefetch((char*)(ptr_l + row6 + cs_l), _MM_HINT_T0);

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);

	//read first set of 16x16 block of L, where 16 is the blk_height and 16 is the blk_width  for L
	/*mat_a_cols[0] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[1] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[2] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[3] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[4] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[5] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[6] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[7] = _mm256_loadu_ps((float const *)ptr_l);*/

	//Shuffle to rearrange/transpose 16x16 block of L into contiguous row-wise registers
	//tmpRegs[0] = _mm256_castps256_ps128(mat_a_cols[0]); //zero latency, no instruction added actually.
	//mat_a_cols_rearr[0] = _mm256_broadcastss_ps(tmpRegs[0]);
	//1st col
	mat_a_cols_rearr[0] = _mm256_broadcast_ss((float const *)(ptr_l+0));
	mat_a_cols_rearr[1] = _mm256_broadcast_ss((float const *)(ptr_l+1));
	mat_a_cols_rearr[3] = _mm256_broadcast_ss((float const *)(ptr_l+2));
	mat_a_cols_rearr[6] = _mm256_broadcast_ss((float const *)(ptr_l+3));
	mat_a_cols_rearr[10] = _mm256_broadcast_ss((float const *)(ptr_l+4));
	mat_a_cols_rearr[15] = _mm256_broadcast_ss((float const *)(ptr_l+5));
	mat_a_cols_rearr[21] = _mm256_broadcast_ss((float const *)(ptr_l+6));
	mat_a_cols_rearr[28] = _mm256_broadcast_ss((float const *)(ptr_l+7));
	//2nd col
	ptr_l += cs_l;
	mat_a_cols_rearr[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_cols_rearr[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[7] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[11] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[16] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[22] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[29] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//3rd col
	ptr_l += cs_l;
	mat_a_cols_rearr[5] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[8] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[12] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[17] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[23] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[30] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//4rth col
	ptr_l += cs_l;
	mat_a_cols_rearr[9] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[13] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[18] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[24] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[31] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//5th col
	ptr_l += cs_l;
	mat_a_cols_rearr[14] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[19] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[25] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[32] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//6th col
	ptr_l += cs_l;
	mat_a_cols_rearr[20] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[26] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[33] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[27] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[34] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//8th col
	//ptr_l += cs_l;
	//mat_a_cols_rearr[35] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	numCols_b -= 8; // blk_width = 8

	//compute reciprocals of L(i,i) and broadcast in registers
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_cols_rearr[14], mat_a_cols_rearr[20]);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_cols_rearr[27], mat_a_cols_rearr[35]);

	//mat_a_diag_inv[1] = _mm256_permute_ps(mat_a_diag_inv[1], 0x55);
	//mat_a_diag_inv[3] = _mm256_permute_ps(mat_a_diag_inv[3], 0x55);
	//mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);
	//mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);
	//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x20);

	//reciprocal of diagnol elements
	//reciprocal_diags = _mm256_div_ps(reciprocal_diags, mat_a_diag_inv[0]);

	//Start loop for cols of B to be processed in size of blk_width
	for (j = 0; j < numCols_b; j += 8)
	{
		ptr_b_dup = ptr_b;

		/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
		mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
		
		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//extract diag a00 from a
		//mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
		//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		//mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);

		//extract diag a11 from a
		//mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
		//mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		//mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		//mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		//mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		//mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		//mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		//mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		//mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		//mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
		//mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		//mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		//mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
		//mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		//mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		//mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		//mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		//mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		//mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		//mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		//mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
		mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
		mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
		mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
		mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
		mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
		mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
		mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
		mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
		mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		//Read next set of B columns
		ptr_b += (cs_b + cs_b_offset[5]);
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);

	//end loop of cols
	}

	//Last block trsm processing
	ptr_b_dup = ptr_b;

	/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

	////unpacklow////
	mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
	
	mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
	mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
	mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
	
	////unpackhigh////
	mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
	mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
	mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
	mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
	mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
	mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

	//extract diag a00 from a
	//mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
	//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
	//mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

	//Merge rearranged high elements into complete rows
	mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
	mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
	mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
	mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

	mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
	mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
	mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
	mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);

	//extract diag a11 from a
	//mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
	//mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
	mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
	//mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

	//extract diag a22 from a
	//mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	//mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

	//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
	//mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

	//extract diag a33 from a
	//mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	//mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

	//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
	//mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

	//extract diag a44 from a
	//mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
	//mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

	//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
	//mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

	//extract diag a55 from a
	//mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
	//mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

	//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
	//mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

	//extract diag a66 from a
	//mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	//mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

	//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
	//mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

	//extract diag a77 from a
	//mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	//mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

	//(Row7): FMA operations of b7 with elements of index (7, 0)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
	//mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
	mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
	mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
	mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
	mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
	mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
	mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
	mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
	mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
	mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

	////unpackhigh////
	mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

	//Merge rearranged high elements into complete rows
	mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

	//Store the computed B columns
	_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
	_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);

	//end loop of cols
}

static void blis_strsm_microkernel_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	//float ones = 1.0;
	int j;
	int cs_b_offset[6];
	//int row2, row4, row6;
	float *ptr_b_dup;

	//70 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_cols[8];
	__m256 mat_a_cols_rearr[36];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);
	//reciprocal_diags = _mm256_broadcast_ss((float const *)&ones);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
	mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
	//_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
	//row2 = (cs_l << 1);
	//row4 = (cs_l << 2);
	mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
	//_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
	mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
	//_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
	mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
	//_mm_prefetch((char*)(ptr_l + row2 + cs_l), _MM_HINT_T0);
	//row6 = row2 + row4;
	mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
	//_mm_prefetch((char*)(ptr_l + row4), _MM_HINT_T0);
	mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
	//_mm_prefetch((char*)(ptr_l + row4 + cs_l), _MM_HINT_T0);
	mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
	//_mm_prefetch((char*)(ptr_l + row6), _MM_HINT_T0);
	mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));
	//_mm_prefetch((char*)(ptr_l + row6 + cs_l), _MM_HINT_T0);

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);

	//read first set of 16x16 block of L, where 16 is the blk_height and 16 is the blk_width  for L
	/*mat_a_cols[0] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[1] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[2] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[3] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[4] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[5] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[6] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[7] = _mm256_loadu_ps((float const *)ptr_l);*/

	//Shuffle to rearrange/transpose 16x16 block of L into contiguous row-wise registers
	//tmpRegs[0] = _mm256_castps256_ps128(mat_a_cols[0]); //zero latency, no instruction added actually.
	//mat_a_cols_rearr[0] = _mm256_broadcastss_ps(tmpRegs[0]);
	//1st col
	mat_a_cols_rearr[0] = _mm256_broadcast_ss((float const *)(ptr_l+0));
	mat_a_cols_rearr[1] = _mm256_broadcast_ss((float const *)(ptr_l+1));
	mat_a_cols_rearr[3] = _mm256_broadcast_ss((float const *)(ptr_l+2));
	mat_a_cols_rearr[6] = _mm256_broadcast_ss((float const *)(ptr_l+3));
	mat_a_cols_rearr[10] = _mm256_broadcast_ss((float const *)(ptr_l+4));
	mat_a_cols_rearr[15] = _mm256_broadcast_ss((float const *)(ptr_l+5));
	mat_a_cols_rearr[21] = _mm256_broadcast_ss((float const *)(ptr_l+6));
	mat_a_cols_rearr[28] = _mm256_broadcast_ss((float const *)(ptr_l+7));
	//2nd col
	ptr_l += cs_l;
	mat_a_cols_rearr[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_cols_rearr[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[7] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[11] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[16] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[22] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[29] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//3rd col
	ptr_l += cs_l;
	mat_a_cols_rearr[5] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[8] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[12] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[17] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[23] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[30] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//4rth col
	ptr_l += cs_l;
	mat_a_cols_rearr[9] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[13] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[18] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[24] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[31] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//5th col
	ptr_l += cs_l;
	mat_a_cols_rearr[14] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[19] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[25] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[32] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//6th col
	ptr_l += cs_l;
	mat_a_cols_rearr[20] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[26] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[33] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[27] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[34] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//8th col
	//ptr_l += cs_l;
	//mat_a_cols_rearr[35] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	numCols_b -= 8; // blk_width = 8

	//compute reciprocals of L(i,i) and broadcast in registers
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_cols_rearr[14], mat_a_cols_rearr[20]);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_cols_rearr[27], mat_a_cols_rearr[35]);

	//mat_a_diag_inv[1] = _mm256_permute_ps(mat_a_diag_inv[1], 0x55);
	//mat_a_diag_inv[3] = _mm256_permute_ps(mat_a_diag_inv[3], 0x55);
	//mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);
	//mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);
	//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x20);

	//reciprocal of diagnol elements
	//reciprocal_diags = _mm256_div_ps(reciprocal_diags, mat_a_diag_inv[0]);

	//Start loop for cols of B to be processed in size of blk_width
	for (j = 0; j < numCols_b; j += 8)
	{
		ptr_b_dup = ptr_b;

		/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//extract diag a00 from a
		//mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
		//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		//mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		//extract diag a11 from a
		//mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
		//mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		//mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		//mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		//mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		//mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		//mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		//mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		//mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		//mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
		//mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		//mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		//mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
		//mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		//mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		//mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		//mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		//mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		//mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		//mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		//mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
		mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
		mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
		mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
		mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
		mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
		mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
		mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
		mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
		mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		//Read next set of B columns
		ptr_b += (cs_b + cs_b_offset[5]);
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);
	//end loop of cols
	}

	//Last block trsm processing
	ptr_b_dup = ptr_b;

	/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

	////unpacklow////
	mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
	
	////unpackhigh////
	mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
	mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
	mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
	mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
	mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
	mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

	//extract diag a00 from a
	//mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
	//mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
	//mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

	//Merge rearranged high elements into complete rows
	mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
	mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
	mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
	mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

	//extract diag a11 from a
	//mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
	//mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
	mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
	//mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

	//extract diag a22 from a
	//mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	//mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

	//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
	//mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

	//extract diag a33 from a
	//mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	//mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

	//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
	//mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

	//extract diag a44 from a
	//mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
	//mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

	//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
	//mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

	//extract diag a55 from a
	//mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
	//mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

	//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
	//mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

	//extract diag a66 from a
	//mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	//mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

	//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
	//mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

	//extract diag a77 from a
	//mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	//mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

	//(Row7): FMA operations of b7 with elements of index (7, 0)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
	//mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
	mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
	mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
	mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
	mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
	mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
	mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
	mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
	mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
	mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

	////unpackhigh////
	mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

	//Merge rearranged high elements into complete rows
	mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

	//Store the computed B columns
	_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
	_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);
	//end loop of cols
}

static void blis_strsm_microkernel(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	float ones = 1.0;
	int j;
	int cs_b_offset[6];
	//int row2, row4, row6;
	float *ptr_b_dup;

	//70 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_cols[8];
	__m256 mat_a_cols_rearr[36];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags;

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);
	reciprocal_diags = _mm256_broadcast_ss((float const *)&ones);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//read first set of 16x8 block of B into registers, where 16 is the blk_height and 8 is the blk_width for B
	mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
	//_mm_prefetch((char*)(ptr_l + 0), _MM_HINT_T0);
	//row2 = (cs_l << 1);
	//row4 = (cs_l << 2);
	mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
	//_mm_prefetch((char*)(ptr_l + cs_l), _MM_HINT_T0);
	mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
	//_mm_prefetch((char*)(ptr_l + row2), _MM_HINT_T0);
	mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
	//_mm_prefetch((char*)(ptr_l + row2 + cs_l), _MM_HINT_T0);
	//row6 = row2 + row4;
	mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
	//_mm_prefetch((char*)(ptr_l + row4), _MM_HINT_T0);
	mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
	//_mm_prefetch((char*)(ptr_l + row4 + cs_l), _MM_HINT_T0);
	mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
	//_mm_prefetch((char*)(ptr_l + row6), _MM_HINT_T0);
	mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));
	//_mm_prefetch((char*)(ptr_l + row6 + cs_l), _MM_HINT_T0);

	//reciprocal_diags = _mm256_loadu_ps((float const *)ones);

	//read first set of 16x16 block of L, where 16 is the blk_height and 16 is the blk_width  for L
	/*mat_a_cols[0] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[1] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[2] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[3] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[4] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[5] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[6] = _mm256_loadu_ps((float const *)ptr_l);
	ptr_l += cs_l;
	mat_a_cols[7] = _mm256_loadu_ps((float const *)ptr_l);*/

	//Shuffle to rearrange/transpose 16x16 block of L into contiguous row-wise registers
	//tmpRegs[0] = _mm256_castps256_ps128(mat_a_cols[0]); //zero latency, no instruction added actually.
	//mat_a_cols_rearr[0] = _mm256_broadcastss_ps(tmpRegs[0]);
	//1st col
	mat_a_cols_rearr[0] = _mm256_broadcast_ss((float const *)(ptr_l+0));
	mat_a_cols_rearr[1] = _mm256_broadcast_ss((float const *)(ptr_l+1));
	mat_a_cols_rearr[3] = _mm256_broadcast_ss((float const *)(ptr_l+2));
	mat_a_cols_rearr[6] = _mm256_broadcast_ss((float const *)(ptr_l+3));
	mat_a_cols_rearr[10] = _mm256_broadcast_ss((float const *)(ptr_l+4));
	mat_a_cols_rearr[15] = _mm256_broadcast_ss((float const *)(ptr_l+5));
	mat_a_cols_rearr[21] = _mm256_broadcast_ss((float const *)(ptr_l+6));
	mat_a_cols_rearr[28] = _mm256_broadcast_ss((float const *)(ptr_l+7));
	//2nd col
	ptr_l += cs_l;
	mat_a_cols_rearr[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_cols_rearr[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[7] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[11] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[16] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[22] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[29] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//3rd col
	ptr_l += cs_l;
	mat_a_cols_rearr[5] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_cols_rearr[8] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[12] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[17] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[23] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[30] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//4rth col
	ptr_l += cs_l;
	mat_a_cols_rearr[9] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_cols_rearr[13] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[18] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[24] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[31] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//5th col
	ptr_l += cs_l;
	mat_a_cols_rearr[14] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_cols_rearr[19] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[25] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[32] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//6th col
	ptr_l += cs_l;
	mat_a_cols_rearr[20] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_cols_rearr[26] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[33] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[27] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_cols_rearr[34] = _mm256_broadcast_ss((float const *)(ptr_l + 7));
	//7th col
	ptr_l += cs_l;
	mat_a_cols_rearr[35] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	numCols_b -= 8; // blk_width = 8

	//compute reciprocals of L(i,i) and broadcast in registers
	mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[2]);
	mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_cols_rearr[5], mat_a_cols_rearr[9]);
	mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_cols_rearr[14], mat_a_cols_rearr[20]);
	mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_cols_rearr[27], mat_a_cols_rearr[35]);

	//mat_a_diag_inv[1] = _mm256_permute_ps(mat_a_diag_inv[1], 0x55);
	//mat_a_diag_inv[3] = _mm256_permute_ps(mat_a_diag_inv[3], 0x55);
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x20);

	//reciprocal of diagnol elements
	reciprocal_diags = _mm256_div_ps(reciprocal_diags, mat_a_diag_inv[0]);

	//Start loop for cols of B to be processed in size of blk_width
	for (j = 0; j < numCols_b; j += 8)
	{
		ptr_b_dup = ptr_b;

		/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

		//--> Transpose and store results of columns of B block <--//
		////unpacklow////
		mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
		mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
		mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
		mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
		mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
		mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
		mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
		mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
		mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
		mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
		mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		//Read next set of B columns
		ptr_b += (cs_b + cs_b_offset[5]);
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + (cs_b)));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0]));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1]));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2]));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3]));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4]));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5]));

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);
	//end loop of cols
	}

	//Last block trsm processing
	ptr_b_dup = ptr_b;

	/*Shuffle to rearrange/transpose 16x8 block of B into contiguous row-wise registers*/

	////unpacklow////
	mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
	
	////unpackhigh////
	mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
	mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
	mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
	mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
	mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
	mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
	mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
	mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
	mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
	mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
	mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags, 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
	mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);

	//Merge rearranged high elements into complete rows
	mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
	mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
	mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
	mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags, 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
	mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_cols_rearr[1], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[3], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[6], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[10], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[15], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[21], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[28], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);

	//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
	mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_cols_rearr[4], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[7], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[11], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[16], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[22], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[29], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
	mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);

	//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
	mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_cols_rearr[8], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[12], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[17], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[23], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[30], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
	mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags, 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);

	//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
	mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_cols_rearr[13], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[18], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[24], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[31], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
	mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags, 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);

	//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
	mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_cols_rearr[19], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[25], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[32], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
	mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags, 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);

	//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
	mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_cols_rearr[26], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[33], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
	mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags, 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);

	//(Row7): FMA operations of b7 with elements of index (7, 0)
	mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_cols_rearr[34], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

	//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
	mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

	//--> Transpose and store results of columns of B block <--//
	////unpacklow////
	mat_a_cols[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_a_cols[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_a_cols[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_a_cols[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange low elements
#if REARRANGE_SHFL == 1
	mat_a_cols[4] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x44);
	mat_a_cols[5] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0xEE);
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x44);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0xEE);
#else
	mat_a_cols[6] = _mm256_shuffle_ps(mat_a_cols[0], mat_a_cols[1], 0x4E);
	mat_a_cols[7] = _mm256_shuffle_ps(mat_a_cols[2], mat_a_cols[3], 0x4E);
	mat_a_cols[4] = _mm256_blend_ps(mat_a_cols[0], mat_a_cols[6], 0xCC);
	mat_a_cols[5] = _mm256_blend_ps(mat_a_cols[1], mat_a_cols[6], 0x33);
	mat_a_cols[6] = _mm256_blend_ps(mat_a_cols[2], mat_a_cols[7], 0xCC);
	mat_a_cols[7] = _mm256_blend_ps(mat_a_cols[3], mat_a_cols[7], 0x33);
#endif
	//Merge rearranged low elements into complete rows
	mat_a_cols[0] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x20);
	mat_a_cols[4] = _mm256_permute2f128_ps(mat_a_cols[4], mat_a_cols[6], 0x31);
	mat_a_cols[1] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x20);
	mat_a_cols[5] = _mm256_permute2f128_ps(mat_a_cols[5], mat_a_cols[7], 0x31);

	////unpackhigh////
	mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
	mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
	mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
	mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

	//Rearrange high elements
#if REARRANGE_SHFL == 1
	mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
	mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
	mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
	mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
	mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
	mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
	mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
	mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

	//Merge rearranged high elements into complete rows
	mat_a_cols[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
	mat_a_cols[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
	mat_a_cols[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
	mat_a_cols[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

	//Store the computed B columns
	_mm256_storeu_ps((float *)ptr_b_dup, mat_a_cols[0]);
	_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_a_cols[1]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_a_cols[2]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_a_cols[3]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_a_cols[4]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_a_cols[5]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_a_cols[6]);
	_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_a_cols[7]);
	//end loop of cols
}

///////////////////////////////////// XA'=B functions ////////////////////////////////
static void dtrsm_XAtB_block_allSmallSizedMatrices_alpha(double *ptr_l,
							 double *ptr_b,
							 int numRows_lb,
							 int numCols_b,
							 int rs_l,
							 int rs_b,
							 int cs_l,
							 int cs_b,
							 double alpha
							)

{

	double ones = 1.0;
	int i,i1,i2,i3,i4,j,k,l;
	int cs_b_offset[3];
	int cs_l_offset[3];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[16][4];
	__m256d mat_a_cols_rearr[4];
	__m256d mat_a_blk_elems[16];
	__m256d mat_a_diag_inv[4];
	__m256d reciprocal_diags[2];
	__m256d alphaReg;
	reciprocal_diags[0] = _mm256_broadcast_sd((double const *)(&ones));
	alphaReg = _mm256_broadcast_sd((double const *)&alpha);

	// ---> considering that the matrix size is multiple of 4 rows and 4 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);

	//read diag elems of L 4x4 block
	mat_a_cols_rearr[0] = _mm256_loadu_pd((double const *)ptr_l);
	mat_a_cols_rearr[1] = _mm256_loadu_pd((double const *)ptr_l + cs_l);
	mat_a_cols_rearr[2] = _mm256_loadu_pd((double const *)ptr_l + cs_l_offset[0]);
	mat_a_cols_rearr[3] = _mm256_loadu_pd((double const *)ptr_l + cs_l_offset[1]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 4 diags together
	mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0x0A);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_pd(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0x0A);//diag 2,3

	mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);//diag 0,1,2,3

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_pd(reciprocal_diags[0], mat_a_diag_inv[0]);

	//Broadcast A10 to A30 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//Broadcast A21 to A31 to registers
	mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 3));

	//Broadcast A32 to register
	mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + 3));

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags[0], 0x03);
	mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags[0], 0x0C);
	mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

	/***************** first set of 4 cols of B processing starts *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 4)
	{
		/////////////////// Complete Upper 4x4 block trsm of B :- upper 4x4 block of B with upper 4x4 block of A
		//read 4x4 block of B into registers

		mat_b_rearr[0][0] = _mm256_loadu_pd((double const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

		mat_b_rearr[0][0] = _mm256_mul_pd(mat_b_rearr[0][0], alphaReg);
		mat_b_rearr[1][0] = _mm256_mul_pd(mat_b_rearr[1][0], alphaReg);
		mat_b_rearr[2][0] = _mm256_mul_pd(mat_b_rearr[2][0], alphaReg);
		mat_b_rearr[3][0] = _mm256_mul_pd(mat_b_rearr[3][0], alphaReg);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_pd(mat_b_rearr[0][0], mat_a_diag_inv[0]);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_pd(mat_b_rearr[1][0], mat_a_diag_inv[1]);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_pd(mat_b_rearr[2][0], mat_a_diag_inv[2]);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_pd(mat_b_rearr[3][0], mat_a_diag_inv[3]);

		//Store the computed B columns
		_mm256_storeu_pd((double *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);

		i += 4;
		ptr_b_dup += 4;

	}

	/***************** first set of 4 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width}

	for (j = 4; j < numRows_lb; j += 4)//m :- 4x4 block row
	{
		ptr_l += 4;
		ptr_b_dup += cs_b_offset[2];
		i1 += cs_b_offset[2];
		//printf("i1 = i3 = %g\n",*(ptr_l+i1));
		//Read next 4x4 block of A to get diag elements
		i3 += cs_l_offset[2];
		mat_a_cols_rearr[0] = _mm256_loadu_pd((double const *)ptr_l + i3);
		mat_a_cols_rearr[1] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l);
		mat_a_cols_rearr[2] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_cols_rearr[3] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l_offset[1]);

		//pack 4 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0x0A);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_pd(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0x0A);//diag 2,3

		mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);//diag 0,1,2,3

		//reciprocal of diagnal elements of A :- 0,1,2,3
		reciprocal_diags[0] = _mm256_div_pd(reciprocal_diags[0], mat_a_diag_inv[0]);

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 4)
		{
			i = i1 + k;
			mat_b_rearr[i2][0] = _mm256_loadu_pd((double const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

			mat_b_rearr[i2][0] = _mm256_mul_pd(mat_b_rearr[i2][0], alphaReg);
		  mat_b_rearr[i2][1] = _mm256_mul_pd(mat_b_rearr[i2][1], alphaReg);
		  mat_b_rearr[i2][2] = _mm256_mul_pd(mat_b_rearr[i2][2], alphaReg);
		  mat_b_rearr[i2][3] = _mm256_mul_pd(mat_b_rearr[i2][3], alphaReg);
			i2++;
		}


		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 4) // move across m
		{

			//Broadcast A4,0 to A7,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));

			//Broadcast A41 to A71 to registers
			mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[7] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 3));

			//Broadcast A4,2 to A7,2 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[9] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 3));

			//Broadcast A4,3 to A7,3 to registers
			mat_a_blk_elems[12] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[13] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[14] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[15] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 3));

			i += cs_l_offset[2];

			for (k = 0; k < numCols_b; k += 4) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[1]));


				i4 = k >> 2;

				//(Row4): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row5): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)


				//(Row6): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[8], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[9], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[10], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[11], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row7): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[12], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[13], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[14], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[15], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				//end loop of cols

 			}
			i2 += cs_b_offset[2];

		}

		//Broadcast A10 to A30 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//Broadcast A21 to A31 to registers
		mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags[0], 0x03);
		mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags[0], 0x0C);
		mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

		k = 0;
		for (i = 0; i < numCols_b; i+=4)
		{



			/////////////////// Complete Lower 4x4 block trsm of B :- lower 4x4 block of B with lower right 4x4 block of A

			//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[k][0] = _mm256_mul_pd(mat_b_rearr[k][0], mat_a_diag_inv[0]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[k][1] = _mm256_mul_pd(mat_b_rearr[k][1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[k][2] = _mm256_mul_pd(mat_b_rearr[k][2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[k][3] = _mm256_mul_pd(mat_b_rearr[k][3], mat_a_diag_inv[3]);

			//Store the computed B columns

			_mm256_storeu_pd((double *)(ptr_b_dup + i), mat_b_rearr[k][0]);
			_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);

			k++;
		}
	}

}

static void dtrsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag(double *ptr_l,
							 double *ptr_b,
							 int numRows_lb,
							 int numCols_b,
							 int rs_l,
							 int rs_b,
							 int cs_l,
							 int cs_b,
							 double alpha
							)

{

	int i,i1,i2,i3,i4,j,k,l;
	int cs_b_offset[3];
	int cs_l_offset[3];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[16][4];
	__m256d mat_a_blk_elems[16];
	__m256d alphaReg;
	alphaReg = _mm256_broadcast_sd((double const *)&alpha);

	// ---> considering that the matrix size is multiple of 4 rows and 4 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);

	//Broadcast A10 to A30 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//Broadcast A21 to A31 to registers
	mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 3));

	//Broadcast A32 to register
	mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + 3));

	/***************** first set of 4 cols of B processing starts *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 4)
	{
		/////////////////// Complete Upper 4x4 block trsm of B :- upper 4x4 block of B with upper 4x4 block of A
		//read 4x4 block of B into registers

		mat_b_rearr[0][0] = _mm256_loadu_pd((double const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

		mat_b_rearr[0][0] = _mm256_mul_pd(mat_b_rearr[0][0], alphaReg);
		mat_b_rearr[1][0] = _mm256_mul_pd(mat_b_rearr[1][0], alphaReg);
		mat_b_rearr[2][0] = _mm256_mul_pd(mat_b_rearr[2][0], alphaReg);
		mat_b_rearr[3][0] = _mm256_mul_pd(mat_b_rearr[3][0], alphaReg);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[0][0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[0][0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[0][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[1][0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[1][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_rearr[2][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//Store the computed B columns
		_mm256_storeu_pd((double *)ptr_b_dup, mat_b_rearr[0][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_b_rearr[1][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3][0]);

		i += 4;
		ptr_b_dup += 4;

	}

	/***************** first set of 4 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width}

	for (j = 4; j < numRows_lb; j += 4)//m :- 4x4 block row
	{
		ptr_l += 4;
		ptr_b_dup += cs_b_offset[2];
		i1 += cs_b_offset[2];
		i3 += cs_l_offset[2];
		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 4)
		{
			i = i1 + k;
			mat_b_rearr[i2][0] = _mm256_loadu_pd((double const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

			mat_b_rearr[i2][0] = _mm256_mul_pd(mat_b_rearr[i2][0], alphaReg);
		        mat_b_rearr[i2][1] = _mm256_mul_pd(mat_b_rearr[i2][1], alphaReg);
		    	mat_b_rearr[i2][2] = _mm256_mul_pd(mat_b_rearr[i2][2], alphaReg);
		    	mat_b_rearr[i2][3] = _mm256_mul_pd(mat_b_rearr[i2][3], alphaReg);
			i2++;
		}


		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 4) // move across m
		{

			//Broadcast A4,0 to A7,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));

			//Broadcast A41 to A71 to registers
			mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[7] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 3));

			//Broadcast A4,2 to A7,2 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[9] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 3));

			//Broadcast A4,3 to A7,3 to registers
			mat_a_blk_elems[12] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[13] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[14] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[15] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 3));

			i += cs_l_offset[2];

			for (k = 0; k < numCols_b; k += 4) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[1]));


				i4 = k >> 2;

				//(Row4): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row5): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)


				//(Row6): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[8], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[9], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[10], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[11], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row7): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[12], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[13], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[14], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[15], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				//end loop of cols

 			}
			i2 += cs_b_offset[2];

		}

		//Broadcast A10 to A30 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//Broadcast A21 to A31 to registers
		mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		k = 0;
		for (i = 0; i < numCols_b; i+=4)
		{



			/////////////////// Complete Lower 4x4 block trsm of B :- lower 4x4 block of B with lower right 4x4 block of A

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)

			//Store the computed B columns

			_mm256_storeu_pd((double *)(ptr_b_dup + i), mat_b_rearr[k][0]);
			_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);

			k++;
		}

	}


}

static void dtrsm_XAtB_block_allSmallSizedMatrices_unitDiag(double *ptr_l,
							 double *ptr_b,
							 int numRows_lb,
							 int numCols_b,
							 int rs_l,
							 int rs_b,
							 int cs_l,
							 int cs_b
							)

{

	int i,i1,i2,i3,i4,j,k,l;
	int cs_b_offset[3];
	int cs_l_offset[3];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[16][4];
	__m256d mat_a_blk_elems[16];

	// ---> considering that the matrix size is multiple of 4 rows and 4 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);

	//Broadcast A10 to A30 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//Broadcast A21 to A31 to registers
	mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 3));

	//Broadcast A32 to register
	mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + 3));

	/***************** first set of 4 cols of B processing starts *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 4)
	{
		/////////////////// Complete Upper 4x4 block trsm of B :- upper 4x4 block of B with upper 4x4 block of A
		//read 4x4 block of B into registers

		mat_b_rearr[0][0] = _mm256_loadu_pd((double const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[0][0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[0][0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[0][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[1][0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[1][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_rearr[2][0], mat_b_rearr[3][0]);//d = c - (a*b)

		//Store the computed B columns
		_mm256_storeu_pd((double *)ptr_b_dup, mat_b_rearr[0][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_b_rearr[1][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2][0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3][0]);

		i += 4;
		ptr_b_dup += 4;

	}

	/***************** first set of 4 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width}

	for (j = 4; j < numRows_lb; j += 4)//m :- 4x4 block row
	{
		ptr_l += 4;
		ptr_b_dup += cs_b_offset[2];
		i1 += cs_b_offset[2];
		i3 += cs_l_offset[2];
		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 4)
		{
			i = i1 + k;
			mat_b_rearr[i2][0] = _mm256_loadu_pd((double const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

			i2++;
		}


		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 4) // move across m
		{

			//Broadcast A4,0 to A7,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));

			//Broadcast A41 to A71 to registers
			mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[7] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 3));

			//Broadcast A4,2 to A7,2 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[9] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 3));

			//Broadcast A4,3 to A7,3 to registers
			mat_a_blk_elems[12] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[13] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[14] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[15] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 3));

			i += cs_l_offset[2];

			for (k = 0; k < numCols_b; k += 4) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[1]));


				i4 = k >> 2;

				//(Row4): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row5): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)


				//(Row6): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[8], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[9], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[10], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[11], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row7): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[12], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[13], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[14], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[15], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				//end loop of cols

 			}
			i2 += cs_b_offset[2];

		}

		//Broadcast A10 to A30 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//Broadcast A21 to A31 to registers
		mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		k = 0;
		for (i = 0; i < numCols_b; i+=4)
		{



			/////////////////// Complete Lower 4x4 block trsm of B :- lower 4x4 block of B with lower right 4x4 block of A

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)

			//Store the computed B columns

			_mm256_storeu_pd((double *)(ptr_b_dup + i), mat_b_rearr[k][0]);
			_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);

			k++;
		}

	}

}
static void dtrsm_XAtB_block_allSmallSizedMatrices(double *ptr_l,
							 double *ptr_b,
							 int numRows_lb,
							 int numCols_b,
							 int rs_l,
							 int rs_b,
							 int cs_l,
							 int cs_b
							)

{

	double ones = 1.0;
	int i,i1,i2,i3,i4,j,k,l;
	int cs_b_offset[3];
	int cs_l_offset[3];
	double *ptr_b_dup;

	__m256d mat_b_col[4];
	__m256d mat_b_rearr[16][4];
	__m256d mat_a_cols_rearr[4];
	__m256d mat_a_blk_elems[16];
	__m256d mat_a_diag_inv[4];
	__m256d reciprocal_diags[2];

	reciprocal_diags[0] = _mm256_broadcast_sd((double const *)(&ones));

	// ---> considering that the matrix size is multiple of 4 rows and 4 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);

	//read diag elems of L 4x4 block
	mat_a_cols_rearr[0] = _mm256_loadu_pd((double const *)ptr_l);
	mat_a_cols_rearr[1] = _mm256_loadu_pd((double const *)ptr_l + cs_l);
	mat_a_cols_rearr[2] = _mm256_loadu_pd((double const *)ptr_l + cs_l_offset[0]);
	mat_a_cols_rearr[3] = _mm256_loadu_pd((double const *)ptr_l + cs_l_offset[1]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 4 diags together
	mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0x0A);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_pd(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0x0A);//diag 2,3

	mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);//diag 0,1,2,3

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_pd(reciprocal_diags[0], mat_a_diag_inv[0]);

	//Broadcast A10 to A30 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + 3));

	//Broadcast A21 to A31 to registers
	mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + 3));

	//Broadcast A32 to register
	mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + 3));

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags[0], 0x03);
	mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags[0], 0x0C);
	mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

	/***************** first set of 4 cols of B processing starts *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 4)
	{
		/////////////////// Complete Upper 4x4 block trsm of B :- upper 4x4 block of B with upper 4x4 block of A
		//read 4x4 block of B into registers

		mat_b_rearr[0][0] = _mm256_loadu_pd((double const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_pd(mat_b_rearr[0][0], mat_a_diag_inv[0]);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_pd(mat_b_rearr[1][0], mat_a_diag_inv[1]);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_pd(mat_b_rearr[2][0], mat_a_diag_inv[2]);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_pd(mat_b_rearr[3][0], mat_a_diag_inv[3]);

		//Store the computed B columns
		_mm256_storeu_pd((double *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);

		i += 4;
		ptr_b_dup += 4;

	}

	/***************** first set of 4 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width}

	for (j = 4; j < numRows_lb; j += 4)//m :- 4x4 block row
	{
		ptr_l += 4;
		ptr_b_dup += cs_b_offset[2];
		i1 += cs_b_offset[2];
		//printf("i1 = i3 = %g\n",*(ptr_l+i1));
		//Read next 4x4 block of A to get diag elements
		i3 += cs_l_offset[2];
		mat_a_cols_rearr[0] = _mm256_loadu_pd((double const *)ptr_l + i3);
		mat_a_cols_rearr[1] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l);
		mat_a_cols_rearr[2] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_cols_rearr[3] = _mm256_loadu_pd((double const *)ptr_l + i3 + cs_l_offset[1]);

		//pack 4 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0x0A);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_pd(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0x0A);//diag 2,3

		mat_a_diag_inv[0] = _mm256_blend_pd(mat_a_diag_inv[0], mat_a_diag_inv[1], 0x0C);//diag 0,1,2,3

		//reciprocal of diagnal elements of A :- 0,1,2,3
		reciprocal_diags[0] = _mm256_div_pd(reciprocal_diags[0], mat_a_diag_inv[0]);

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 4)
		{
			i = i1 + k;
			mat_b_rearr[i2][0] = _mm256_loadu_pd((double const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_pd((double const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_pd((double const *)(ptr_b + cs_b_offset[1] + i));

			i2++;
		}


		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 4) // move across m
		{

			//Broadcast A4,0 to A7,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));

			//Broadcast A41 to A71 to registers
			mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[6] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[7] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l + i + 3));

			//Broadcast A4,2 to A7,2 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[9] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[0] + i + 3));

			//Broadcast A4,3 to A7,3 to registers
			mat_a_blk_elems[12] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[13] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[14] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[15] = _mm256_broadcast_sd((double const *)(ptr_l + cs_l_offset[1] + i + 3));

			i += cs_l_offset[2];

			for (k = 0; k < numCols_b; k += 4) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_pd((double const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_pd((double const *)(ptr_b + i4 + cs_b_offset[1]));


				i4 = k >> 2;

				//(Row4): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row5): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)


				//(Row6): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[8], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[9], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[10], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[11], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				//(Row7): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_pd(mat_a_blk_elems[12], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_pd(mat_a_blk_elems[13], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_pd(mat_a_blk_elems[14], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_pd(mat_a_blk_elems[15], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				//end loop of cols

 			}
			i2 += cs_b_offset[2];

		}

		//Broadcast A10 to A30 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_sd((double const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_pd(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);

		//Broadcast A21 to A31 to registers
		mat_a_blk_elems[3] = _mm256_broadcast_sd((double const *)(ptr_l + i + 2));
		mat_a_blk_elems[4] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_pd(reciprocal_diags[0], 0x03);
		mat_a_diag_inv[1] = _mm256_permute2f128_pd(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);

		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[5] = _mm256_broadcast_sd((double const *)(ptr_l + i + 3));
		i += cs_l;

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_pd(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[2] = _mm256_permute2f128_pd(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x11);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_pd(reciprocal_diags[0], 0x0C);
		mat_a_diag_inv[3] = _mm256_permute2f128_pd(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x11);

		k = 0;
		for (i = 0; i < numCols_b; i+=4)
		{



			/////////////////// Complete Lower 4x4 block trsm of B :- lower 4x4 block of B with lower right 4x4 block of A

			//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[k][0] = _mm256_mul_pd(mat_b_rearr[k][0], mat_a_diag_inv[0]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (3, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_pd(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[k][1] = _mm256_mul_pd(mat_b_rearr[k][1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_pd(mat_a_blk_elems[3], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[4], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[k][2] = _mm256_mul_pd(mat_b_rearr[k][2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_pd(mat_a_blk_elems[5], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[k][3] = _mm256_mul_pd(mat_b_rearr[k][3], mat_a_diag_inv[3]);

			//Store the computed B columns

			_mm256_storeu_pd((double *)(ptr_b_dup + i), mat_b_rearr[k][0]);
			_mm256_storeu_pd((double *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_pd((double *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);

			k++;
		}

	}

}
#if OPT_CACHE_BLOCKING_L1 //new intrinsic kernels
static void trsm_XAtB_block_allSmallSizedMatrices(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);
#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif
	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], mat_a_diag_inv[0]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], mat_a_diag_inv[1]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], mat_a_diag_inv[2]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], mat_a_diag_inv[3]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], mat_a_diag_inv[4]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], mat_a_diag_inv[5]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], mat_a_diag_inv[6]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += cs_l_offset[6];
		mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			i = i1 + r;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
#endif
			i = 0;
			i2 = 0;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				{
					/////////////////// Partial Lower 8x8 block trsm of B
					ptr_l_dup = ptr_l;
					i4 = i2 + r;
					//Read current 8 cols of B columns from specified 8x8 current-block of B
					mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
					mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
					mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
					mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
					mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
					mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
					mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
					mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

					//Broadcast A8,0 to A15,0 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					i4 = k >> 3;
					ptr_l_dup += cs_l;

#if GEMM_ACCUM_A
					//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
					mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
					mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
					mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
					mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
					mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
					mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
					mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
					//Broadcast A21 to A71 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				}
				i2 += cs_b_offset[6];
				i += cs_l_offset[6];
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			{
				i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i2);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i2));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i2));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i2));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i2));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i2));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i2));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i2));
#endif
				//Broadcast A10 to A70 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

#if GEMM_ACCUM_A
				//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
				mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
				mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#endif

#if GEMM_ACCUM_A
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
				mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
				mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
				mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
				mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
				mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
				mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

				//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
				mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

				//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A32 to A72 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
				mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

				//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A43 to A73 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
				mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

				//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A54 to A74 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
				mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

				//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A65 to A75 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
				mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

				//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A76 to register
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));

				//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
				mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

				//(Row7): FMA operations of b7 with elements of index (7, 0)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

				//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
				mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

				////////////////////////////////////////////////////////////////////////////////

				//Store the computed B columns
				_mm256_storeu_ps((float *)ptr_b_dup + r, mat_b_rearr[0]);
				_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+r), mat_b_rearr[1]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + r), mat_b_rearr[2]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + r), mat_b_rearr[3]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + r), mat_b_rearr[4]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + r), mat_b_rearr[5]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + r), mat_b_rearr[6]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + r), mat_b_rearr[7]);
				//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
				k++;
			}
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_alpha(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];
	__m256 alphaReg;

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);
#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif
	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], mat_a_diag_inv[0]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], mat_a_diag_inv[1]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], mat_a_diag_inv[2]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], mat_a_diag_inv[3]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], mat_a_diag_inv[4]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], mat_a_diag_inv[5]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], mat_a_diag_inv[6]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += cs_l_offset[6];
		mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			i = i1 + r;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
	        mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
	    	mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
	    	mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
	    	mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
	    	mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
	    	mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
	    	mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);
#endif
			i = 0;
			i2 = 0;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				{
					/////////////////// Partial Lower 8x8 block trsm of B
					ptr_l_dup = ptr_l;
					i4 = i2 + r;
					//Read current 8 cols of B columns from specified 8x8 current-block of B
					mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
					mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
					mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
					mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
					mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
					mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
					mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
					mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

					//Broadcast A8,0 to A15,0 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					i4 = k >> 3;
					ptr_l_dup += cs_l;

#if GEMM_ACCUM_A
					//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
					mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
					mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
					mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
					mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
					mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
					mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
					mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
					//Broadcast A21 to A71 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				}
				i2 += cs_b_offset[6];
				i += cs_l_offset[6];
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			{
				i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i2);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i2));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i2));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i2));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i2));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i2));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i2));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i2));
				
				mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
				mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
				mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
				mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
				mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
				mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
				mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
				mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);
#endif
				//Broadcast A10 to A70 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

#if GEMM_ACCUM_A
				//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
				mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
				mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#endif

#if GEMM_ACCUM_A
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
				mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
				mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
				mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
				mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
				mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
				mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

				//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
				mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

				//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A32 to A72 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
				mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

				//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A43 to A73 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
				mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

				//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A54 to A74 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
				mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

				//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A65 to A75 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
				mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

				//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A76 to register
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));

				//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
				mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

				//(Row7): FMA operations of b7 with elements of index (7, 0)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

				//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
				mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

				////////////////////////////////////////////////////////////////////////////////

				//Store the computed B columns

				_mm256_storeu_ps((float *)ptr_b_dup + r, mat_b_rearr[0]);
				_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+r), mat_b_rearr[1]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + r), mat_b_rearr[2]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + r), mat_b_rearr[3]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + r), mat_b_rearr[4]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + r), mat_b_rearr[5]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + r), mat_b_rearr[6]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + r), mat_b_rearr[7]);
				//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
				k++;
			}
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	//float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		//(Row0)
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)


		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];
		i3 += cs_l_offset[6];

		i = 0;
		i2 = 0;
		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			i = i1 + r;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
#endif
			i = 0;
			i2 = 0;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				{
					/////////////////// Partial Lower 8x8 block trsm of B
					ptr_l_dup = ptr_l;
					i4 = i2 + r;
					//Read current 8 cols of B columns from specified 8x8 current-block of B
					mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
					mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
					mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
					mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
					mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
					mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
					mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
					mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

					//Broadcast A8,0 to A15,0 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					i4 = k >> 3;
					ptr_l_dup += cs_l;

#if GEMM_ACCUM_A
					//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
					mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
					mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
					mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
					mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
					mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
					mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
					mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
					//Broadcast A21 to A71 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				}
				i2 += cs_b_offset[6];
				i += cs_l_offset[6];
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			{
				i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i2);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i2));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i2));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i2));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i2));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i2));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i2));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i2));
#endif
				//Broadcast A10 to A70 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

#if GEMM_ACCUM_A
			//(Row0): already done
#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
#endif

#if GEMM_ACCUM_A
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
				mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
				mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
				mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
				mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
				mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
				mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

				//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;


				//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A32 to A72 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;


				//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A43 to A73 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;


				//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A54 to A74 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;


				//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A65 to A75 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;


				//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A76 to register
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));


				//(Row7): FMA operations of b7 with elements of index (7, 0)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)


				////////////////////////////////////////////////////////////////////////////////

				//Store the computed B columns
				_mm256_storeu_ps((float *)ptr_b_dup + r, mat_b_rearr[0]);
				_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+r), mat_b_rearr[1]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + r), mat_b_rearr[2]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + r), mat_b_rearr[3]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + r), mat_b_rearr[4]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + r), mat_b_rearr[5]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + r), mat_b_rearr[6]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + r), mat_b_rearr[7]);
				//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
				k++;
			}
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	//float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];
	__m256 alphaReg;
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);
		
		//(Row0)
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];
		i3 += cs_l_offset[6];

		i = 0;
		i2 = 0;
		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			i = i1 + r;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
			
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
		    mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
		    mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
		    mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
		    mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
		    mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
		    mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
		    mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);
#endif
			i = 0;
			i2 = 0;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				{
					/////////////////// Partial Lower 8x8 block trsm of B
					ptr_l_dup = ptr_l;
					i4 = i2 + r;
					//Read current 8 cols of B columns from specified 8x8 current-block of B
					mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
					mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
					mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
					mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
					mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
					mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
					mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
					mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

					//Broadcast A8,0 to A15,0 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					i4 = k >> 3;
					ptr_l_dup += cs_l;

#if GEMM_ACCUM_A
					//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
					mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
					mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
					mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
					mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
					mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
					mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
					mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
					//Broadcast A21 to A71 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
					mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i));
					mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 1));
					mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 2));
					mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 3));
					mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 4));
					mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 5));
					mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 6));
					mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + i + 7));
					ptr_l_dup += cs_l;
#if GEMM_ACCUM_A
					//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
					mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
					mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
					mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
					mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
					mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
					mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
					mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
					mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
					mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				}
				i2 += cs_b_offset[6];
				i += cs_l_offset[6];
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			{
				i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i2);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i2));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i2));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i2));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i2));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i2));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i2));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i2));
				
				mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
				mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
				mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
				mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
				mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
				mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
				mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
				mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);
#endif
				//Broadcast A10 to A70 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

#if GEMM_ACCUM_A
			//(Row0): already done

#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
#endif

#if GEMM_ACCUM_A
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
				mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
				mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
				mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
				mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
				mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
				mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

				//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				
				//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A32 to A72 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				
				//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A43 to A73 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				
				//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A54 to A74 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				
				//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A65 to A75 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
				i += cs_l;

				
				//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

				//Broadcast A76 to register
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));

				
				//(Row7): FMA operations of b7 with elements of index (7, 0)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

				
				////////////////////////////////////////////////////////////////////////////////

				//Store the computed B columns
				_mm256_storeu_ps((float *)ptr_b_dup + r, mat_b_rearr[0]);
				_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+r), mat_b_rearr[1]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + r), mat_b_rearr[2]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + r), mat_b_rearr[3]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + r), mat_b_rearr[4]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + r), mat_b_rearr[5]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + r), mat_b_rearr[6]);
				_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + r), mat_b_rearr[7]);
				//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
				k++;
			}
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}
#else //rel 1.0 intrisic kernels (NOT OPT_CACHE_BLOCKING_L1)
static void trsm_XAtB_block_allSmallSizedMatrices(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[16][8];
	__m256 mat_a_cols_rearr[8];
	__m256 mat_a_blk_elems[64];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_cols_rearr[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_cols_rearr[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_cols_rearr[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_cols_rearr[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_cols_rearr[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_cols_rearr[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_cols_rearr[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_cols_rearr[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_cols_rearr[4], mat_a_cols_rearr[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_cols_rearr[6], mat_a_cols_rearr[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0][0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_rearr[0][0], mat_a_diag_inv[0]);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_rearr[1][0], mat_a_diag_inv[1]);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_rearr[2][0], mat_a_diag_inv[2]);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[2], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[2], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_rearr[3][0], mat_a_diag_inv[3]);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[3], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[3], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[3], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[3], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_rearr[4][0], mat_a_diag_inv[4]);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[4], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[4], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[4], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_rearr[5][0], mat_a_diag_inv[5]);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[5], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[5], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_rearr[6][0], mat_a_diag_inv[6]);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[6], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_rearr[7][0], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += cs_l_offset[6];
		mat_a_cols_rearr[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_cols_rearr[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_cols_rearr[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_cols_rearr[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_cols_rearr[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_cols_rearr[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_cols_rearr[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_cols_rearr[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_cols_rearr[4], mat_a_cols_rearr[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_cols_rearr[6], mat_a_cols_rearr[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 8)
		{
			i = i1 + k;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[i2][0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[i2][4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[i2][5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[i2][6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[i2][7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
			i2++;
		}
		
		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 8) // move across m
		{
			//Broadcast A8,0 to A15,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
			mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
			//Broadcast A21 to A71 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 3));
			mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 4));
			mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 5));
			mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 6));
			mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 7));
			
			//Broadcast A8,2 to A15,2 to registers
			mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 3));
			mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 4));
			mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 5));
			mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 6));
			mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 7));
		
			//Broadcast A8,3 to A15,3 to registers
			mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 3));
			mat_a_blk_elems[28] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 4));
			mat_a_blk_elems[29] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 5));
			mat_a_blk_elems[30] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 6));
			mat_a_blk_elems[31] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 7));
			
			// _mm256_permute2f128_ps()
			
			//Broadcast A8,4 to A15,4 to registers
			mat_a_blk_elems[32] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i));
			mat_a_blk_elems[33] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 1));
			mat_a_blk_elems[34] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 2));
			mat_a_blk_elems[35] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 3));
			mat_a_blk_elems[36] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 4));
			mat_a_blk_elems[37] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 5));
			mat_a_blk_elems[38] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 6));
			mat_a_blk_elems[39] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 7));
			
			//Broadcast A8,5 to A15,5 to registers
			mat_a_blk_elems[40] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i));
			mat_a_blk_elems[41] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 1));
			mat_a_blk_elems[42] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 2));
			mat_a_blk_elems[43] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 3));
			mat_a_blk_elems[44] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 4));
			mat_a_blk_elems[45] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 5));
			mat_a_blk_elems[46] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 6));
			mat_a_blk_elems[47] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 7));
			
			//Broadcast A8,6 to A15,6 to registers
			mat_a_blk_elems[48] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i));
			mat_a_blk_elems[49] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 1));
			mat_a_blk_elems[50] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 2));
			mat_a_blk_elems[51] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 3));
			mat_a_blk_elems[52] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 4));
			mat_a_blk_elems[53] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 5));
			mat_a_blk_elems[54] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 6));
			mat_a_blk_elems[55] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 7));
			
			//Broadcast A8,7 to A15,7 to registers
			mat_a_blk_elems[56] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i));
			mat_a_blk_elems[57] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 1));
			mat_a_blk_elems[58] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 2));
			mat_a_blk_elems[59] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 3));
			mat_a_blk_elems[60] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 4));
			mat_a_blk_elems[61] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 5));
			mat_a_blk_elems[62] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 6));
			mat_a_blk_elems[63] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 7));
						
			i += cs_l_offset[6];
			
			
			for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				i4 = k >> 3;
				
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[1], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[1], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[1], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[2], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[2], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[2], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[2], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[28], mat_b_col[3], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[29], mat_b_col[3], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[30], mat_b_col[3], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[31], mat_b_col[3], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[32], mat_b_col[4], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[33], mat_b_col[4], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[34], mat_b_col[4], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[35], mat_b_col[4], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[36], mat_b_col[4], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[37], mat_b_col[4], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[38], mat_b_col[4], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[39], mat_b_col[4], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[40], mat_b_col[5], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[41], mat_b_col[5], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[42], mat_b_col[5], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[43], mat_b_col[5], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[44], mat_b_col[5], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[45], mat_b_col[5], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[46], mat_b_col[5], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[47], mat_b_col[5], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[48], mat_b_col[6], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[49], mat_b_col[6], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[50], mat_b_col[6], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[51], mat_b_col[6], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[52], mat_b_col[6], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[53], mat_b_col[6], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[54], mat_b_col[6], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[55], mat_b_col[6], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[56], mat_b_col[7], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[57], mat_b_col[7], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[58], mat_b_col[7], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[59], mat_b_col[7], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[60], mat_b_col[7], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[61], mat_b_col[7], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[62], mat_b_col[7], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[63], mat_b_col[7], mat_b_rearr[i4][7]);//d = c - (a*b)

				//end loop of cols					
			}
			i2 += cs_b_offset[6];
		}
		
		//Broadcast A10 to A70 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);
		
		//Broadcast A21 to A71 to registers
		mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	
		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	
		//Broadcast A43 to A73 to registers
		mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	
		//Broadcast A54 to A74 to registers
		mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	
		//Broadcast A65 to A75 to registers
		mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	
		//Broadcast A76 to register
		mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		k = 0;
		for (i = 0; i < numCols_b; i+=8)
		{
			/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
			
			//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[k][0] = _mm256_mul_ps(mat_b_rearr[k][0], mat_a_diag_inv[0]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[k][0], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[k][0], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[k][0], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[k][0], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[k][1] = _mm256_mul_ps(mat_b_rearr[k][1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_rearr[k][1], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_rearr[k][1], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_rearr[k][1], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_rearr[k][1], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[k][2] = _mm256_mul_ps(mat_b_rearr[k][2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_rearr[k][2], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_rearr[k][2], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_rearr[k][2], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_rearr[k][2], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[k][3] = _mm256_mul_ps(mat_b_rearr[k][3], mat_a_diag_inv[3]);

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_rearr[k][3], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_rearr[k][3], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_rearr[k][3], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_rearr[k][3], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
			mat_b_rearr[k][4] = _mm256_mul_ps(mat_b_rearr[k][4], mat_a_diag_inv[4]);

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_rearr[k][4], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_rearr[k][4], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_rearr[k][4], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
			mat_b_rearr[k][5] = _mm256_mul_ps(mat_b_rearr[k][5], mat_a_diag_inv[5]);

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_rearr[k][5], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_rearr[k][5], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
			mat_b_rearr[k][6] = _mm256_mul_ps(mat_b_rearr[k][6], mat_a_diag_inv[6]);

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_rearr[k][6], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
			mat_b_rearr[k][7] = _mm256_mul_ps(mat_b_rearr[k][7], mat_a_diag_inv[7]);

			////////////////////////////////////////////////////////////////////////////////

			//Store the computed B columns

			_mm256_storeu_ps((float *)ptr_b_dup + i, mat_b_rearr[k][0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i), mat_b_rearr[k][4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i), mat_b_rearr[k][5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i), mat_b_rearr[k][6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i), mat_b_rearr[k][7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
		}


	}
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_alpha(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[16][8];
	__m256 mat_a_cols_rearr[8];
	__m256 mat_a_blk_elems[64];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];
	__m256 alphaReg;

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_cols_rearr[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_cols_rearr[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_cols_rearr[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_cols_rearr[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_cols_rearr[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_cols_rearr[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_cols_rearr[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_cols_rearr[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_cols_rearr[4], mat_a_cols_rearr[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_cols_rearr[6], mat_a_cols_rearr[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));

	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0][0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		mat_b_rearr[0][0] = _mm256_mul_ps(mat_b_rearr[0][0], alphaReg);
		mat_b_rearr[1][0] = _mm256_mul_ps(mat_b_rearr[1][0], alphaReg);
		mat_b_rearr[2][0] = _mm256_mul_ps(mat_b_rearr[2][0], alphaReg);
		mat_b_rearr[3][0] = _mm256_mul_ps(mat_b_rearr[3][0], alphaReg);
		mat_b_rearr[4][0] = _mm256_mul_ps(mat_b_rearr[4][0], alphaReg);
		mat_b_rearr[5][0] = _mm256_mul_ps(mat_b_rearr[5][0], alphaReg);
		mat_b_rearr[6][0] = _mm256_mul_ps(mat_b_rearr[6][0], alphaReg);
		mat_b_rearr[7][0] = _mm256_mul_ps(mat_b_rearr[7][0], alphaReg);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_rearr[0][0], mat_a_diag_inv[0]);

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_rearr[1][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_rearr[1][0], mat_a_diag_inv[1]);

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_rearr[2][0], mat_a_diag_inv[2]);

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[2], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[2], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_rearr[3][0], mat_a_diag_inv[3]);

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[3], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[3], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[3], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[3], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_rearr[4][0], mat_a_diag_inv[4]);

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[4], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[4], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[4], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_rearr[5][0], mat_a_diag_inv[5]);

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[5], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[5], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_rearr[6][0], mat_a_diag_inv[6]);

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[6], mat_b_rearr[7][0]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_rearr[7][0], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += cs_l_offset[6];
		mat_a_cols_rearr[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_cols_rearr[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_cols_rearr[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_cols_rearr[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_cols_rearr[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_cols_rearr[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_cols_rearr[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_cols_rearr[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_cols_rearr[0], mat_a_cols_rearr[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_cols_rearr[2], mat_a_cols_rearr[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_cols_rearr[4], mat_a_cols_rearr[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_cols_rearr[6], mat_a_cols_rearr[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 8)
		{
			i = i1 + k;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[i2][0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[i2][4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[i2][5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[i2][6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[i2][7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
			
			mat_b_rearr[i2][0] = _mm256_mul_ps(mat_b_rearr[i2][0], alphaReg);
		        mat_b_rearr[i2][1] = _mm256_mul_ps(mat_b_rearr[i2][1], alphaReg);
		    	mat_b_rearr[i2][2] = _mm256_mul_ps(mat_b_rearr[i2][2], alphaReg);
		    	mat_b_rearr[i2][3] = _mm256_mul_ps(mat_b_rearr[i2][3], alphaReg);
		    	mat_b_rearr[i2][4] = _mm256_mul_ps(mat_b_rearr[i2][4], alphaReg);
		    	mat_b_rearr[i2][5] = _mm256_mul_ps(mat_b_rearr[i2][5], alphaReg);
		    	mat_b_rearr[i2][6] = _mm256_mul_ps(mat_b_rearr[i2][6], alphaReg);
		    	mat_b_rearr[i2][7] = _mm256_mul_ps(mat_b_rearr[i2][7], alphaReg);
			
			i2++;
		}
		
		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 8) // move across m
		{
			//Broadcast A8,0 to A15,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
			mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
			//Broadcast A21 to A71 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 3));
			mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 4));
			mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 5));
			mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 6));
			mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 7));
			
			//Broadcast A8,2 to A15,2 to registers
			mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 3));
			mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 4));
			mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 5));
			mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 6));
			mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 7));
		
			//Broadcast A8,3 to A15,3 to registers
			mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 3));
			mat_a_blk_elems[28] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 4));
			mat_a_blk_elems[29] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 5));
			mat_a_blk_elems[30] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 6));
			mat_a_blk_elems[31] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 7));
			
			// _mm256_permute2f128_ps()
			
			//Broadcast A8,4 to A15,4 to registers
			mat_a_blk_elems[32] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i));
			mat_a_blk_elems[33] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 1));
			mat_a_blk_elems[34] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 2));
			mat_a_blk_elems[35] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 3));
			mat_a_blk_elems[36] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 4));
			mat_a_blk_elems[37] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 5));
			mat_a_blk_elems[38] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 6));
			mat_a_blk_elems[39] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 7));
			
			//Broadcast A8,5 to A15,5 to registers
			mat_a_blk_elems[40] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i));
			mat_a_blk_elems[41] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 1));
			mat_a_blk_elems[42] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 2));
			mat_a_blk_elems[43] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 3));
			mat_a_blk_elems[44] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 4));
			mat_a_blk_elems[45] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 5));
			mat_a_blk_elems[46] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 6));
			mat_a_blk_elems[47] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 7));
			
			//Broadcast A8,6 to A15,6 to registers
			mat_a_blk_elems[48] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i));
			mat_a_blk_elems[49] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 1));
			mat_a_blk_elems[50] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 2));
			mat_a_blk_elems[51] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 3));
			mat_a_blk_elems[52] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 4));
			mat_a_blk_elems[53] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 5));
			mat_a_blk_elems[54] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 6));
			mat_a_blk_elems[55] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 7));
			
			//Broadcast A8,7 to A15,7 to registers
			mat_a_blk_elems[56] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i));
			mat_a_blk_elems[57] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 1));
			mat_a_blk_elems[58] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 2));
			mat_a_blk_elems[59] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 3));
			mat_a_blk_elems[60] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 4));
			mat_a_blk_elems[61] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 5));
			mat_a_blk_elems[62] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 6));
			mat_a_blk_elems[63] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 7));
						
			i += cs_l_offset[6];
			
			
			for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				i4 = k >> 3;
				
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[1], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[1], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[1], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[2], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[2], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[2], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[2], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[28], mat_b_col[3], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[29], mat_b_col[3], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[30], mat_b_col[3], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[31], mat_b_col[3], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[32], mat_b_col[4], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[33], mat_b_col[4], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[34], mat_b_col[4], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[35], mat_b_col[4], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[36], mat_b_col[4], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[37], mat_b_col[4], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[38], mat_b_col[4], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[39], mat_b_col[4], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[40], mat_b_col[5], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[41], mat_b_col[5], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[42], mat_b_col[5], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[43], mat_b_col[5], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[44], mat_b_col[5], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[45], mat_b_col[5], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[46], mat_b_col[5], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[47], mat_b_col[5], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[48], mat_b_col[6], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[49], mat_b_col[6], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[50], mat_b_col[6], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[51], mat_b_col[6], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[52], mat_b_col[6], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[53], mat_b_col[6], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[54], mat_b_col[6], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[55], mat_b_col[6], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[56], mat_b_col[7], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[57], mat_b_col[7], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[58], mat_b_col[7], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[59], mat_b_col[7], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[60], mat_b_col[7], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[61], mat_b_col[7], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[62], mat_b_col[7], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[63], mat_b_col[7], mat_b_rearr[i4][7]);//d = c - (a*b)

				//end loop of cols					
			}
			i2 += cs_b_offset[6];
		}
		
		//Broadcast A10 to A70 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);
		
		//Broadcast A21 to A71 to registers
		mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	
		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	
		//Broadcast A43 to A73 to registers
		mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	
		//Broadcast A54 to A74 to registers
		mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	
		//Broadcast A65 to A75 to registers
		mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	
		//Broadcast A76 to register
		mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		k = 0;
		for (i = 0; i < numCols_b; i+=8)
		{
			/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
			
			//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[k][0] = _mm256_mul_ps(mat_b_rearr[k][0], mat_a_diag_inv[0]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[k][0], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[k][0], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[k][0], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[k][0], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[k][1] = _mm256_mul_ps(mat_b_rearr[k][1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_rearr[k][1], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_rearr[k][1], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_rearr[k][1], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_rearr[k][1], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[k][2] = _mm256_mul_ps(mat_b_rearr[k][2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_rearr[k][2], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_rearr[k][2], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_rearr[k][2], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_rearr[k][2], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[k][3] = _mm256_mul_ps(mat_b_rearr[k][3], mat_a_diag_inv[3]);

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_rearr[k][3], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_rearr[k][3], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_rearr[k][3], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_rearr[k][3], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
			mat_b_rearr[k][4] = _mm256_mul_ps(mat_b_rearr[k][4], mat_a_diag_inv[4]);

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_rearr[k][4], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_rearr[k][4], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_rearr[k][4], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
			mat_b_rearr[k][5] = _mm256_mul_ps(mat_b_rearr[k][5], mat_a_diag_inv[5]);

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_rearr[k][5], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_rearr[k][5], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
			mat_b_rearr[k][6] = _mm256_mul_ps(mat_b_rearr[k][6], mat_a_diag_inv[6]);

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_rearr[k][6], mat_b_rearr[k][7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
			mat_b_rearr[k][7] = _mm256_mul_ps(mat_b_rearr[k][7], mat_a_diag_inv[7]);

			////////////////////////////////////////////////////////////////////////////////

			//Store the computed B columns

			_mm256_storeu_ps((float *)ptr_b_dup + i, mat_b_rearr[k][0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i), mat_b_rearr[k][4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i), mat_b_rearr[k][5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i), mat_b_rearr[k][6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i), mat_b_rearr[k][7]);
			k++;
		}


	}
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	//float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[16][8];
	//__m256 mat_a_cols_rearr[8];
	__m256 mat_a_blk_elems[64];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0][0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		//(Row0)
		mat_b_col[0] = mat_b_rearr[0][0];

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[2], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[2], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[3], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[3], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[3], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[3], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[4], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[4], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[4], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[5], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[5], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[6], mat_b_rearr[7][0]);//d = c - (a*b)

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];
		i3 += cs_l_offset[6];

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 8)
		{
			i = i1 + k;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[i2][0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[i2][4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[i2][5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[i2][6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[i2][7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
			i2++;
		}
		
		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 8) // move across m
		{
			//Broadcast A8,0 to A15,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
			mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
			//Broadcast A21 to A71 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 3));
			mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 4));
			mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 5));
			mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 6));
			mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 7));
			
			//Broadcast A8,2 to A15,2 to registers
			mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 3));
			mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 4));
			mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 5));
			mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 6));
			mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 7));
		
			//Broadcast A8,3 to A15,3 to registers
			mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 3));
			mat_a_blk_elems[28] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 4));
			mat_a_blk_elems[29] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 5));
			mat_a_blk_elems[30] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 6));
			mat_a_blk_elems[31] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 7));
			
			// _mm256_permute2f128_ps()
			
			//Broadcast A8,4 to A15,4 to registers
			mat_a_blk_elems[32] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i));
			mat_a_blk_elems[33] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 1));
			mat_a_blk_elems[34] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 2));
			mat_a_blk_elems[35] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 3));
			mat_a_blk_elems[36] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 4));
			mat_a_blk_elems[37] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 5));
			mat_a_blk_elems[38] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 6));
			mat_a_blk_elems[39] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 7));
			
			//Broadcast A8,5 to A15,5 to registers
			mat_a_blk_elems[40] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i));
			mat_a_blk_elems[41] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 1));
			mat_a_blk_elems[42] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 2));
			mat_a_blk_elems[43] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 3));
			mat_a_blk_elems[44] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 4));
			mat_a_blk_elems[45] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 5));
			mat_a_blk_elems[46] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 6));
			mat_a_blk_elems[47] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 7));
			
			//Broadcast A8,6 to A15,6 to registers
			mat_a_blk_elems[48] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i));
			mat_a_blk_elems[49] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 1));
			mat_a_blk_elems[50] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 2));
			mat_a_blk_elems[51] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 3));
			mat_a_blk_elems[52] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 4));
			mat_a_blk_elems[53] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 5));
			mat_a_blk_elems[54] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 6));
			mat_a_blk_elems[55] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 7));
			
			//Broadcast A8,7 to A15,7 to registers
			mat_a_blk_elems[56] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i));
			mat_a_blk_elems[57] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 1));
			mat_a_blk_elems[58] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 2));
			mat_a_blk_elems[59] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 3));
			mat_a_blk_elems[60] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 4));
			mat_a_blk_elems[61] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 5));
			mat_a_blk_elems[62] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 6));
			mat_a_blk_elems[63] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 7));
						
			i += cs_l_offset[6];
			
			for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				i4 = k >> 3;
				
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[1], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[1], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[1], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[2], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[2], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[2], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[2], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[28], mat_b_col[3], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[29], mat_b_col[3], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[30], mat_b_col[3], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[31], mat_b_col[3], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[32], mat_b_col[4], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[33], mat_b_col[4], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[34], mat_b_col[4], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[35], mat_b_col[4], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[36], mat_b_col[4], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[37], mat_b_col[4], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[38], mat_b_col[4], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[39], mat_b_col[4], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[40], mat_b_col[5], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[41], mat_b_col[5], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[42], mat_b_col[5], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[43], mat_b_col[5], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[44], mat_b_col[5], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[45], mat_b_col[5], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[46], mat_b_col[5], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[47], mat_b_col[5], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[48], mat_b_col[6], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[49], mat_b_col[6], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[50], mat_b_col[6], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[51], mat_b_col[6], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[52], mat_b_col[6], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[53], mat_b_col[6], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[54], mat_b_col[6], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[55], mat_b_col[6], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[56], mat_b_col[7], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[57], mat_b_col[7], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[58], mat_b_col[7], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[59], mat_b_col[7], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[60], mat_b_col[7], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[61], mat_b_col[7], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[62], mat_b_col[7], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[63], mat_b_col[7], mat_b_rearr[i4][7]);//d = c - (a*b)

				//end loop of cols					
			}
			i2 += cs_b_offset[6];
		}
		
		//Broadcast A10 to A70 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
				
		//Broadcast A21 to A71 to registers
		mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A43 to A73 to registers
		mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A54 to A74 to registers
		mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A65 to A75 to registers
		mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A76 to register
		mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
		k = 0;
		for (i = 0; i < numCols_b; i+=8)
		{
			/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
			
			//(Row0): already done

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[k][0], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[k][0], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[k][0], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[k][0], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_rearr[k][1], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_rearr[k][1], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_rearr[k][1], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_rearr[k][1], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_rearr[k][2], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_rearr[k][2], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_rearr[k][2], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_rearr[k][2], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_rearr[k][3], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_rearr[k][3], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_rearr[k][3], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_rearr[k][3], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_rearr[k][4], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_rearr[k][4], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_rearr[k][4], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_rearr[k][5], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_rearr[k][5], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_rearr[k][6], mat_b_rearr[k][7]);//d = c - (a*b)

			////////////////////////////////////////////////////////////////////////////////

			//Store the computed B columns

			_mm256_storeu_ps((float *)ptr_b_dup + i, mat_b_rearr[k][0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i), mat_b_rearr[k][4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i), mat_b_rearr[k][5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i), mat_b_rearr[k][6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i), mat_b_rearr[k][7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
		}


	}
	///////////////////loop ends /////////////////////
}

static void trsm_XAtB_block_allSmallSizedMatrices_alpha_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	//float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[16][8];
	//__m256 mat_a_cols_rearr[8];
	__m256 mat_a_blk_elems[64];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];
	__m256 alphaReg;
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0][0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7][0] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		mat_b_rearr[0][0] = _mm256_mul_ps(mat_b_rearr[0][0], alphaReg);
		mat_b_rearr[1][0] = _mm256_mul_ps(mat_b_rearr[1][0], alphaReg);
		mat_b_rearr[2][0] = _mm256_mul_ps(mat_b_rearr[2][0], alphaReg);
		mat_b_rearr[3][0] = _mm256_mul_ps(mat_b_rearr[3][0], alphaReg);
		mat_b_rearr[4][0] = _mm256_mul_ps(mat_b_rearr[4][0], alphaReg);
		mat_b_rearr[5][0] = _mm256_mul_ps(mat_b_rearr[5][0], alphaReg);
		mat_b_rearr[6][0] = _mm256_mul_ps(mat_b_rearr[6][0], alphaReg);
		mat_b_rearr[7][0] = _mm256_mul_ps(mat_b_rearr[7][0], alphaReg);
		
		//(Row0)
		mat_b_col[0] = mat_b_rearr[0][0];

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[1][0]);//d = c - (a*b)
		mat_b_rearr[2][0] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[2][0]);//d = c - (a*b)
		mat_b_rearr[3][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[2], mat_b_rearr[3][0]);//d = c - (a*b)
		mat_b_rearr[4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[2], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[2], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[3], mat_b_rearr[4][0]);//d = c - (a*b)
		mat_b_rearr[5][0] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[3], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[3], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[3], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[4], mat_b_rearr[5][0]);//d = c - (a*b)
		mat_b_rearr[6][0] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[4], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[4], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[5], mat_b_rearr[6][0]);//d = c - (a*b)
		mat_b_rearr[7][0] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[5], mat_b_rearr[7][0]);//d = c - (a*b)

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[6], mat_b_rearr[7][0]);//d = c - (a*b)

		////////////////////////////////////////////////////////////////////////////////

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_col[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_col[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_col[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_col[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_col[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_col[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_col[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_col[7]);

		//i += cs_b_offset[6];
		//ptr_b_dup += cs_b_offset[6];
		i += 8;
		ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += 8;
		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += cs_b_offset[6];
		i1 += cs_b_offset[6];
		i3 += cs_l_offset[6];

		i = 0;
		i2 = 0;
		for (k = 0; k < numCols_b; k += 8)
		{
			i = i1 + k;
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[i2][0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[i2][1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[i2][2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[i2][3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[i2][4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[i2][5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[i2][6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[i2][7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));
			
			mat_b_rearr[i2][0] = _mm256_mul_ps(mat_b_rearr[i2][0], alphaReg);
		        mat_b_rearr[i2][1] = _mm256_mul_ps(mat_b_rearr[i2][1], alphaReg);
		    	mat_b_rearr[i2][2] = _mm256_mul_ps(mat_b_rearr[i2][2], alphaReg);
		    	mat_b_rearr[i2][3] = _mm256_mul_ps(mat_b_rearr[i2][3], alphaReg);
		    	mat_b_rearr[i2][4] = _mm256_mul_ps(mat_b_rearr[i2][4], alphaReg);
		    	mat_b_rearr[i2][5] = _mm256_mul_ps(mat_b_rearr[i2][5], alphaReg);
		    	mat_b_rearr[i2][6] = _mm256_mul_ps(mat_b_rearr[i2][6], alphaReg);
		    	mat_b_rearr[i2][7] = _mm256_mul_ps(mat_b_rearr[i2][7], alphaReg);
			
			i2++;
		}
		
		i = 0;
		i2 = 0;
		for (l = 0; l < j; l += 8) // move across m
		{
			//Broadcast A8,0 to A15,0 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
			mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
			//Broadcast A21 to A71 to registers
			mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i));
			mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 1));
			mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 2));
			mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 3));
			mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 4));
			mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 5));
			mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 6));
			mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + i + 7));
			
			//Broadcast A8,2 to A15,2 to registers
			mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i));
			mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 1));
			mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 2));
			mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 3));
			mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 4));
			mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 5));
			mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 6));
			mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + i + 7));
		
			//Broadcast A8,3 to A15,3 to registers
			mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i));
			mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 1));
			mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 2));
			mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 3));
			mat_a_blk_elems[28] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 4));
			mat_a_blk_elems[29] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 5));
			mat_a_blk_elems[30] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 6));
			mat_a_blk_elems[31] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + i + 7));
			
			// _mm256_permute2f128_ps()
			
			//Broadcast A8,4 to A15,4 to registers
			mat_a_blk_elems[32] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i));
			mat_a_blk_elems[33] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 1));
			mat_a_blk_elems[34] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 2));
			mat_a_blk_elems[35] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 3));
			mat_a_blk_elems[36] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 4));
			mat_a_blk_elems[37] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 5));
			mat_a_blk_elems[38] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 6));
			mat_a_blk_elems[39] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + i + 7));
			
			//Broadcast A8,5 to A15,5 to registers
			mat_a_blk_elems[40] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i));
			mat_a_blk_elems[41] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 1));
			mat_a_blk_elems[42] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 2));
			mat_a_blk_elems[43] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 3));
			mat_a_blk_elems[44] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 4));
			mat_a_blk_elems[45] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 5));
			mat_a_blk_elems[46] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 6));
			mat_a_blk_elems[47] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + i + 7));
			
			//Broadcast A8,6 to A15,6 to registers
			mat_a_blk_elems[48] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i));
			mat_a_blk_elems[49] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 1));
			mat_a_blk_elems[50] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 2));
			mat_a_blk_elems[51] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 3));
			mat_a_blk_elems[52] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 4));
			mat_a_blk_elems[53] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 5));
			mat_a_blk_elems[54] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 6));
			mat_a_blk_elems[55] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + i + 7));
			
			//Broadcast A8,7 to A15,7 to registers
			mat_a_blk_elems[56] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i));
			mat_a_blk_elems[57] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 1));
			mat_a_blk_elems[58] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 2));
			mat_a_blk_elems[59] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 3));
			mat_a_blk_elems[60] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 4));
			mat_a_blk_elems[61] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 5));
			mat_a_blk_elems[62] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 6));
			mat_a_blk_elems[63] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5] + i + 7));
						
			i += cs_l_offset[6];
			
			for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
			{
				/////////////////// Partial Lower 8x8 block trsm of B

				i4 = i2 + k;
				//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				i4 = k >> 3;
				
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_col[1], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_col[1], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_col[1], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_col[1], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_col[1], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_col[1], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_col[1], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_col[1], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_col[2], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_col[2], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_col[2], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_col[2], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_col[2], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_col[2], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_col[2], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_col[2], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_col[3], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_col[3], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_col[3], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_col[3], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[28], mat_b_col[3], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[29], mat_b_col[3], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[30], mat_b_col[3], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[31], mat_b_col[3], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[32], mat_b_col[4], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[33], mat_b_col[4], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[34], mat_b_col[4], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[35], mat_b_col[4], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[36], mat_b_col[4], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[37], mat_b_col[4], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[38], mat_b_col[4], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[39], mat_b_col[4], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[40], mat_b_col[5], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[41], mat_b_col[5], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[42], mat_b_col[5], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[43], mat_b_col[5], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[44], mat_b_col[5], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[45], mat_b_col[5], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[46], mat_b_col[5], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[47], mat_b_col[5], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[48], mat_b_col[6], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[49], mat_b_col[6], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[50], mat_b_col[6], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[51], mat_b_col[6], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[52], mat_b_col[6], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[53], mat_b_col[6], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[54], mat_b_col[6], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[55], mat_b_col[6], mat_b_rearr[i4][7]);//d = c - (a*b)

				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[i4][0] = _mm256_fnmadd_ps(mat_a_blk_elems[56], mat_b_col[7], mat_b_rearr[i4][0]);//d = c - (a*b)
				mat_b_rearr[i4][1] = _mm256_fnmadd_ps(mat_a_blk_elems[57], mat_b_col[7], mat_b_rearr[i4][1]);//d = c - (a*b)
				mat_b_rearr[i4][2] = _mm256_fnmadd_ps(mat_a_blk_elems[58], mat_b_col[7], mat_b_rearr[i4][2]);//d = c - (a*b)
				mat_b_rearr[i4][3] = _mm256_fnmadd_ps(mat_a_blk_elems[59], mat_b_col[7], mat_b_rearr[i4][3]);//d = c - (a*b)
				mat_b_rearr[i4][4] = _mm256_fnmadd_ps(mat_a_blk_elems[60], mat_b_col[7], mat_b_rearr[i4][4]);//d = c - (a*b)
				mat_b_rearr[i4][5] = _mm256_fnmadd_ps(mat_a_blk_elems[61], mat_b_col[7], mat_b_rearr[i4][5]);//d = c - (a*b)
				mat_b_rearr[i4][6] = _mm256_fnmadd_ps(mat_a_blk_elems[62], mat_b_col[7], mat_b_rearr[i4][6]);//d = c - (a*b)
				mat_b_rearr[i4][7] = _mm256_fnmadd_ps(mat_a_blk_elems[63], mat_b_col[7], mat_b_rearr[i4][7]);//d = c - (a*b)

				//end loop of cols					
			}
			i2 += cs_b_offset[6];
		}
		
		//Broadcast A10 to A70 to registers
		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + i + 1));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
				
		//Broadcast A21 to A71 to registers
		mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + i + 2));
		mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A32 to A72 to registers
		mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + i + 3));
		mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A43 to A73 to registers
		mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + i + 4));
		mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A54 to A74 to registers
		mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + i + 5));
		mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A65 to A75 to registers
		mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + i + 6));
		mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		i += cs_l;
			
		//Broadcast A76 to register
		mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + i + 7));
		
		k = 0;
		for (i = 0; i < numCols_b; i+=8)
		{
			/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
			
			//(Row0): already done

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[k][1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[k][0], mat_b_rearr[k][1]);//d = c - (a*b)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[k][0], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[k][0], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[k][0], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[k][0], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[k][0], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[k][0], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[k][2] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_rearr[k][1], mat_b_rearr[k][2]);//d = c - (a*b)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[8], mat_b_rearr[k][1], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[9], mat_b_rearr[k][1], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[10], mat_b_rearr[k][1], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[11], mat_b_rearr[k][1], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[12], mat_b_rearr[k][1], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[k][3] = _mm256_fnmadd_ps(mat_a_blk_elems[13], mat_b_rearr[k][2], mat_b_rearr[k][3]);//d = c - (a*b)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[14], mat_b_rearr[k][2], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[15], mat_b_rearr[k][2], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[16], mat_b_rearr[k][2], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[17], mat_b_rearr[k][2], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[k][4] = _mm256_fnmadd_ps(mat_a_blk_elems[18], mat_b_rearr[k][3], mat_b_rearr[k][4]);//d = c - (a*b)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[19], mat_b_rearr[k][3], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[20], mat_b_rearr[k][3], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[21], mat_b_rearr[k][3], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[k][5] = _mm256_fnmadd_ps(mat_a_blk_elems[22], mat_b_rearr[k][4], mat_b_rearr[k][5]);//d = c - (a*b)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[23], mat_b_rearr[k][4], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[24], mat_b_rearr[k][4], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[k][6] = _mm256_fnmadd_ps(mat_a_blk_elems[25], mat_b_rearr[k][5], mat_b_rearr[k][6]);//d = c - (a*b)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[26], mat_b_rearr[k][5], mat_b_rearr[k][7]);//d = c - (a*b)

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[k][7] = _mm256_fnmadd_ps(mat_a_blk_elems[27], mat_b_rearr[k][6], mat_b_rearr[k][7]);//d = c - (a*b)

			////////////////////////////////////////////////////////////////////////////////

			//Store the computed B columns

			_mm256_storeu_ps((float *)ptr_b_dup + i, mat_b_rearr[k][0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b) + i), mat_b_rearr[k][1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i), mat_b_rearr[k][2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i), mat_b_rearr[k][3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i), mat_b_rearr[k][4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i), mat_b_rearr[k][5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i), mat_b_rearr[k][6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i), mat_b_rearr[k][7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
		}


	}
	///////////////////loop ends /////////////////////
}
#endif //OPT_CACHE_BLOCKING_L1

//////////////////////////// AutX=B ///////////////////////
static void trsm_AutXB_block_allSmallSizedMatrices(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);
#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif
	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		/* transpose steps start */
		////unpacklow////
		mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
		/* transpose steps end */


		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], mat_a_diag_inv[0]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4]));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5]));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], mat_a_diag_inv[1]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[0]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[1]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[2]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[3]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[4]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[5]));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], mat_a_diag_inv[2]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[1]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[2]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[3]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[4]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[5]));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], mat_a_diag_inv[3]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[2]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[3]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[4]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[5]));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], mat_a_diag_inv[4]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[3]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[4]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[5]));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], mat_a_diag_inv[5]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[4]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[5]));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], mat_a_diag_inv[6]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 6 + cs_l_offset[5]));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		/* transpose steps start */
		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
		/* transpose steps end */

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_rearr[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_rearr[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_rearr[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_rearr[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_rearr[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_rearr[7]);

		i += cs_b_offset[6];
		ptr_b_dup += cs_b_offset[6];
		//i += 8;
		//ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += cs_l_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += 8;
		mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += 8;
		i1 += 8;
		i = i1;
		i2 = 0;

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
			////unpacklow////
			mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

			////unpackhigh////
			mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
			/* transpose steps end */
#endif
			//i = 0;
			ptr_l_dup = ptr_l;
			i4 = i2;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				//{
					/////////////////// Partial Lower 8x8 block trsm of B
					//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				/* transpose steps start */
		////unpacklow////
				mat_b_col[0] = _mm256_unpacklo_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_b_col[1] = _mm256_unpacklo_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_b_col[2] = _mm256_unpacklo_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_b_col[3] = _mm256_unpacklo_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange low elements
#if REARRANGE_SHFL == 1
				mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
				mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
				mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
				mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
				mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
				mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
				//Merge rearranged low elements into complete rows
				mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
				mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
				mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
				mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

				////unpackhigh////
				mat_a_blk_elems[0] = _mm256_unpackhi_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_a_blk_elems[1] = _mm256_unpackhi_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_a_blk_elems[2] = _mm256_unpackhi_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_a_blk_elems[3] = _mm256_unpackhi_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange high elements
#if REARRANGE_SHFL == 1
				mat_a_blk_elems[4] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x44);
				mat_a_blk_elems[5] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xEE);
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x44);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xEE);
#else
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x4E);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x4E);
				mat_a_blk_elems[4] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[6], 0xCC);
				mat_a_blk_elems[5] = _mm256_blend_ps(mat_a_blk_elems[1], mat_a_blk_elems[6], 0x33);
				mat_a_blk_elems[6] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[7], 0xCC);
				mat_a_blk_elems[7] = _mm256_blend_ps(mat_a_blk_elems[3], mat_a_blk_elems[7], 0x33);
#endif

				//Merge rearranged high elements into complete rows
				mat_b_col[2] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x20);
				mat_b_col[6] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x31);
				mat_b_col[3] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x20);
				mat_b_col[7] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x31);
				/* transpose steps end */

						//Broadcast A8,0 to A15,0 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				//i4 = k >> 3;
				ptr_l_dup++;

#if GEMM_ACCUM_A
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
				mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
				mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
				mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
				mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
				mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
				mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
				mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				//}
				//i2 += cs_b_offset[6];
				i4 += 8;
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			//{
				//i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */
#endif
				//Broadcast A10 to A70 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
			//i += cs_l;

#if GEMM_ACCUM_A
				//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#else
			mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#endif

#if GEMM_ACCUM_A
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
			mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
			mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
			mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
			mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
			mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
			mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
			mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[0]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[1]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[2]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[3]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[4]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A32 to A72 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[1]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[2]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[3]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[4]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A43 to A73 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[2]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[3]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[4]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A54 to A74 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[3]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[4]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
			mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A65 to A75 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[4]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
			mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A76 to register
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 6 + cs_l_offset[5]));

			//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
			mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
			mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

			////////////////////////////////////////////////////////////////////////////////

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */

					//Store the computed B columns
			_mm256_storeu_ps((float *)ptr_b_dup + i2, mat_b_col[0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+i2), mat_b_col[1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i2), mat_b_col[2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i2), mat_b_col[3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i2), mat_b_col[4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i2), mat_b_col[5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i2), mat_b_col[6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i2), mat_b_col[7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
			//}
			i += cs_b_offset[6];
			i2 += cs_b_offset[6];
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_AutXB_block_allSmallSizedMatrices_alpha(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	float ones = 1.0;
	int i, i1, i2, i3, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	__m256 mat_a_diag_inv[8];
	__m256 reciprocal_diags[2];
	__m256 alphaReg;

	reciprocal_diags[0] = _mm256_broadcast_ss((float const *)(&ones));
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	//read diag elems of L 16x16 block
	mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l);
	mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + cs_l);
	mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[0]);
	mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[1]);
	mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[2]);
	mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[3]);
	mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[4]);
	mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + cs_l_offset[5]);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

	reciprocal_diags[1] = reciprocal_diags[0];

	//pack first 8 diags together
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
	mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
	mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
	mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
	mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

	//reciprocal of diagnal elements 0,1,2,3,4,5,6,7
	reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);
#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif
	//extract diag a00 from a
	mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
	//mat_a_diag_inv[0] = _mm256_unpacklo_ps(mat_a_diag_inv[0], mat_a_diag_inv[0]);
	//extract diag a11 from a
	mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
	//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);
	//extract diag a22 from a
	mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
	//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);
	//extract diag a33 from a
	mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
	//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);
	//extract diag a44 from a
	mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
	mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
	//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);
	//extract diag a55 from a
	mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
	mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
	//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);
	//extract diag a66 from a
	mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
	mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
	//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);
	//extract diag a77 from a
	mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
	mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
	//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		/* transpose steps start */
		////unpacklow////
		mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
		/* transpose steps end */

		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);

		//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], mat_a_diag_inv[0]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4]));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5]));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], mat_a_diag_inv[1]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[0]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[1]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[2]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[3]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[4]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[5]));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], mat_a_diag_inv[2]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[1]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[2]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[3]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[4]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[5]));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], mat_a_diag_inv[3]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[2]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[3]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[4]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[5]));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], mat_a_diag_inv[4]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[3]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[4]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[5]));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], mat_a_diag_inv[5]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[4]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[5]));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], mat_a_diag_inv[6]);

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 6 + cs_l_offset[5]));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)

		//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], mat_a_diag_inv[7]);

		////////////////////////////////////////////////////////////////////////////////

		/* transpose steps start */
		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
		/* transpose steps end */

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_rearr[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_rearr[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_rearr[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_rearr[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_rearr[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_rearr[7]);

		i += cs_b_offset[6];
		ptr_b_dup += cs_b_offset[6];
		//i += 8;
		//ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i3 = 0;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += cs_l_offset[6];

		//Read next 8x8 block of A to get diag elements
		i3 += 8;
		mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_l + i3);
		mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l);
		mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[0]);
		mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[1]);
		mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[2]);
		mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[3]);
		mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[4]);
		mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)ptr_l + i3 + cs_l_offset[5]);

		//pack 8 diags of A together
		reciprocal_diags[0] = reciprocal_diags[1];
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xAA);//diag 0,1
		mat_a_diag_inv[1] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xAA);//diag 2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_blk_elems[4], mat_a_blk_elems[5], 0xAA);//diag 4,5
		mat_a_diag_inv[3] = _mm256_blend_ps(mat_a_blk_elems[6], mat_a_blk_elems[7], 0xAA);//diag 6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[1], 0xCC);//diag 0,1,2,3
		mat_a_diag_inv[2] = _mm256_blend_ps(mat_a_diag_inv[2], mat_a_diag_inv[3], 0xCC);//diag 4,5,6,7
		mat_a_diag_inv[0] = _mm256_blend_ps(mat_a_diag_inv[0], mat_a_diag_inv[2], 0xF0);//diag 0,1,2,3,4,5,6,7

		//reciprocal of diagnal elements of A :- 0,1,2,3,4,5,6,7
		reciprocal_diags[0] = _mm256_div_ps(reciprocal_diags[0], mat_a_diag_inv[0]);

		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += 8;
		i1 += 8;
		i = i1;
		i2 = 0;

		//extract diag a00 from a
		mat_a_diag_inv[0] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[0] = _mm256_permute2f128_ps(mat_a_diag_inv[0], mat_a_diag_inv[0], 0x00);
		//mat_a_diag_inv2[0] = _mm256_unpacklo_ps(mat_a_diag_inv2[0], mat_a_diag_inv2[0]);

		//extract diag a11 from a
		mat_a_diag_inv[1] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[1] = _mm256_permute2f128_ps(mat_a_diag_inv[1], mat_a_diag_inv[1], 0x00);
		//mat_a_diag_inv[1] = _mm256_unpacklo_ps(mat_a_diag_inv[1], mat_a_diag_inv[1]);

		//extract diag a22 from a
		mat_a_diag_inv[2] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[2] = _mm256_permute2f128_ps(mat_a_diag_inv[2], mat_a_diag_inv[2], 0x00);
		//mat_a_diag_inv[2] = _mm256_unpacklo_ps(mat_a_diag_inv[2], mat_a_diag_inv[2]);

		//extract diag a33 from a
		mat_a_diag_inv[3] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[3] = _mm256_permute2f128_ps(mat_a_diag_inv[3], mat_a_diag_inv[3], 0x00);
		//mat_a_diag_inv[3] = _mm256_unpacklo_ps(mat_a_diag_inv[3], mat_a_diag_inv[3]);

		//extract diag a44 from a
		mat_a_diag_inv[4] = _mm256_permute_ps(reciprocal_diags[0], 0x00);
		mat_a_diag_inv[4] = _mm256_permute2f128_ps(mat_a_diag_inv[4], mat_a_diag_inv[4], 0x11);
		//mat_a_diag_inv[4] = _mm256_unpacklo_ps(mat_a_diag_inv[4], mat_a_diag_inv[4]);

		//extract diag a55 from a
		mat_a_diag_inv[5] = _mm256_permute_ps(reciprocal_diags[0], 0x55);
		mat_a_diag_inv[5] = _mm256_permute2f128_ps(mat_a_diag_inv[5], mat_a_diag_inv[5], 0x11);
		//mat_a_diag_inv[5] = _mm256_unpacklo_ps(mat_a_diag_inv[5], mat_a_diag_inv[5]);

		//extract diag a66 from a
		mat_a_diag_inv[6] = _mm256_permute_ps(reciprocal_diags[0], 0xAA);
		mat_a_diag_inv[6] = _mm256_permute2f128_ps(mat_a_diag_inv[6], mat_a_diag_inv[6], 0x11);
		//mat_a_diag_inv[6] = _mm256_unpacklo_ps(mat_a_diag_inv[6], mat_a_diag_inv[6]);

		//extract diag a77 from a
		mat_a_diag_inv[7] = _mm256_permute_ps(reciprocal_diags[0], 0xFF);
		mat_a_diag_inv[7] = _mm256_permute2f128_ps(mat_a_diag_inv[7], mat_a_diag_inv[7], 0x11);
		//mat_a_diag_inv[7] = _mm256_unpacklo_ps(mat_a_diag_inv[7], mat_a_diag_inv[7]);

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
			////unpacklow////
			mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

			////unpackhigh////
			mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
			/* transpose steps end */
			
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
	        	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
	    		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
	    		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
	    		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
	    		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
	    		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
	    		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);
#endif
		
			//i = 0;
			ptr_l_dup = ptr_l;
			i4 = i2;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				//{
					/////////////////// Partial Lower 8x8 block trsm of B
					//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				/* transpose steps start */
		////unpacklow////
				mat_b_col[0] = _mm256_unpacklo_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_b_col[1] = _mm256_unpacklo_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_b_col[2] = _mm256_unpacklo_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_b_col[3] = _mm256_unpacklo_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange low elements
#if REARRANGE_SHFL == 1
				mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
				mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
				mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
				mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
				mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
				mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
				//Merge rearranged low elements into complete rows
				mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
				mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
				mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
				mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

				////unpackhigh////
				mat_a_blk_elems[0] = _mm256_unpackhi_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_a_blk_elems[1] = _mm256_unpackhi_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_a_blk_elems[2] = _mm256_unpackhi_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_a_blk_elems[3] = _mm256_unpackhi_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange high elements
#if REARRANGE_SHFL == 1
				mat_a_blk_elems[4] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x44);
				mat_a_blk_elems[5] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xEE);
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x44);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xEE);
#else
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x4E);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x4E);
				mat_a_blk_elems[4] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[6], 0xCC);
				mat_a_blk_elems[5] = _mm256_blend_ps(mat_a_blk_elems[1], mat_a_blk_elems[6], 0x33);
				mat_a_blk_elems[6] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[7], 0xCC);
				mat_a_blk_elems[7] = _mm256_blend_ps(mat_a_blk_elems[3], mat_a_blk_elems[7], 0x33);
#endif

				//Merge rearranged high elements into complete rows
				mat_b_col[2] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x20);
				mat_b_col[6] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x31);
				mat_b_col[3] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x20);
				mat_b_col[7] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x31);
				/* transpose steps end */

						//Broadcast A8,0 to A15,0 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				//i4 = k >> 3;
				ptr_l_dup++;

#if GEMM_ACCUM_A
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
				mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
				mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
				mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
				mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
				mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
				mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
				mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				//}
				//i2 += cs_b_offset[6];
				i4 += 8;
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			//{
				//i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */
			
			mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
			mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
			mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
			mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
			mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
			mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
			mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
			mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);
#endif
				//Broadcast A10 to A70 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
			//i += cs_l;

#if GEMM_ACCUM_A
				//(Row0): Perform mul operation of reciprocal of L(0,0) element with 1st row elements of B
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#else
			mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], mat_a_diag_inv[0]);
#endif

#if GEMM_ACCUM_A
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
			mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
			mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
			mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
			mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
			mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
			mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
			mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[0]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[1]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[2]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[3]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[4]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(1,1) element with 2nd row elements of B
			mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], mat_a_diag_inv[1]);

			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A32 to A72 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[1]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[2]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[3]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[4]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(2, 2) element with 3rd row elements of B
			mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], mat_a_diag_inv[2]);

			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A43 to A73 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[2]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[3]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[4]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(3, 3) element with 4rth row elements of B
			mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], mat_a_diag_inv[3]);

			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A54 to A74 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[3]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[4]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(4, 4) element with 4rth row elements of B
			mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], mat_a_diag_inv[4]);

			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A65 to A75 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[4]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[5]));
			//i += cs_l;

			//Perform mul operation of reciprocal of L(5, 5) element with 5th row elements of B
			mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], mat_a_diag_inv[5]);

			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A76 to register
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 6 + cs_l_offset[5]));

			//Perform mul operation of reciprocal of L(6, 6) element with 6th row elements of B
			mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], mat_a_diag_inv[6]);

			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)

			//Perform mul operation of reciprocal of L(7, 7) element with 7th row elements of B
			mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], mat_a_diag_inv[7]);

			////////////////////////////////////////////////////////////////////////////////

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */

					//Store the computed B columns
			_mm256_storeu_ps((float *)ptr_b_dup + i2, mat_b_col[0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+i2), mat_b_col[1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i2), mat_b_col[2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i2), mat_b_col[3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i2), mat_b_col[4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i2), mat_b_col[5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i2), mat_b_col[6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i2), mat_b_col[7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
			//}
			i += cs_b_offset[6];
			i2 += cs_b_offset[6];
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_AutXB_block_allSmallSizedMatrices_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b)
{
	//float ones = 1.0;
	int i, i1, i2, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		/* transpose steps start */
		////unpacklow////
		mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
		/* transpose steps end */


		//(Row0)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4]));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5]));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[0]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[1]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[2]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[3]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[4]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[5]));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[1]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[2]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[3]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[4]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[5]));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[2]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[3]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[4]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[5]));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[3]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[4]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[5]));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[4]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[5]));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 6 + cs_l_offset[5]));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)



		////////////////////////////////////////////////////////////////////////////////

		/* transpose steps start */
		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
		/* transpose steps end */

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_rearr[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_rearr[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_rearr[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_rearr[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_rearr[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_rearr[7]);

		i += cs_b_offset[6];
		ptr_b_dup += cs_b_offset[6];
		//i += 8;
		//ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += cs_l_offset[6];


		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += 8;
		i1 += 8;
		i = i1;
		i2 = 0;

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
			////unpacklow////
			mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

			////unpackhigh////
			mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
			/* transpose steps end */
#endif
		
			//i = 0;
			ptr_l_dup = ptr_l;
			i4 = i2;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				//{
					/////////////////// Partial Lower 8x8 block trsm of B
					//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				/* transpose steps start */
		////unpacklow////
				mat_b_col[0] = _mm256_unpacklo_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_b_col[1] = _mm256_unpacklo_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_b_col[2] = _mm256_unpacklo_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_b_col[3] = _mm256_unpacklo_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange low elements
#if REARRANGE_SHFL == 1
				mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
				mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
				mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
				mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
				mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
				mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
				//Merge rearranged low elements into complete rows
				mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
				mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
				mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
				mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

				////unpackhigh////
				mat_a_blk_elems[0] = _mm256_unpackhi_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_a_blk_elems[1] = _mm256_unpackhi_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_a_blk_elems[2] = _mm256_unpackhi_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_a_blk_elems[3] = _mm256_unpackhi_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange high elements
#if REARRANGE_SHFL == 1
				mat_a_blk_elems[4] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x44);
				mat_a_blk_elems[5] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xEE);
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x44);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xEE);
#else
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x4E);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x4E);
				mat_a_blk_elems[4] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[6], 0xCC);
				mat_a_blk_elems[5] = _mm256_blend_ps(mat_a_blk_elems[1], mat_a_blk_elems[6], 0x33);
				mat_a_blk_elems[6] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[7], 0xCC);
				mat_a_blk_elems[7] = _mm256_blend_ps(mat_a_blk_elems[3], mat_a_blk_elems[7], 0x33);
#endif

				//Merge rearranged high elements into complete rows
				mat_b_col[2] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x20);
				mat_b_col[6] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x31);
				mat_b_col[3] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x20);
				mat_b_col[7] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x31);
				/* transpose steps end */

						//Broadcast A8,0 to A15,0 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				//i4 = k >> 3;
				ptr_l_dup++;

#if GEMM_ACCUM_A
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
				mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
				mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
				mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
				mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
				mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
				mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
				mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				//}
				//i2 += cs_b_offset[6];
				i4 += 8;
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			//{
				//i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */
#endif
				//Broadcast A10 to A70 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
			//i += cs_l;

#if GEMM_ACCUM_A
			//(Row0): already done

#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
#endif

#if GEMM_ACCUM_A
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
			mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
			mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
			mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
			mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
			mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
			mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
			mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[0]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[1]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[2]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[3]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[4]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[5]));
			//i += cs_l;



			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A32 to A72 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[1]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[2]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[3]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[4]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[5]));
			//i += cs_l;



			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A43 to A73 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[2]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[3]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[4]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[5]));
			//i += cs_l;



			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A54 to A74 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[3]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[4]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[5]));
			//i += cs_l;



			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A65 to A75 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[4]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[5]));
			//i += cs_l;



			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A76 to register
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 6 + cs_l_offset[5]));



			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)



			////////////////////////////////////////////////////////////////////////////////

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */

					//Store the computed B columns
			_mm256_storeu_ps((float *)ptr_b_dup + i2, mat_b_col[0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+i2), mat_b_col[1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i2), mat_b_col[2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i2), mat_b_col[3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i2), mat_b_col[4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i2), mat_b_col[5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i2), mat_b_col[6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i2), mat_b_col[7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
			//}
			i += cs_b_offset[6];
			i2 += cs_b_offset[6];
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}

static void trsm_AutXB_block_allSmallSizedMatrices_alpha_unitDiag(float *ptr_l, float *ptr_b, int numRows_lb, int numCols_b, int rs_l, int rs_b, int cs_l, int cs_b, float alpha)
{
	//float ones = 1.0;
	int i, i1, i2, i4, j, k, l, r;
	int cs_b_offset[7];
	int cs_l_offset[7];
	float *ptr_b_dup, *ptr_l_dup;

	//57 number of ymm(256 bits) registers used
	__m256 mat_b_col[8];
	__m256 mat_b_rearr[8];
	__m256 mat_a_blk_elems[8];
	//__m256 mat_a_diag_inv[8];
	//__m256 reciprocal_diags[2];
	__m256 alphaReg;
	alphaReg = _mm256_broadcast_ss((float const *)&alpha);

	// ---> considering that the matrix size is multiple of 16 rows and 8 cols <--- //

	//L matrix offsets
	cs_l_offset[0] = (cs_l << 1);
	cs_l_offset[1] = cs_l + cs_l_offset[0];
	cs_l_offset[2] = (cs_l << 2);
	cs_l_offset[3] = cs_l + cs_l_offset[2];
	cs_l_offset[4] = cs_l_offset[0] + cs_l_offset[2];
	cs_l_offset[5] = cs_l + cs_l_offset[4];
	cs_l_offset[6] = (cs_l_offset[5] + cs_l);

	cs_b_offset[0] = (cs_b << 1);
	cs_b_offset[1] = cs_b + cs_b_offset[0];
	cs_b_offset[2] = (cs_b << 2);
	cs_b_offset[3] = cs_b + cs_b_offset[2];
	cs_b_offset[4] = cs_b_offset[0] + cs_b_offset[2];
	cs_b_offset[5] = cs_b + cs_b_offset[4];
	cs_b_offset[6] = (cs_b_offset[5] + cs_b);

#if 0
	//Broadcast A10 to A70 to registers
	mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1));
	mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2));
	mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3));
	mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 4));
	mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 5));
	mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 6));
	mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + 7));

	//Broadcast A21 to A71 to registers
	mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 2));
	mat_a_blk_elems[8] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 3));
	mat_a_blk_elems[9] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 4));
	mat_a_blk_elems[10] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 5));
	mat_a_blk_elems[11] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 6));
	mat_a_blk_elems[12] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l + 7));

	//Broadcast A32 to A72 to registers
	mat_a_blk_elems[13] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 3));
	mat_a_blk_elems[14] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 4));
	mat_a_blk_elems[15] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 5));
	mat_a_blk_elems[16] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 6));
	mat_a_blk_elems[17] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0] + 7));

	//Broadcast A43 to A73 to registers
	mat_a_blk_elems[18] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 4));
	mat_a_blk_elems[19] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 5));
	mat_a_blk_elems[20] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 6));
	mat_a_blk_elems[21] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1] + 7));

	//Broadcast A54 to A74 to registers
	mat_a_blk_elems[22] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 5));
	mat_a_blk_elems[23] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 6));
	mat_a_blk_elems[24] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2] + 7));

	//Broadcast A65 to A75 to registers
	mat_a_blk_elems[25] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 6));
	mat_a_blk_elems[26] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3] + 7));

	//Broadcast A76 to register
	mat_a_blk_elems[27] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4] + 7));
#endif


	/*****************   first set of 8 rows of B processing starts    *****************/
	ptr_b_dup = ptr_b;
	i = 0;
	for (j = 0; j < numCols_b; j += 8)
	{
		/////////////////// Complete Upper 8x8 block trsm of B :- upper 8x8 block of B with upper 8x8 block of A
		//read 8x8 block of B into registers
		mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
		mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
		mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
		mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
		mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
		mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
		mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
		mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

		/* transpose steps start */
		////unpacklow////
		mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

		////unpackhigh////
		mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
		mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
		mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
		mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
		/* transpose steps end */

		mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
		mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
		mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
		mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
		mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
		mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
		mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
		mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);

		//(Row0)

		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[0]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[1]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[2]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[3]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[4]));
		mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l + cs_l_offset[5]));

		//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
		mat_b_col[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_col[1]);//d = c - (a*b)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[0]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[1]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[2]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[3]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[4]));
		mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l + 1 + cs_l_offset[5]));

		//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
		mat_b_col[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_col[2]);//d = c - (a*b)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[1]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[2]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[3]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[4]));
		mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l + 2 + cs_l_offset[5]));

		//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
		mat_b_col[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_col[3]);//d = c - (a*b)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[2]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[3]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[4]));
		mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l + 3 + cs_l_offset[5]));

		//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
		mat_b_col[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_col[4]);//d = c - (a*b)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[3]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[4]));
		mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l + 4 + cs_l_offset[5]));

		//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
		mat_b_col[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_col[5]);//d = c - (a*b)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[4]));
		mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l + 5 + cs_l_offset[5]));

		//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
		mat_b_col[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_col[6]);//d = c - (a*b)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_col[7]);//d = c - (a*b)



		mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l + 6 + cs_l_offset[5]));

		//(Row7): FMA operations of b7 with elements of index (7, 0)
		mat_b_col[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_col[7]);//d = c - (a*b)



		////////////////////////////////////////////////////////////////////////////////

		/* transpose steps start */
		////unpacklow////
		mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange low elements
#if REARRANGE_SHFL == 1
		mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
		mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
		mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
		mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
		mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
		mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
		mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
		mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
		//Merge rearranged low elements into complete rows
		mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
		mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
		mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
		mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

		////unpackhigh////
		mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
		mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
		mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
		mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

		//Rearrange high elements
#if REARRANGE_SHFL == 1
		mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
		mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
		mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
		mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
		mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
		mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
		mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
		mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

		//Merge rearranged high elements into complete rows
		mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
		mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
		mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
		mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
		/* transpose steps end */

		//Store the computed B columns
		_mm256_storeu_ps((float *)ptr_b_dup, mat_b_rearr[0]);
		_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)), mat_b_rearr[1]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0]), mat_b_rearr[2]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1]), mat_b_rearr[3]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2]), mat_b_rearr[4]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3]), mat_b_rearr[5]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4]), mat_b_rearr[6]);
		_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5]), mat_b_rearr[7]);

		i += cs_b_offset[6];
		ptr_b_dup += cs_b_offset[6];
		//i += 8;
		//ptr_b_dup += 8;
	}

	//c = 0;
	/***************** first set of 8 cols of B processing done *****************/
	ptr_b_dup = ptr_b;
	i1 = 0;
	//Start loop for cols of B to be processed in size of blk_width
	for (j = 8; j < numRows_lb; j += 8)//m :- 8x8 block row
	{
		ptr_l += cs_l_offset[6];


		//ptr_b += j;
		//ptr_b_dup += 8;
		ptr_b_dup += 8;
		i1 += 8;
		i = i1;
		i2 = 0;

		for (r = 0; r < numCols_b; r += GEMM_BLK_V1)
		{
#if GEMM_ACCUM_A
			//Read 8 cols of B columns of Block-to-be-solved
			mat_b_col[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_col[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_col[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_col[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_col[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_col[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_col[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_col[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
			////unpacklow////
			mat_b_rearr[0] = _mm256_unpacklo_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_rearr[1] = _mm256_unpacklo_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_rearr[2] = _mm256_unpacklo_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_rearr[3] = _mm256_unpacklo_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_rearr[0] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_rearr[4] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_rearr[1] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_rearr[5] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);

			////unpackhigh////
			mat_b_col[0] = _mm256_unpackhi_ps(mat_b_col[0], mat_b_col[1]);
			mat_b_col[1] = _mm256_unpackhi_ps(mat_b_col[2], mat_b_col[3]);
			mat_b_col[2] = _mm256_unpackhi_ps(mat_b_col[4], mat_b_col[5]);
			mat_b_col[3] = _mm256_unpackhi_ps(mat_b_col[6], mat_b_col[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_rearr[2] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_rearr[6] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_rearr[3] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_rearr[7] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);
			/* transpose steps end */
			
			mat_b_rearr[0] = _mm256_mul_ps(mat_b_rearr[0], alphaReg);
	        	mat_b_rearr[1] = _mm256_mul_ps(mat_b_rearr[1], alphaReg);
	    		mat_b_rearr[2] = _mm256_mul_ps(mat_b_rearr[2], alphaReg);
	    		mat_b_rearr[3] = _mm256_mul_ps(mat_b_rearr[3], alphaReg);
	    		mat_b_rearr[4] = _mm256_mul_ps(mat_b_rearr[4], alphaReg);
	    		mat_b_rearr[5] = _mm256_mul_ps(mat_b_rearr[5], alphaReg);
	    		mat_b_rearr[6] = _mm256_mul_ps(mat_b_rearr[6], alphaReg);
	    		mat_b_rearr[7] = _mm256_mul_ps(mat_b_rearr[7], alphaReg);
#endif
		
			//i = 0;
			ptr_l_dup = ptr_l;
			i4 = i2;
			for (l = 0; l < j; l += 8) // move across m
			{
				//for (k = 0; k < numCols_b; k += 8) // move across n for the same value of l (index of m)
				//{
					/////////////////// Partial Lower 8x8 block trsm of B
					//Read current 8 cols of B columns from specified 8x8 current-block of B
				mat_a_blk_elems[0] = _mm256_loadu_ps((float const *)ptr_b + i4);
				mat_a_blk_elems[1] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b));
				mat_a_blk_elems[2] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[0]));
				mat_a_blk_elems[3] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[1]));
				mat_a_blk_elems[4] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[2]));
				mat_a_blk_elems[5] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[3]));
				mat_a_blk_elems[6] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[4]));
				mat_a_blk_elems[7] = _mm256_loadu_ps((float const *)(ptr_b + i4 + cs_b_offset[5]));

				/* transpose steps start */
		////unpacklow////
				mat_b_col[0] = _mm256_unpacklo_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_b_col[1] = _mm256_unpacklo_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_b_col[2] = _mm256_unpacklo_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_b_col[3] = _mm256_unpacklo_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange low elements
#if REARRANGE_SHFL == 1
				mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
				mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
				mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
				mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
				mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
				mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
				mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
				mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
				//Merge rearranged low elements into complete rows
				mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
				mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
				mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
				mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

				////unpackhigh////
				mat_a_blk_elems[0] = _mm256_unpackhi_ps(mat_a_blk_elems[0], mat_a_blk_elems[1]);
				mat_a_blk_elems[1] = _mm256_unpackhi_ps(mat_a_blk_elems[2], mat_a_blk_elems[3]);
				mat_a_blk_elems[2] = _mm256_unpackhi_ps(mat_a_blk_elems[4], mat_a_blk_elems[5]);
				mat_a_blk_elems[3] = _mm256_unpackhi_ps(mat_a_blk_elems[6], mat_a_blk_elems[7]);

				//Rearrange high elements
#if REARRANGE_SHFL == 1
				mat_a_blk_elems[4] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x44);
				mat_a_blk_elems[5] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0xEE);
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x44);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0xEE);
#else
				mat_a_blk_elems[6] = _mm256_shuffle_ps(mat_a_blk_elems[0], mat_a_blk_elems[1], 0x4E);
				mat_a_blk_elems[7] = _mm256_shuffle_ps(mat_a_blk_elems[2], mat_a_blk_elems[3], 0x4E);
				mat_a_blk_elems[4] = _mm256_blend_ps(mat_a_blk_elems[0], mat_a_blk_elems[6], 0xCC);
				mat_a_blk_elems[5] = _mm256_blend_ps(mat_a_blk_elems[1], mat_a_blk_elems[6], 0x33);
				mat_a_blk_elems[6] = _mm256_blend_ps(mat_a_blk_elems[2], mat_a_blk_elems[7], 0xCC);
				mat_a_blk_elems[7] = _mm256_blend_ps(mat_a_blk_elems[3], mat_a_blk_elems[7], 0x33);
#endif

				//Merge rearranged high elements into complete rows
				mat_b_col[2] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x20);
				mat_b_col[6] = _mm256_permute2f128_ps(mat_a_blk_elems[4], mat_a_blk_elems[6], 0x31);
				mat_b_col[3] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x20);
				mat_b_col[7] = _mm256_permute2f128_ps(mat_a_blk_elems[5], mat_a_blk_elems[7], 0x31);
				/* transpose steps end */

						//Broadcast A8,0 to A15,0 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				//i4 = k >> 3;
				ptr_l_dup++;

#if GEMM_ACCUM_A
				//(Row8): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[0], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[0], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[0], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[0], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[0], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[0], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[0], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[0], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_mul_ps(mat_a_blk_elems[0], mat_b_col[0]);
				mat_b_rearr[1] = _mm256_mul_ps(mat_a_blk_elems[1], mat_b_col[0]);
				mat_b_rearr[2] = _mm256_mul_ps(mat_a_blk_elems[2], mat_b_col[0]);
				mat_b_rearr[3] = _mm256_mul_ps(mat_a_blk_elems[3], mat_b_col[0]);
				mat_b_rearr[4] = _mm256_mul_ps(mat_a_blk_elems[4], mat_b_col[0]);
				mat_b_rearr[5] = _mm256_mul_ps(mat_a_blk_elems[5], mat_b_col[0]);
				mat_b_rearr[6] = _mm256_mul_ps(mat_a_blk_elems[6], mat_b_col[0]);
				mat_b_rearr[7] = _mm256_mul_ps(mat_a_blk_elems[7], mat_b_col[0]);
#endif
				//Broadcast A21 to A71 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row9): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[1], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[1], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[1], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[1], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[1], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[1], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[1], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[1], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,2 to A15,2 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row10): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[2], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[2], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[2], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[2], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[2], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[2], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[2], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[2], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,3 to A15,3 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row11): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[3], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[3], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[3], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[3], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[3], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[3], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[3], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[3], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,4 to A15,4 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row12): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[4], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[4], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[4], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[4], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[4], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[4], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[4], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[4], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,5 to A15,5 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row13): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[5], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[5], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[5], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[5], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[5], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[5], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[5], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[5], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,6 to A15,6 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row14): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[6], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[6], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[6], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[6], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[6], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[6], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[6], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[6], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//Broadcast A8,7 to A15,7 to registers
				mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup));
				mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
				mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
				mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
				mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
				mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
				mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
				mat_a_blk_elems[7] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
				ptr_l_dup++;
#if GEMM_ACCUM_A
				//(Row15): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
				mat_b_rearr[0] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#else
				mat_b_rearr[0] = _mm256_fmadd_ps(mat_a_blk_elems[0], mat_b_col[7], mat_b_rearr[0]);//d = c - (a*b)
				mat_b_rearr[1] = _mm256_fmadd_ps(mat_a_blk_elems[1], mat_b_col[7], mat_b_rearr[1]);//d = c - (a*b)
				mat_b_rearr[2] = _mm256_fmadd_ps(mat_a_blk_elems[2], mat_b_col[7], mat_b_rearr[2]);//d = c - (a*b)
				mat_b_rearr[3] = _mm256_fmadd_ps(mat_a_blk_elems[3], mat_b_col[7], mat_b_rearr[3]);//d = c - (a*b)
				mat_b_rearr[4] = _mm256_fmadd_ps(mat_a_blk_elems[4], mat_b_col[7], mat_b_rearr[4]);//d = c - (a*b)
				mat_b_rearr[5] = _mm256_fmadd_ps(mat_a_blk_elems[5], mat_b_col[7], mat_b_rearr[5]);//d = c - (a*b)
				mat_b_rearr[6] = _mm256_fmadd_ps(mat_a_blk_elems[6], mat_b_col[7], mat_b_rearr[6]);//d = c - (a*b)
				mat_b_rearr[7] = _mm256_fmadd_ps(mat_a_blk_elems[7], mat_b_col[7], mat_b_rearr[7]);//d = c - (a*b)
#endif
					//end loop of cols					
				//}
				//i2 += cs_b_offset[6];
				i4 += 8;
			}
			//trsm solve

			k = 0;
			//for (i2 = 0; i2 < numCols_b; i2 += 8)
			//{
				//i2 = i1 + r;
				/////////////////// Complete Lower 8x8 block trsm of B :- lower 8x8 block of B with lower right 8x8 block of A
#if !GEMM_ACCUM_A
				//Read 8 cols of B columns of Block-to-be-solved
			mat_b_rearr[0] = _mm256_loadu_ps((float const *)ptr_b + i);
			mat_b_rearr[1] = _mm256_loadu_ps((float const *)(ptr_b + cs_b + i));
			mat_b_rearr[2] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[0] + i));
			mat_b_rearr[3] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[1] + i));
			mat_b_rearr[4] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[2] + i));
			mat_b_rearr[5] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[3] + i));
			mat_b_rearr[6] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[4] + i));
			mat_b_rearr[7] = _mm256_loadu_ps((float const *)(ptr_b + cs_b_offset[5] + i));

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */
			
			mat_b_col[0] = _mm256_mul_ps(mat_b_col[0], alphaReg);
			mat_b_col[1] = _mm256_mul_ps(mat_b_col[1], alphaReg);
			mat_b_col[2] = _mm256_mul_ps(mat_b_col[2], alphaReg);
			mat_b_col[3] = _mm256_mul_ps(mat_b_col[3], alphaReg);
			mat_b_col[4] = _mm256_mul_ps(mat_b_col[4], alphaReg);
			mat_b_col[5] = _mm256_mul_ps(mat_b_col[5], alphaReg);
			mat_b_col[6] = _mm256_mul_ps(mat_b_col[6], alphaReg);
			mat_b_col[7] = _mm256_mul_ps(mat_b_col[7], alphaReg);
#endif
				//Broadcast A10 to A70 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[0]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[1]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[2]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[3]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[4]));
			mat_a_blk_elems[6] = _mm256_broadcast_ss((float const *)(ptr_l_dup + cs_l_offset[5]));
			//i += cs_l;

#if GEMM_ACCUM_A
			//(Row0): already done

#else
				mat_b_rearr[0] = _mm256_sub_ps(mat_b_col[0], mat_b_rearr[0]);
#endif

#if GEMM_ACCUM_A
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#else
			mat_b_rearr[1] = _mm256_sub_ps(mat_b_col[1], mat_b_rearr[1]);
			mat_b_rearr[2] = _mm256_sub_ps(mat_b_col[2], mat_b_rearr[2]);
			mat_b_rearr[3] = _mm256_sub_ps(mat_b_col[3], mat_b_rearr[3]);
			mat_b_rearr[4] = _mm256_sub_ps(mat_b_col[4], mat_b_rearr[4]);
			mat_b_rearr[5] = _mm256_sub_ps(mat_b_col[5], mat_b_rearr[5]);
			mat_b_rearr[6] = _mm256_sub_ps(mat_b_col[6], mat_b_rearr[6]);
			mat_b_rearr[7] = _mm256_sub_ps(mat_b_col[7], mat_b_rearr[7]);

			//(Row1): FMA operations of b1 with elements of indices from (1, 0) uptill (7, 0)
			mat_b_rearr[1] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[0], mat_b_rearr[1]);//d = c - (a*b)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[0], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[0], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[0], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[0], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[0], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[6], mat_b_rearr[0], mat_b_rearr[7]);//d = c - (a*b)
#endif
				//Broadcast A21 to A71 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[0]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[1]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[2]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[3]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[4]));
			mat_a_blk_elems[5] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 1 + cs_l_offset[5]));
			//i += cs_l;



			//(Row2): FMA operations of b2 with elements of indices from (2, 0) uptill (7, 0)
			mat_b_rearr[2] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[1], mat_b_rearr[2]);//d = c - (a*b)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[1], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[1], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[1], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[1], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[5], mat_b_rearr[1], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A32 to A72 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[1]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[2]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[3]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[4]));
			mat_a_blk_elems[4] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 2 + cs_l_offset[5]));
			//i += cs_l;



			//(Row3): FMA operations of b3 with elements of indices from (3, 0) uptill (7, 0)
			mat_b_rearr[3] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[2], mat_b_rearr[3]);//d = c - (a*b)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[2], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[2], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[2], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[4], mat_b_rearr[2], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A43 to A73 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[2]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[3]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[4]));
			mat_a_blk_elems[3] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 3 + cs_l_offset[5]));
			//i += cs_l;



			//(Row4): FMA operations of b4 with elements of indices from (4, 0) uptill (7, 0)
			mat_b_rearr[4] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[3], mat_b_rearr[4]);//d = c - (a*b)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[3], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[3], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[3], mat_b_rearr[3], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A54 to A74 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[3]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[4]));
			mat_a_blk_elems[2] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 4 + cs_l_offset[5]));
			//i += cs_l;



			//(Row5): FMA operations of b5 with elements of indices from (5, 0) uptill (7, 0)
			mat_b_rearr[5] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[4], mat_b_rearr[5]);//d = c - (a*b)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[4], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[2], mat_b_rearr[4], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A65 to A75 to registers
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[4]));
			mat_a_blk_elems[1] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 5 + cs_l_offset[5]));
			//i += cs_l;



			//(Row6): FMA operations of b6 with elements of indices from (6, 0) uptill (7, 0)
			mat_b_rearr[6] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[5], mat_b_rearr[6]);//d = c - (a*b)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[1], mat_b_rearr[5], mat_b_rearr[7]);//d = c - (a*b)

			//Broadcast A76 to register
			mat_a_blk_elems[0] = _mm256_broadcast_ss((float const *)(ptr_l_dup + 6 + cs_l_offset[5]));



			//(Row7): FMA operations of b7 with elements of index (7, 0)
			mat_b_rearr[7] = _mm256_fnmadd_ps(mat_a_blk_elems[0], mat_b_rearr[6], mat_b_rearr[7]);//d = c - (a*b)



			////////////////////////////////////////////////////////////////////////////////

			/* transpose steps start */
	////unpacklow////
			mat_b_col[0] = _mm256_unpacklo_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_col[1] = _mm256_unpacklo_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_col[2] = _mm256_unpacklo_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_col[3] = _mm256_unpacklo_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange low elements
#if REARRANGE_SHFL == 1
			mat_b_col[4] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x44);
			mat_b_col[5] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0xEE);
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x44);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0xEE);
#else
			mat_b_col[6] = _mm256_shuffle_ps(mat_b_col[0], mat_b_col[1], 0x4E);
			mat_b_col[7] = _mm256_shuffle_ps(mat_b_col[2], mat_b_col[3], 0x4E);
			mat_b_col[4] = _mm256_blend_ps(mat_b_col[0], mat_b_col[6], 0xCC);
			mat_b_col[5] = _mm256_blend_ps(mat_b_col[1], mat_b_col[6], 0x33);
			mat_b_col[6] = _mm256_blend_ps(mat_b_col[2], mat_b_col[7], 0xCC);
			mat_b_col[7] = _mm256_blend_ps(mat_b_col[3], mat_b_col[7], 0x33);
#endif
			//Merge rearranged low elements into complete rows
			mat_b_col[0] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x20);
			mat_b_col[4] = _mm256_permute2f128_ps(mat_b_col[4], mat_b_col[6], 0x31);
			mat_b_col[1] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x20);
			mat_b_col[5] = _mm256_permute2f128_ps(mat_b_col[5], mat_b_col[7], 0x31);

			////unpackhigh////
			mat_b_rearr[0] = _mm256_unpackhi_ps(mat_b_rearr[0], mat_b_rearr[1]);
			mat_b_rearr[1] = _mm256_unpackhi_ps(mat_b_rearr[2], mat_b_rearr[3]);
			mat_b_rearr[2] = _mm256_unpackhi_ps(mat_b_rearr[4], mat_b_rearr[5]);
			mat_b_rearr[3] = _mm256_unpackhi_ps(mat_b_rearr[6], mat_b_rearr[7]);

			//Rearrange high elements
#if REARRANGE_SHFL == 1
			mat_b_rearr[4] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x44);
			mat_b_rearr[5] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0xEE);
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x44);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0xEE);
#else
			mat_b_rearr[6] = _mm256_shuffle_ps(mat_b_rearr[0], mat_b_rearr[1], 0x4E);
			mat_b_rearr[7] = _mm256_shuffle_ps(mat_b_rearr[2], mat_b_rearr[3], 0x4E);
			mat_b_rearr[4] = _mm256_blend_ps(mat_b_rearr[0], mat_b_rearr[6], 0xCC);
			mat_b_rearr[5] = _mm256_blend_ps(mat_b_rearr[1], mat_b_rearr[6], 0x33);
			mat_b_rearr[6] = _mm256_blend_ps(mat_b_rearr[2], mat_b_rearr[7], 0xCC);
			mat_b_rearr[7] = _mm256_blend_ps(mat_b_rearr[3], mat_b_rearr[7], 0x33);
#endif

			//Merge rearranged high elements into complete rows
			mat_b_col[2] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x20);
			mat_b_col[6] = _mm256_permute2f128_ps(mat_b_rearr[4], mat_b_rearr[6], 0x31);
			mat_b_col[3] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x20);
			mat_b_col[7] = _mm256_permute2f128_ps(mat_b_rearr[5], mat_b_rearr[7], 0x31);
			/* transpose steps end */

					//Store the computed B columns
			_mm256_storeu_ps((float *)ptr_b_dup + i2, mat_b_col[0]);
			_mm256_storeu_ps((float *)(ptr_b_dup + (cs_b)+i2), mat_b_col[1]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[0] + i2), mat_b_col[2]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[1] + i2), mat_b_col[3]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[2] + i2), mat_b_col[4]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[3] + i2), mat_b_col[5]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[4] + i2), mat_b_col[6]);
			_mm256_storeu_ps((float *)(ptr_b_dup + cs_b_offset[5] + i2), mat_b_col[7]);
			//printf("writing B => m[%d], n[%d], [%f]\n", j, k, *(ptr_b_dup + k));
			k++;
			//}
			i += cs_b_offset[6];
			i2 += cs_b_offset[6];
		}
	} //numRows of A
	///////////////////loop ends /////////////////////
}
#endif
