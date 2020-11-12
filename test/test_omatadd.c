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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

int main( int argc, char** argv )
{
 //printf("omatadd_ test.....start\n");

 obj_t data_aptr;
 obj_t data_bptr;
 obj_t data_cptr;
 obj_t alpha;
 obj_t beta;
 dim_t p;
 dim_t p_begin, p_end, p_inc;
 p_begin = 200;
 p_end   = 2000;
 p_inc   = 200;

 trans_t  transa,transb;
 f77_char f77_transa,f77_transb;
 
 num_t dt;
 int rows;
 int cols;

 inc_t cs_data;

 double dtime;
 double dtime_save;
 double gflops;

 dt = BLIS_DOUBLE;
 transa = BLIS_NO_TRANSPOSE;
 transb = BLIS_NO_TRANSPOSE;
 bli_param_map_blis_to_netlib_trans( transa, &f77_transa);
 bli_param_map_blis_to_netlib_trans( transb, &f77_transb);

 dtime_save = DBL_MAX;
 for ( p = p_begin; p <= p_end; p += p_inc )
 {

  rows = p;
  cols = p; 
  cs_data = rows;

  bli_obj_create( dt, 1, 1, 0, 0, &alpha);
  bli_obj_create( dt, 1, 1, 0, 0, &beta);
  bli_obj_create( dt, rows, cols, 1, cs_data, &data_aptr );
  bli_obj_create( dt, rows, cols, 1, cs_data, &data_bptr );
  bli_obj_create( dt, rows, cols, 1, cs_data, &data_cptr );

  bli_randm( &data_aptr );
  bli_randm( &data_bptr );
  bli_setsc(  (2.0/1.0), 0.0, &alpha );
  bli_setsc(  (2.0/1.0), 0.0, &beta );

  dtime = bli_clock();

  if ( bli_is_float( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);
   f77_int  ldc    = bli_obj_col_stride( &data_cptr);

   float*   alpha_p = bli_obj_buffer( &alpha );
   float*   beta_p = bli_obj_buffer( &beta );
   float*   a_p     = bli_obj_buffer( &data_aptr);
   float*   b_p     = bli_obj_buffer( &data_bptr);
   float*   c_p     = bli_obj_buffer( &data_cptr);

   somatadd_ (&f77_transa,&f77_transb, &rows_p, &cols_p, alpha_p, a_p, &lda, beta_p, b_p, &ldb, c_p, &ldc);
  }
  else if ( bli_is_double( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);
   f77_int  ldc    = bli_obj_col_stride( &data_cptr);

   double*   alpha_p = bli_obj_buffer( &alpha );
   double*   beta_p = bli_obj_buffer( &beta );
   double*   a_p     = bli_obj_buffer( &data_aptr);
   double*   b_p     = bli_obj_buffer( &data_bptr);
   double*   c_p     = bli_obj_buffer( &data_cptr);

   domatadd_ (&f77_transa,&f77_transb, &rows_p, &cols_p, alpha_p, a_p, &lda, beta_p, b_p, &ldb, c_p, &ldc);
  }
  else if ( bli_is_scomplex( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);
   f77_int  ldc    = bli_obj_col_stride( &data_cptr);

   scomplex*   alpha_p = bli_obj_buffer( &alpha );
   scomplex*   beta_p = bli_obj_buffer( &beta );
   scomplex*   a_p     = bli_obj_buffer( &data_aptr);
   scomplex*   b_p     = bli_obj_buffer( &data_bptr);
   scomplex*   c_p     = bli_obj_buffer( &data_cptr);

   comatadd_ (&f77_transa,&f77_transb, &rows_p, &cols_p, alpha_p, a_p, &lda, beta_p, b_p, &ldb, c_p, &ldc);
  }
  else if ( bli_is_dcomplex( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);
   f77_int  ldc    = bli_obj_col_stride( &data_cptr);

   dcomplex*   alpha_p = bli_obj_buffer( &alpha );
   dcomplex*   beta_p = bli_obj_buffer( &beta );
   dcomplex*   a_p     = bli_obj_buffer( &data_aptr);
   dcomplex*   b_p     = bli_obj_buffer( &data_bptr);
   dcomplex*   c_p     = bli_obj_buffer( &data_cptr);

   zomatadd_ (&f77_transa,&f77_transb, &rows_p, &cols_p, alpha_p, a_p, &lda, beta_p, b_p, &ldb, c_p, &ldc);
  }

  dtime_save = bli_clock_min_diff( dtime_save, dtime );
  gflops = ( 2.0 * rows * cols ) / ( dtime_save * 1.0e9 );

  if ( bli_is_complex( dt ) ) gflops *= 4.0;

  printf( "( %2lu, 1:4 ) = [ %4lu %4lu %7.2f ];\n",
                        ( unsigned long )(p - p_begin)/p_inc + 1,
                        ( unsigned long )rows,
                        ( unsigned long )cols, gflops );
  bli_obj_free( &alpha );
  bli_obj_free( &beta );
  bli_obj_free( &data_aptr );
  bli_obj_free( &data_bptr );
  bli_obj_free( &data_cptr );
 }
 //printf("omatadd_ test.....end\n");
 return (0);
}
