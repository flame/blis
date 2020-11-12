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
 //printf("omatcopy_ test.....start\n");

 obj_t data_aptr;
 obj_t data_bptr;
 obj_t alpha;
 dim_t p;
 dim_t p_begin, p_end, p_inc;
 p_begin = 200;
 p_end   = 2000;
 p_inc   = 200;

 trans_t  trans;
 f77_char f77_trans;
 
 num_t dt;
 int rows;
 int cols;

 inc_t cs_data;

 double dtime;
 double dtime_save;
 double gflops;

 dt = BLIS_DOUBLE;
 trans = BLIS_NO_TRANSPOSE;
 bli_param_map_blis_to_netlib_trans( trans, &f77_trans);

 dtime_save = DBL_MAX;
 for ( p = p_begin; p <= p_end; p += p_inc )
 {

  rows = p;
  cols = p; 
  cs_data = rows;

  bli_obj_create( dt, 1, 1, 0, 0, &alpha);
  bli_obj_create( dt, rows, cols, 1, cs_data, &data_aptr );
  bli_obj_create( dt, rows, cols, 1, cs_data, &data_bptr );

  bli_randm( &data_aptr );
  bli_setsc(  (2.0/1.0), 0.0, &alpha );

  dtime = bli_clock();

  if ( bli_is_float( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);

   float*   alpha_p = bli_obj_buffer( &alpha );
   float*   a_p     = bli_obj_buffer( &data_aptr);
   float*   b_p     = bli_obj_buffer( &data_bptr);

   somatcopy_ (&f77_trans, &rows_p, &cols_p, alpha_p, a_p, &lda, b_p, &ldb);
  }
  else if ( bli_is_double( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);

   double*   alpha_p = bli_obj_buffer( &alpha );
   double*   a_p     = bli_obj_buffer( &data_aptr);
   double*   b_p     = bli_obj_buffer( &data_bptr);

   domatcopy_ (&f77_trans, &rows_p, &cols_p, alpha_p, a_p, &lda, b_p, &ldb);
  }
  else if ( bli_is_scomplex( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);

   scomplex*   alpha_p = bli_obj_buffer( &alpha );
   scomplex*   a_p     = bli_obj_buffer( &data_aptr);
   scomplex*   b_p     = bli_obj_buffer( &data_bptr);

   comatcopy_ (&f77_trans, &rows_p, &cols_p, alpha_p, a_p, &lda, b_p, &ldb);
  }
  else if ( bli_is_dcomplex( dt ) )
  {
   f77_int  rows_p     = bli_obj_length( &data_aptr);
   f77_int  cols_p     = bli_obj_width( &data_aptr);

   f77_int  lda    = bli_obj_col_stride( &data_aptr);
   f77_int  ldb    = bli_obj_col_stride( &data_bptr);

   dcomplex*   alpha_p = bli_obj_buffer( &alpha );
   dcomplex*   a_p     = bli_obj_buffer( &data_aptr);
   dcomplex*   b_p     = bli_obj_buffer( &data_bptr);

   zomatcopy_ (&f77_trans, &rows_p, &cols_p, alpha_p, a_p, &lda, b_p, &ldb);
  }

  dtime_save = bli_clock_min_diff( dtime_save, dtime );
  gflops = ( 2.0 * rows * cols ) / ( dtime_save * 1.0e9 );

  if ( bli_is_complex( dt ) ) gflops *= 4.0;

  printf( "( %2lu, 1:4 ) = [ %4lu %4lu %7.2f ];\n",
                        ( unsigned long )(p - p_begin)/p_inc + 1,
                        ( unsigned long )rows,
                        ( unsigned long )cols, gflops );
  bli_obj_free( &alpha );
  bli_obj_free( &data_aptr );
  bli_obj_free( &data_bptr );
 }
 //printf("omatcopy_ test.....end\n");
 return (0);
}
