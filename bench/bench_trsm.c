
/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.
   Redistribution and use in source and binary forms, with or without
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
/*
Benchmark application to process aocl logs generated
by BLIS library for trsm.
*/
#ifndef N_REPEAT
#define N_REPEAT 30
#endif
#define AOCL_MATRIX_INITIALISATION

/* For BLIS since logs are collected at BLAS interfaces
 * we disable cblas interfaces for this benchmark application
 */

/*
#ifdef BLIS_ENABLE_CBLAS
#define CBLAS
#endif
*/

int main( int argc, char** argv )
{
    obj_t a, b;
    obj_t b_save;
    obj_t alpha;
    dim_t m, n;
    dim_t  p_inc = 0; // to keep track of number of inputs
    num_t dt = BLIS_DOUBLE;
    dim_t   r, n_repeats;
    f77_char side;
    uplo_t uploa;
    trans_t transa;
    diag_t diaga;
    f77_char f77_side;
    f77_char f77_uploa;
    f77_char f77_transa;
    f77_char f77_diaga;
    double dtime;
    double dtime_save;
    double gflops;
    double alphaR;
    double alphaI;
    FILE* fin = NULL;
    FILE* fout = NULL;
    n_repeats = N_REPEAT;
    if(argc < 3)
    {
        printf("Usage: ./test_trsm_XX.x input.csv output.csv\n");
        exit(1);
    }
    fin = fopen(argv[1], "r");
    if(fin == NULL)
    {
        printf("Error opening the file %s\n", argv[1]);
        exit(1);
    }
    fout = fopen(argv[2], "w");
    if(fout == NULL)
    {
        printf("Error opening the file %s\n", argv[2]);
        exit(1);
    }
    fprintf(fout,"dt\t side\t uploa\t transa\t diaga\t m\t n\t lda\t ldb\t alphaR\t alphaI\t gflops\n");

    dim_t lda,ldb;
    f77_char dt_type_arg, side_arg, uploa_arg, transa_arg, diaga_arg;
    f77_char logline[255];
    // input order: {S,D,C,Z} {side, uplo, transa, diag, m, n, lda, ldb, alphaR, alphaI}
    while(fscanf(fin, "%s %c %c %c %c %c %ld %ld %ld %ld %lf %lf\n",
          logline, &dt_type_arg, &side_arg, &uploa_arg, &transa_arg, &diaga_arg, &m, &n, &lda, &ldb,
           &alphaR, &alphaI) == 12)
    {
        if( (dt_type_arg=='S') || (dt_type_arg=='s') )    dt = BLIS_FLOAT;
        if( (dt_type_arg=='D') || (dt_type_arg=='d') )    dt = BLIS_DOUBLE;
        if( (dt_type_arg=='C') || (dt_type_arg=='c') )    dt = BLIS_SCOMPLEX;
        if( (dt_type_arg=='Z') || (dt_type_arg=='z') )    dt = BLIS_DCOMPLEX;
        if( 'l' == side_arg|| 'L' == side_arg )
            side = BLIS_LEFT;
        else if( 'r' == side_arg || 'R' == side_arg )
            side = BLIS_RIGHT;
        else
        {
            printf("Invalid entry for the argument 'side':%c\n", side_arg);
            continue;
        }

        if('l' == uploa_arg || 'L' == uploa_arg)
            uploa = BLIS_LOWER;
        else if('u' == uploa_arg || 'U' == uploa_arg)
            uploa = BLIS_UPPER;
        else
        {
            printf("Invalid entry for the argument 'uplo':%c\n",uploa_arg);
            continue;
        }

        if('t' == transa_arg || 'T' == transa_arg)
            transa = BLIS_TRANSPOSE;
        else if('n' == transa_arg || 'N' == transa_arg)
            transa = BLIS_NO_TRANSPOSE;
	    else if('c' == transa_arg || 'C' == transa_arg)
	        transa = BLIS_CONJ_TRANSPOSE;
	    else
        {
            printf("Invalid entry for the argument 'transa':%c\n",transa_arg);
            continue;
        }

        if('u' == diaga_arg || 'U' == diaga_arg)
            diaga = BLIS_UNIT_DIAG;
        else if('n' == diaga_arg || 'N' == diaga_arg)
            diaga = BLIS_NONUNIT_DIAG;
        else
        {
            printf("Invalid entry for the argument 'diaga':%c\n", diaga_arg);
            continue;
        }

        bli_param_map_blis_to_netlib_side( side, &f77_side );
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
        bli_param_map_blis_to_netlib_diag( diaga, &f77_diaga );

        if ( bli_is_left( side ) )
            bli_obj_create( dt, m, m, 1, lda, &a );
        else
            bli_obj_create( dt, n, n, 1, lda, &a );

        bli_obj_create( dt, m, n, 1, ldb, &b );
        bli_obj_create( dt, m, n, 1, ldb, &b_save );

#ifdef AOCL_MATRIX_INITIALISATION
        bli_randm( &a );
        bli_randm( &b );
#endif
        bli_obj_set_struc( BLIS_TRIANGULAR, &a );
        bli_obj_set_uplo( uploa, &a );
        bli_obj_set_conjtrans( transa, &a );
        bli_obj_set_diag( diaga, &a );
        // Randomize A and zero the unstored triangle to ensure the
        // implementation reads only from the stored region.
        bli_randm( &a );
        bli_mktrim( &a );
        // Load the diagonal of A to make it more likely to be invertible.
        bli_shiftd( &BLIS_TWO, &a );
        bli_obj_create( dt, 1, 1, 0, 0, &alpha );
        bli_setsc(  alphaR, alphaI, &alpha );
        bli_copym( &b, &b_save );
        dtime_save = DBL_MAX;
        for ( r = 0; r < n_repeats; ++r )
        {
            bli_copym( &b_save, &b );
#ifdef PRINT
            bli_printm( "a", &a, "%4.1f", "" );
            bli_printm( "b", &b, "%4.1f", "" );
#endif
            dtime = bli_clock();
#ifdef BLIS
            bli_trsm( &side,
                      &alpha,
                      &a,
                      &b );
#else
#ifdef CBLAS
            enum CBLAS_ORDER     cblas_order;
            enum CBLAS_TRANSPOSE cblas_transa;
            enum CBLAS_UPLO cblas_uplo;
            enum CBLAS_SIDE cblas_side;
            enum CBLAS_DIAG cblas_diag;

            if ( bli_obj_row_stride( &b ) == 1 )
              cblas_order = CblasColMajor;
            else
              cblas_order = CblasRowMajor;

            if( bli_is_trans( transa ) )
              cblas_transa = CblasTrans;
            else if( bli_is_conjtrans( transa ) )
              cblas_transa = CblasConjTrans;
            else
              cblas_transa = CblasNoTrans;

            if ('u' == diaga_arg || 'U' == diaga_arg)
              cblas_diag = CblasUnit;
            else
              cblas_diag = CblasNonUnit;

            if( 'l' == side_arg || 'L' == side_arg )
                cblas_side = CblasLeft;
            else if( 'r' == side_arg || 'R' == side_arg )
                cblas_side = CblasRight;
            else
            {
                printf("Invalid entry for the argument 'side':%c\n", side_arg);
                continue;
            }

            if('l' == uploa_arg || 'L' == uploa_arg)
                cblas_uplo = CblasLower;
            else if('u' == uploa_arg || 'U' == uploa_arg)
                cblas_uplo = CblasUpper;
            else
            {
                printf("Invalid entry for the argument 'uplo':%c\n",uploa_arg);
                continue;
            }

#else
            f77_char f77_transa;
            bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
#endif
            if ( bli_is_float( dt ) )
            {
                f77_int  mm     = bli_obj_length( &b );
                f77_int  nn     = bli_obj_width( &b );
                f77_int  lda    = bli_obj_col_stride( &a );
                f77_int  ldb    = bli_obj_col_stride( &b );
                float*   alphap = bli_obj_buffer( &alpha );
                float*   ap     = bli_obj_buffer( &a );
                float*   bp     = bli_obj_buffer( &b );
#ifdef CBLAS
                cblas_strsm( cblas_order,
                             cblas_side,
                             cblas_uplo,
                             cblas_transa,
                             cblas_diag,
                             mm,
                             nn,
                             *alphap,
                             ap, lda,
                             bp, ldb
                             );
#else
                strsm_( &f77_side,
                        &f77_uploa,
                        &f77_transa,
                        &f77_diaga,
                        &mm,
                        &nn,
                        alphap,
                        ap, &lda,
                        bp, &ldb );
#endif
            }
            else if ( bli_is_double( dt ) )
            {
                f77_int  mm     = bli_obj_length( &b );
                f77_int  nn     = bli_obj_width( &b );
                f77_int  lda    = bli_obj_col_stride( &a );
                f77_int  ldb    = bli_obj_col_stride( &b );
                double*  alphap = bli_obj_buffer( &alpha );
                double*  ap     = bli_obj_buffer( &a );
                double*  bp     = bli_obj_buffer( &b );
#ifdef CBLAS
                cblas_dtrsm( cblas_order,
                             cblas_side,
                             cblas_uplo,
                             cblas_transa,
                             cblas_diag,
                             mm,
                             nn,
                             *alphap,
                             ap, lda,
                             bp, ldb
                             );
#else
                dtrsm_( &f77_side,
                        &f77_uploa,
                        &f77_transa,
                        &f77_diaga,
                        &mm,
                        &nn,
                        alphap,
                        ap, &lda,
                        bp, &ldb );
#endif
            }
            else if ( bli_is_scomplex( dt ) )
            {
                f77_int  mm     = bli_obj_length( &b );
                f77_int  nn     = bli_obj_width( &b );
                f77_int  lda    = bli_obj_col_stride( &a );
                f77_int  ldb    = bli_obj_col_stride( &b );
                scomplex*  alphap = bli_obj_buffer( &alpha );
                scomplex*  ap     = bli_obj_buffer( &a );
                scomplex*  bp     = bli_obj_buffer( &b );
#ifdef CBLAS
                cblas_ctrsm( cblas_order,
                             cblas_side,
                             cblas_uplo,
                             cblas_transa,
                             cblas_diag,
                             mm,
                             nn,
                             alphap,
                             ap, lda,
                             bp, ldb
                             );
#else
                ctrsm_( &f77_side,
                        &f77_uploa,
                        &f77_transa,
                        &f77_diaga,
                        &mm,
                        &nn,
                        alphap,
                        ap, &lda,
                        bp, &ldb );
#endif
            }
            else if ( bli_is_dcomplex( dt ) )
            {
                f77_int  mm     = bli_obj_length( &b );
                f77_int  nn     = bli_obj_width( &b );
                f77_int  lda    = bli_obj_col_stride( &a );
                f77_int  ldb    = bli_obj_col_stride( &b );
                dcomplex*  alphap = bli_obj_buffer( &alpha );
                dcomplex*  ap     = bli_obj_buffer( &a );
                dcomplex*  bp     = bli_obj_buffer( &b );
#ifdef CBLAS
                cblas_ztrsm( cblas_order,
                             cblas_side,
                             cblas_uplo,
                             cblas_transa,
                             cblas_diag,
                             mm,
                             nn,
                             alphap,
                             ap, lda,
                             bp, ldb
                             );
#else
                ztrsm_( &f77_side,
                        &f77_uploa,
                        &f77_transa,
                        &f77_diaga,
                        &mm,
                        &nn,
                        alphap,
                        ap, &lda,
                        bp, &ldb );
#endif
            }else{
                printf("Invalid data type! Exiting!\n");
                exit(1);
            }
#endif
            dtime_save = bli_clock_min_diff( dtime_save, dtime );
        }
        if ( bli_is_left( side ) )
            gflops = ( 1.0 * m * m * n ) / ( dtime_save * 1.0e9 );
        else
            gflops = ( 1.0 * m * n * n ) / ( dtime_save * 1.0e9 );
        if ( bli_is_complex( dt ) ) gflops *= 4.0;
#ifdef BLIS
        printf( "data_trsm_blis\t\t");
#else
        printf( "data_trsm_%s\t\t",BLAS );
#endif
        p_inc++;
        printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
                ( unsigned long )p_inc,
                ( unsigned long )m, gflops );
        fprintf(fout,"%c\t %c\t %c\t %c\t %c\t %4lu\t %4lu\t %4lu\t %4lu\t %6.3f\t %6.3f\t %6.3f\n",
               dt_type_arg, side_arg, uploa_arg, transa_arg,
               diaga_arg, (unsigned long )m, (unsigned long ) n, (unsigned long )lda,
               (unsigned long )ldb, alphaR, alphaI, gflops);
        fflush(fout);
        bli_obj_free( &alpha );
        bli_obj_free( &a );
        bli_obj_free( &b );
        bli_obj_free( &b_save );
    }
    fclose(fin);
    fclose(fout);
    //bli_finalize();
    return 0;
}
