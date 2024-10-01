/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "bench_lpgemm_helpers.h"

#define GEN_UNPACK_ACC_CHK_DRV_FUNC( B_type, LP_SFX ) \
void unpack_accuracy_check_driver_ ## LP_SFX \
    ( \
        FILE*   fout, \
        const char stor_order, \
        const char mat_type, \
        dim_t   m, \
        dim_t   n, \
        dim_t   rs, \
        dim_t   cs, \
        B_type* b, \
        B_type* b_unpacked \
    ) \
{ \
    for( dim_t i = 0; i < m; i++ ) \
    { \
        for( dim_t j = 0; j < n; j++ ) \
        { \
            if ( b[ i * ( rs ) + j * cs ] != b_unpacked[ i * ( rs ) + j * cs ] ) \
            { \
                if( fout ) \
                { \
                    fprintf( fout, "failure, m: %ld, n: %ld, computed:%f, ref:%f, diff:%f\n", \
                        i, j, ( float )b[ i * ( rs ) + j * cs ], ( float )b_unpacked[ i * ( rs ) + j * cs ], \
                        ( float )( b[ i * ( rs ) + j * cs ] - b_unpacked[ i * ( rs ) + j * cs ] ) ); \
                } \
                printf( "failure, m: %ld, n: %ld, computed:%f, ref:%f, diff:%f\n", \
                        i, j, ( float )b[ i * ( rs ) + j * cs ], ( float )b_unpacked[ i * ( rs ) + j * cs ], \
                        ( float )( b[ i * ( rs ) + j * cs ] - b_unpacked[ i * ( rs ) + j * cs ] ) ); \
                goto cleanup_acc; \
            } \
        } \
    } \
 \
cleanup_acc: \
    fflush(stdout); \
    return; \
} \

GEN_UNPACK_ACC_CHK_DRV_FUNC( bfloat16, bf16bf16f32of32 )

#define GEN_UNPACK_BENCH_MAIN_FUNC( B_type, LP_SFX ) \
void unpack_bench_main_ ## LP_SFX \
    ( \
        FILE*   fout, \
        char    stor_order, \
        char    mat_type, \
        dim_t   m, \
        dim_t   n, \
        dim_t   stride \
    ) \
{ \
    dim_t size_B = 0; \
 \
    dim_t rs, cs; \
    if ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ) \
    { \
        size_B = m * stride; \
        rs = stride; \
        cs = 1; \
    } \
    else \
    { \
        size_B = stride * n; \
        rs = 1; \
        cs = stride; \
    } \
 \
    /* Original B matrix */ \
    B_type* b = ( B_type* ) lpgemm_malloc( sizeof( B_type ) * size_B ); \
    GEN_FUNC_NAME(fill_array_,B_type)(b, size_B ); \
 \
    /* Matrix to be unpacked into */ \
    B_type* b_unpacked = ( B_type* ) lpgemm_malloc( sizeof( B_type ) * size_B ); \
    memset( ( void* ) b_unpacked, 0, sizeof( B_type ) * size_B ); \
 \
    /* Reorder matrix */ \
    B_type* b_reorder = NULL; \
 \
    siz_t b_reorder_buf_siz_req = \
        GEN_FUNC_NAME(aocl_get_reorder_buf_size_,LP_SFX)( stor_order, 'n', mat_type, m, n ); \
 \
    b_reorder = ( B_type* ) lpgemm_malloc( b_reorder_buf_siz_req ); \
    /* reorder B. */ \
    GEN_FUNC_NAME(aocl_reorder_,LP_SFX)( stor_order, 'n', mat_type, \
                                         b, b_reorder, m, n, stride ); \
 \
    /* Unpack B. */ \
    GEN_FUNC_NAME(aocl_unreorder_,LP_SFX)( stor_order, mat_type, \
                                           b_reorder, b_unpacked, m, n, stride ); \
 \
    /* Accuracy check */ \
    printf("Running accuracy check\n"); \
    printf("%s m: %ld, n: %ld, stride: %ld, stor_order: %c\n", \
            XSTR(LP_SFX), m, n, stride, stor_order ); \
    GEN_FUNC_NAME( unpack_accuracy_check_driver_, LP_SFX ) \
    ( \
        fout, stor_order, mat_type, m, n, rs, cs,\
        b, b_unpacked \
    ); \
 \
    lpgemm_free( b ); \
    lpgemm_free( b_unpacked ); \
    lpgemm_free( b_reorder ); \
} \

GEN_UNPACK_BENCH_MAIN_FUNC( bfloat16, bf16bf16f32of32 )

int main( int argc, char** argv )
{
    // By default bench mode is set to accuracy.
    bench_mode = 'a';

    FILE* fin  = NULL;
    if ( argc < 3 )
    {
        printf
        (
          "Usage: ./bench_lpgemm_unpack -i input.txt \n" \
        );
        exit( 1 );
    }

    char* file_name = NULL;

    // Parse CLI arguments.
     getopt_t state;
     // Initialize the state for running bli_getopt(). Here, 0 is the
     // initial value for opterr, which suppresses error messages.
     bli_getopt_init_state( 0, &state );

#define UNPACK_OPS_TYPE_STR_LEN 24
    char unpack_ops_type_str[UNPACK_OPS_TYPE_STR_LEN];

     int opt;
     // Process all option arguments until we get a -1, which means we're done.
     while( (opt = bli_getopt( argc, argv, "i:", &state )) != -1 )
    {
        char opt_ch = ( char )opt;
        switch( opt_ch )
        {
            case 'i':
                    file_name = state.optarg;
                    break;
            default:
                    break;
        }
    }

    if ( file_name == NULL )
    {
        printf( " File name provided is invalid.\n" );
        exit( 1 );
    }

    fin = fopen( file_name, "r" );
    if (fin == NULL)
    {
        printf( "Error opening the file %s\n", argv[1] );
        exit( 1 );
    }

    FILE* fout = NULL;

    fout = fopen( "lpgemm_unpack_accuracy_test_failures.txt", "w" );

    char stor_order;
    char mat_type;
    dim_t m, n;
    dim_t stride;

    const dim_t len_list_omp_cores_for_testing = 2;
    const dim_t list_omp_cores_for_testing[2] = { 1, 64 };

    dim_t core_index = 0;
    bool can_run = TRUE;
    while ( ( can_run == TRUE ) && ( fseek( fin, 0L, SEEK_SET ) == 0 ) )
    {
        if ( bench_mode == 'p' )
        {
            can_run = FALSE;
        }
        else if ( bench_mode == 'a' )
        {
            // For accuracy testing, we test accuracy using multiple different
            // number of cores. This helps uncover any bugs related to over
            // subscription or varying thread factorizations.
            // Set current number of cores.
#ifdef BLIS_ENABLE_OPENMP
            omp_set_num_threads( list_omp_cores_for_testing[core_index] );
#endif
            printf( "Accuracy test using %ld threads.\n",
                            list_omp_cores_for_testing[core_index] );

            core_index++;
            if ( core_index < len_list_omp_cores_for_testing )
            {
                can_run = TRUE;
            }
            else
            {
                can_run = FALSE;
            }
        }

        // Input format: stor_type, mat_type, m, n, stride
        while ( fscanf( fin, "%c %c %ld %ld %ld %s\n",
                &stor_order, &mat_type, &m, &n,
                &stride, unpack_ops_type_str ) == 6 )
        {
            str_tolower( unpack_ops_type_str );

            stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
                            ( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
                            stor_order : 'r';

            if ( ( strcmp( unpack_ops_type_str, "bf16bf16f32of32" ) == 0 ) ||
                 ( strcmp( unpack_ops_type_str, "bf16bf16f32ofbf16" ) == 0 ) ||
                 ( strcmp( unpack_ops_type_str, "*" ) == 0 ) )
            {
                GEN_FUNC_NAME(unpack_bench_main_, bf16bf16f32of32)
                (
                    fout, stor_order, mat_type,
                    m, n, stride
                );
            }
        }
    }

    if( fin )
    {
        fclose( fin );
    }
    if( fout )
    {
        fclose( fout );
    }

    return 0;
}