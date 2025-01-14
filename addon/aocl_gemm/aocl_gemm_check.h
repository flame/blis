/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define AOCL_GEMM_CHECK( op_str, \
                         order, transa, transb, \
                         m, n, k, \
                         a, lda, mtag_a, \
                         b, ldb, mtag_b, \
                         c, ldc, \
                         err_no \
                       ) \
{ \
    int32_t info = 0; \
    bool col_stored, row_stored; \
    bool nota, notb, ta, tb; \
 \
    col_stored = ( order == 'c' ) || ( order == 'C' ); \
    row_stored = ( order == 'r' ) || ( order == 'R' ); \
 \
    nota = ( transa == 'n' ) || ( transa == 'N' ); \
    notb = ( transb == 'n' ) || ( transb == 'N' ); \
 \
    ta   = ( transa == 't' ) || ( transa == 'T' ); \
    tb   = ( transb == 't' ) || ( transb == 'T' ); \
 \
    if( ( order != 'r') && ( order != 'R' ) && ( order != 'c' ) && ( order != 'C' ) ) \
        info = 1; \
    else if( ( transa != 'n' ) && ( transa != 'N' ) && ( transa != 't' ) && ( transa != 'T' ) ) \
        info = 2; \
    else if( ( transb != 'n' ) && ( transb != 'N' ) && ( transb != 't' ) && ( transb != 'T' ) ) \
        info = 3; \
    else if ( m <= 0 ) \
        info = 4; \
    else if ( n <= 0 ) \
        info = 5; \
    else if ( k <= 0 ) \
        info = 6; \
    else if ( a == NULL ) \
        info = 8; \
    else if ( row_stored && ( ( nota && ( lda < k ) ) || ( ta && ( lda < m ) ) ) ) \
        info = 9; \
    else if ( col_stored && ( ( nota && ( lda < m ) ) || ( ta && ( lda < k ) ) ) ) \
        info = 9; \
    else if ( ( mtag_a != 'n' ) && ( mtag_a != 'N' ) && \
              ( mtag_a != 'p' ) && ( mtag_a != 'P' ) && \
              ( mtag_a != 'r' ) && ( mtag_a != 'R' ) ) \
        info = 10; \
    else if ( b == NULL ) \
        info = 11; \
    else if ( row_stored && ( ( notb && ( ldb < n ) ) || ( tb && ( ldb < k ) ) ) ) \
        info = 12; \
    else if ( col_stored && ( ( notb && ( ldb < k ) ) || ( tb && ( ldb < n ) ) ) ) \
        info = 12; \
    else if ( ( mtag_b != 'n' ) && ( mtag_b != 'N' ) && \
              ( mtag_b != 'p' ) && ( mtag_b != 'P' ) && \
              ( mtag_b != 'r' ) && ( mtag_b != 'R' ) ) \
        info = 13; \
    else if ( c == NULL ) \
        info = 15; \
    else if ( row_stored && ( ldc < n ) ) \
        info = 16; \
    else if ( col_stored && ( ldc < m ) ) \
        info = 16; \
 \
    if( info != 0 ) \
    { \
        char print_msg[ 100 ]; \
 \
        sprintf( print_msg, "** On entry to %6s, parameter number %2i had an illegal value", op_str, info); \
        bli_print_msg(print_msg, __FILE__, __LINE__); \
        err_no = info; \
    } \
}

#define AOCL_BATCH_GEMM_CHECK( op_str, \
                         order, transa, transb, \
                         gemm_no, \
                         m, n, k, \
                         a, lda, mtag_a, \
                         b, ldb, mtag_b, \
                         c, ldc, \
                         err_no \
                       ) \
{ \
    int32_t info = 0; \
    bool col_stored, row_stored; \
    bool nota, notb, ta, tb; \
 \
    col_stored = ( order == 'c' ) || ( order == 'C' ); \
    row_stored = ( order == 'r' ) || ( order == 'R' ); \
 \
    nota = ( transa == 'n' ) || ( transa == 'N' ); \
    notb = ( transb == 'n' ) || ( transb == 'N' ); \
 \
    ta   = ( transa == 't' ) || ( transa == 'T' ); \
    tb   = ( transb == 't' ) || ( transb == 'T' ); \
 \
    if( ( order != 'r') && ( order != 'R' ) && ( order != 'c' ) && ( order != 'C' ) ) \
        info = 1; \
    else if( ( transa != 'n' ) && ( transa != 'N' ) && ( transa != 't' ) && ( transa != 'T' ) ) \
        info = 2; \
    else if( ( transb != 'n' ) && ( transb != 'N' ) && ( transb != 't' ) && ( transb != 'T' ) ) \
        info = 3; \
    else if ( m <= 0 ) \
        info = 5; \
    else if ( n <= 0 ) \
        info = 6; \
    else if ( k <= 0 ) \
        info = 7; \
    else if ( a == NULL ) \
        info = 9; \
    else if ( row_stored && ( ( nota && ( lda < k ) ) || ( ta && ( lda < m ) ) ) ) \
        info = 10; \
    else if ( col_stored && ( ( nota && ( lda < m ) ) || ( ta && ( lda < k ) ) ) ) \
        info = 10; \
    else if ( ( mtag_a != 'n' ) && ( mtag_a != 'N' ) && \
              ( mtag_a != 'p' ) && ( mtag_a != 'P' ) && \
              ( mtag_a != 'r' ) && ( mtag_a != 'R' ) ) \
        info = 11; \
    else if ( b == NULL ) \
        info = 12; \
    else if ( row_stored && ( ( notb && ( ldb < n ) ) || ( tb && ( ldb < k ) ) ) ) \
        info = 13; \
    else if ( col_stored && ( ( notb && ( ldb < k ) ) || ( tb && ( ldb < n ) ) ) ) \
        info = 13; \
    else if ( ( mtag_b != 'n' ) && ( mtag_b != 'N' ) && \
              ( mtag_b != 'p' ) && ( mtag_b != 'P' ) && \
              ( mtag_b != 'r' ) && ( mtag_b != 'R' ) ) \
        info = 14; \
    else if ( c == NULL ) \
        info = 16; \
    else if ( row_stored && ( ldc < n ) ) \
        info = 17; \
    else if ( col_stored && ( ldc < m ) ) \
        info = 17; \
 \
    if( info != 0 ) \
    { \
        char print_msg[ 150 ]; \
 \
        sprintf( print_msg, "** On entry to %6s, parameter number %2i of problem %ld had an illegal value", op_str, info, (long int) gemm_no); \
        bli_print_msg(print_msg, __FILE__, __LINE__); \
        err_no = info; \
    } \
}

#define AOCL_UTIL_ELTWISE_OPS_CHECK( op_str, \
                         order, transa, transb, \
                         m, n, \
                         a, lda, \
                         b, ldb \
                       ) \
{ \
    int32_t info = 0; \
    bool col_stored, row_stored; \
    bool nota, notb, ta, tb; \
 \
    col_stored = ( order == 'c' ) || ( order == 'C' ); \
    row_stored = ( order == 'r' ) || ( order == 'R' ); \
 \
    nota = ( transa == 'n' ) || ( transa == 'N' ); \
    notb = ( transb == 'n' ) || ( transb == 'N' ); \
 \
    ta   = ( transa == 't' ) || ( transa == 'T' ); \
    tb   = ( transb == 't' ) || ( transb == 'T' ); \
 \
    if( ( order != 'r') && ( order != 'R' ) && ( order != 'c' ) && ( order != 'C' ) ) \
        info = 1; \
    else if( ( transa != 'n' ) && ( transa != 'N' ) && ( transa != 't' ) && ( transa != 'T' ) ) \
        info = 2; \
    else if( ( transb != 'n' ) && ( transb != 'N' ) && ( transb != 't' ) && ( transb != 'T' ) ) \
        info = 3; \
    else if ( m <= 0 ) \
        info = 4; \
    else if ( n <= 0 ) \
        info = 5; \
    else if ( a == NULL ) \
        info = 6; \
    else if ( row_stored && ( ( nota && ( lda < n ) ) || ( ta && ( lda < m ) ) ) ) \
        info = 7; \
    else if ( col_stored && ( ( nota && ( lda < m ) ) || ( ta && ( lda < n ) ) ) ) \
        info = 8; \
    else if ( b == NULL ) \
        info = 9; \
    else if ( row_stored && ( ( notb && ( ldb < n ) ) || ( tb && ( ldb < m ) ) ) ) \
        info = 10; \
    else if ( col_stored && ( ( notb && ( ldb < m ) ) || ( tb && ( ldb < n ) ) ) ) \
        info = 11; \
 \
    if( info != 0 ) \
    { \
        char print_msg[ 100 ]; \
 \
        sprintf( print_msg, "** On entry to %6s, parameter number %2i had an illegal value", op_str, info); \
        bli_print_msg(print_msg, __FILE__, __LINE__); \
        return; \
    } \
}
