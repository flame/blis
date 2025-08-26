/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "blis_int_type.h"

#include <stdio.h>

static const char* yesno(gint_t v) { return v ? "yes" : "no"; }

static void print_config_sizes(void)
{
    printf("== Configuration sizes ==\n");
    printf("version:                         %s\n",  bli_info_get_version_str());
    //printf("int type size (string):          %s\n",  bli_info_get_int_type_size_str());
    fprintf(stderr, "sizeof(bla_integer) =   %zu bytes\n", sizeof(bla_integer));
    printf("BLIS integer type size (bits):   %d\n",   (int)bli_info_get_int_type_size());
    printf("BLAS int type size (bits):       %lld\n", (long long)bli_info_get_blas_int_type_size());
    printf("num FP types:                    %lld\n", (long long)bli_info_get_num_fp_types());
    printf("max type size (bytes):           %lld\n", (long long)bli_info_get_max_type_size());
    printf("page size (bytes):               %lld\n", (long long)bli_info_get_page_size());
    printf("simd num registers:              %lld\n", (long long)bli_info_get_simd_num_registers());
    printf("simd size (bytes):               %lld\n", (long long)bli_info_get_simd_size());
    printf("simd align size (bytes):         %lld\n", (long long)bli_info_get_simd_align_size());
    printf("stack buf max size (bytes):      %lld\n", (long long)bli_info_get_stack_buf_max_size());
    printf("stack buf align size (bytes):    %lld\n", (long long)bli_info_get_stack_buf_align_size());
    printf("heap addr align size (bytes):    %lld\n", (long long)bli_info_get_heap_addr_align_size());
    printf("heap stride align size (bytes):  %lld\n", (long long)bli_info_get_heap_stride_align_size());
    printf("pool addr align size A (bytes):  %lld\n", (long long)bli_info_get_pool_addr_align_size_a());
    printf("pool addr align size B (bytes):  %lld\n", (long long)bli_info_get_pool_addr_align_size_b());
    printf("pool addr align size C (bytes):  %lld\n", (long long)bli_info_get_pool_addr_align_size_c());
    printf("pool addr align size GEN (bytes):%lld\n", (long long)bli_info_get_pool_addr_align_size_gen());
    printf("pool addr offset A (bytes):      %lld\n", (long long)bli_info_get_pool_addr_offset_size_a());
    printf("pool addr offset B (bytes):      %lld\n", (long long)bli_info_get_pool_addr_offset_size_b());
    printf("pool addr offset C (bytes):      %lld\n", (long long)bli_info_get_pool_addr_offset_size_c());
    printf("pool addr offset GEN (bytes):    %lld\n", (long long)bli_info_get_pool_addr_offset_size_gen());
}

static void print_features(void)
{
    printf("\n== Feature toggles ==\n");
    printf("BLAS API enabled:                %s\n", yesno(bli_info_get_enable_blas()));
    printf("CBLAS API enabled:               %s\n", yesno(bli_info_get_enable_cblas()));
    printf("PBA pools enabled:               %s\n", yesno(bli_info_get_enable_pba_pools()));
    printf("SBA pools enabled:               %s\n", yesno(bli_info_get_enable_sba_pools()));
    printf("Threading enabled:               %s\n", yesno(bli_info_get_enable_threading()));
    printf("OpenMP enabled:                  %s\n", yesno(bli_info_get_enable_openmp()));
    printf("Pthreads enabled:                %s\n", yesno(bli_info_get_enable_pthreads()));
    printf("JR/IR slab partitioning:         %s\n", yesno(bli_info_get_thread_part_jrir_slab()));
    printf("JR/IR RR partitioning:           %s\n", yesno(bli_info_get_thread_part_jrir_rr()));
    printf("Memkind enabled:                 %s\n", yesno(bli_info_get_enable_memkind()));
    printf("Sandbox enabled:                 %s\n", yesno(bli_info_get_enable_sandbox()));
    printf("integer type size (bits):      %lld\n", (long long)bli_info_get_blas_int_type_size());
}

static void print_info_value(void)
{
    printf("\n== Last BLIS info value (xerbla) ==\n");
    printf("info_value:                      %lld\n", (long long)bli_info_get_info_value());
}

static const char* dt_name(num_t dt)
{
    switch (dt)
    {
        case BLIS_FLOAT:    return "s (float)";
        case BLIS_DOUBLE:   return "d (double)";
        case BLIS_SCOMPLEX: return "c (scomplex)";
        case BLIS_DCOMPLEX: return "z (dcomplex)";
        default:            return "unknown";
    }
}

static void print_impl_for_dt(num_t dt)
{
    printf("  [%s]\n", dt_name(dt));
    printf("    gemm:   %s\n", bli_info_get_gemm_impl_string(dt));
    printf("    hemm:   %s\n", bli_info_get_hemm_impl_string(dt));
    printf("    herk:   %s\n", bli_info_get_herk_impl_string(dt));
    printf("    her2k:  %s\n", bli_info_get_her2k_impl_string(dt));
    printf("    symm:   %s\n", bli_info_get_symm_impl_string(dt));
    printf("    syrk:   %s\n", bli_info_get_syrk_impl_string(dt));
    printf("    syr2k:  %s\n", bli_info_get_syr2k_impl_string(dt));
    printf("    trmm:   %s\n", bli_info_get_trmm_impl_string(dt));
    printf("    trmm3:  %s\n", bli_info_get_trmm3_impl_string(dt));
    printf("    trsm:   %s\n", bli_info_get_trsm_impl_string(dt));
}

static void print_impl_overview(void)
{
    printf("\n== Level-3 implementation overview ==\n");
    /* Print for the common four datatypes; this will work regardless of which are internally optimized. */
    print_impl_for_dt(BLIS_FLOAT);
    print_impl_for_dt(BLIS_DOUBLE);
    print_impl_for_dt(BLIS_SCOMPLEX);
    print_impl_for_dt(BLIS_DCOMPLEX);
}

int main(void)
{
    bli_init();

    printf("========================================\n");
    printf("BLIS Library Information\n");
    printf("========================================\n");

    print_config_sizes();
    print_features();
    print_info_value();
    //print_impl_overview();
#if 0
    bla_integer n, k;
    n = 46341;
    k = n * (n + 1) / 2;
    printf("\n== Example of integer arithmetic ""n * (n+1) /2"" ==\n");
    printf("n = %lld, k = %lld\n", (long long)n, (long long)k);
    
    printf("Accurate way of computing 'n * (n + 1) / 2' is (n & 1) ? n * ((n + 1) / 2) : (n / 2) * (n + 1) is \n");
    k = (n & 1) ? n * ((n + 1) / 2) : (n / 2) * (n + 1);
    printf("Accurate k = %lld\n", (long long)k); 
#endif
    printf("\nDone.\n");
#if 0
     printf("sizeof(dim_t) = %zu bytes \n", sizeof(dim_t));

    dim_t kk = (dim_t)(n) * (n + 1) / 2;
    printf("dim_t kk = %ld\n", kk);

    dim_t nn = n;
    dim_t uk =  nn * (nn + 1) / 2;
    printf("dim_t with nn uk = %ld\n", uk);
#endif
    bli_finalize();
    return 0;
}
