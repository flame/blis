/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

//#define GLOB_DEF_PC_STR   ""
#define GLOB_DEF_DT_STR    "d"
#define GLOB_DEF_SC_STR    "ccc"
#define GLOB_DEF_IM_STR    "native"

#define GLOB_DEF_PS_STR    "50 1000 50"
#define GLOB_DEF_M_STR     "-1"
#define GLOB_DEF_N_STR     "-1"
#define GLOB_DEF_K_STR     "-1"

#define GLOB_DEF_NR_STR    "3"

#define GLOB_DEF_ALPHA_STR "1.0"
#define GLOB_DEF_BETA_STR  "1.0"

// String arrays allocated using this constant will always add 1 to
// the value defined below, and so the total allocated will still be
// a nice power of two.
#define MAX_STRING_SIZE    31


typedef struct params_s
{
	// Binary name.
	char bin[ MAX_STRING_SIZE + 1 ];

	// Operation name.
	char opname[ MAX_STRING_SIZE + 1 ];

	// Implementation name.
	char impl[ MAX_STRING_SIZE + 1 ];

#if 0
	char jc_nt_str[ MAX_STRING_SIZE + 1 ];
	char pc_nt_str[ MAX_STRING_SIZE + 1 ];
	char ic_nt_str[ MAX_STRING_SIZE + 1 ];
	char jr_nt_str[ MAX_STRING_SIZE + 1 ];
	char ir_nt_str[ MAX_STRING_SIZE + 1 ];
	long int jc_nt;
	long int pc_nt;
	long int ic_nt;
	long int jr_nt;
	long int ir_nt;
#endif

	// Multithreading parameters: number of threads and affinity string.
	char  nt_str[ MAX_STRING_SIZE + 1 ];
	char  af_str[ MAX_STRING_SIZE + 1 ];

	// Parameter combinations, datatype, operand storage combination,
	// and induced method.
	char  pc_str[ MAX_STRING_SIZE + 1 ];
	char  dt_str[ MAX_STRING_SIZE + 1 ];
	char  sc_str[ MAX_STRING_SIZE + 1 ];
	num_t dt;

	char  im_str[ MAX_STRING_SIZE + 1 ];
	ind_t im;

	// Problem size range and dimension specifiers.
	char     ps_str[ MAX_STRING_SIZE + 1 ];
	char     m_str[ MAX_STRING_SIZE + 1 ];
	char     n_str[ MAX_STRING_SIZE + 1 ];
	char     k_str[ MAX_STRING_SIZE + 1 ];
	long int sta;
	long int end;
	long int inc;
	long int m;
	long int n;
	long int k;

	// Number of repeats.
	char     nr_str[ MAX_STRING_SIZE + 1 ];
	long int nr;

	// Value of alpha and beta.
	char   alpha_str[ MAX_STRING_SIZE + 1 ];
	char   beta_str[ MAX_STRING_SIZE + 1 ];
	double alpha;
	double beta;

	// A flag controlling whether to print informational messages.
	bool verbose;

} params_t;

typedef void (*init_fp)( params_t* params );

// -----------------------------------------------------------------------------

void init_def_params( params_t* params );
void parse_cl_params( int argc, char** argv, init_fp fp, params_t* params );
void proc_params( params_t* params );

// -----------------------------------------------------------------------------

bool is_match( const char* str1, const char* str2 );
bool is_gemm( params_t* params );
bool is_hemm( params_t* params );
bool is_herk( params_t* params );
bool is_trmm( params_t* params );
bool is_trsm( params_t* params );

#ifdef __cplusplus
}
#endif

#endif
