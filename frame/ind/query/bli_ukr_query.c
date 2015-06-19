/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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


// 3mh micro-kernels
extern func_t* gemm3mh_ukrs;

// 3m3 micro-kernels
extern func_t* gemm3m3_ukrs;

// 3m2 micro-kernels
extern func_t* gemm3m2_ukrs;

// 3m1 micro-kernels
extern func_t* gemm3m1_ukrs;
extern func_t* gemmtrsm3m1_l_ukrs;
extern func_t* gemmtrsm3m1_u_ukrs;
extern func_t* trsm3m1_l_ukrs;
extern func_t* trsm3m1_u_ukrs;

// 4mh micro-kernels
extern func_t* gemm4mh_ukrs;

// 4m1b micro-kernels
extern func_t* gemm4mb_ukrs;

// 4m1a micro-kernels
extern func_t* gemm4m1_ukrs;
extern func_t* gemmtrsm4m1_l_ukrs;
extern func_t* gemmtrsm4m1_u_ukrs;
extern func_t* trsm4m1_l_ukrs;
extern func_t* trsm4m1_u_ukrs;

// Native micro-kernels
extern func_t* gemm_ukrs;
extern func_t* gemmtrsm_l_ukrs;
extern func_t* gemmtrsm_u_ukrs;
extern func_t* trsm_l_ukrs;
extern func_t* trsm_u_ukrs;

// Reference micro-kernels
extern func_t* gemm_ref_ukrs;
extern func_t* gemmtrsm_l_ref_ukrs;
extern func_t* gemmtrsm_u_ref_ukrs;
extern func_t* trsm_l_ref_ukrs;
extern func_t* trsm_u_ref_ukrs;

//
// NOTE: We have to use the address of the func_t*, since the value
// will not yet be set at compile-time (since they are allocated at
// runtime).
//
static func_t** bli_ukrs[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_UKRS] =
{
        /*   gemm   gemmtrsm_l   gemmtrsm_u   trsm_l   trsm_u   */
/* 3mh  */ { &gemm3mh_ukrs,                NULL,                NULL,
                                           NULL,                NULL, },
/* 3m3  */ { &gemm3m3_ukrs,                NULL,                NULL,
                                           NULL,                NULL, },
/* 3m2  */ { &gemm3m2_ukrs,                NULL,                NULL,
                                           NULL,                NULL, },
/* 3m1  */ { &gemm3m1_ukrs, &gemmtrsm3m1_l_ukrs, &gemmtrsm3m1_u_ukrs,
                                &trsm3m1_l_ukrs,     &trsm3m1_u_ukrs, },
/* 4mh  */ { &gemm4mh_ukrs,                NULL,                NULL,
                                           NULL,                NULL, },
/* 4mb  */ { &gemm4mb_ukrs,                NULL,                NULL,
                                           NULL,                NULL, },
/* 4m1  */ { &gemm4m1_ukrs, &gemmtrsm4m1_l_ukrs, &gemmtrsm4m1_u_ukrs,
                                &trsm4m1_l_ukrs,     &trsm4m1_u_ukrs, },
/* nat  */ { &gemm_ukrs,       &gemmtrsm_l_ukrs,    &gemmtrsm_u_ukrs,
                                   &trsm_l_ukrs,        &trsm_u_ukrs, },
};

static func_t** bli_ref_ukrs[BLIS_NUM_LEVEL3_UKRS] =
{
	&gemm_ref_ukrs,
	&gemmtrsm_l_ref_ukrs,
	&gemmtrsm_u_ref_ukrs,
	&trsm_l_ref_ukrs,
	&trsm_u_ref_ukrs,
};

static char* bli_ukr_impl_str[BLIS_NUM_UKR_IMPL_TYPES] =
{
	"refrnce",
	"virtual",
	"optimzd",
	"notappl",
};

// -----------------------------------------------------------------------------

char* bli_ukr_impl_string( l3ukr_t ukr, ind_t method, num_t dt )
{
	func_t* p;
	kimpl_t ki;

//printf( "ukr method dt = %u %u %u\n", ukr, method, dt );
	// Look up the ukr func_t for the given ukr type and method.
	p = bli_ukr_get_funcs( ukr, method );

	// Check whether the ukrs func_t is NULL for the given ukr type.
	// If the queried ukr func_t is NULL, return the string for not
	// applicable. Otherwise, query the ukernel implementation type
	// using the method provided and return the associated string.
	if ( p == NULL ) ki = BLIS_NOTAPPLIC_UKERNEL;
	else             ki = bli_ukr_impl_type( ukr, method, dt );

	return bli_ukr_impl_str[ ki ];
}

// -----------------------------------------------------------------------------

char* bli_ukr_avail_impl_string( l3ukr_t ukr, num_t dt )
{
	opid_t  oper;
	ind_t   method;
	kimpl_t ki;

	// We need to decide which operation we will use to query the
	// current available induced method. If the ukr type given is
	// BLIS_GEMM_UKR, we use gemm. Otherwise, we use trsm (since
	// the four other defined ukr types are trsm-related).
	if ( ukr == BLIS_GEMM_UKR ) oper = BLIS_GEMM;
	else                        oper = BLIS_TRSM;

	// Query the current available induced method using the
	// chosen operation id type.
	method = bli_ind_oper_find_avail( oper, dt );

	// Query the ukernel implementation type using the current
	// available method.
	ki = bli_ukr_impl_type( ukr, method, dt );

	return bli_ukr_impl_str[ ki ];
}

// -----------------------------------------------------------------------------

kimpl_t bli_ukr_impl_type( l3ukr_t ukr, ind_t method, num_t dt )
{
	// If the current available induced method is not native, it
	// must be virtual.
	if ( method != BLIS_NAT ) return BLIS_VIRTUAL_UKERNEL;
	else
	{
		// If the current available induced method for the gemm
		// operation is native, then it might be reference or
		// optimized. To determine which, we compare the
		// datatype-specific function pointer within the ukrs
		// object corresponding to the current available induced
		// method to the typed function pointer within the known
		// reference ukrs object.
		func_t* funcs     = bli_ukr_get_funcs( ukr, method );
		void*   p         = bli_func_obj_query( dt, funcs );
		func_t* ref_funcs = bli_ukr_get_ref_funcs( ukr );
		void*   ref_p     = bli_func_obj_query( dt, ref_funcs );
	
		if ( p == ref_p ) return BLIS_REFERENCE_UKERNEL;
		else              return BLIS_OPTIMIZED_UKERNEL;
	}
}

// -----------------------------------------------------------------------------

func_t* bli_ukr_get_funcs( l3ukr_t ukr, ind_t method )
{
	func_t** p = bli_ukrs[ method ][ ukr ];

	// Initialize the cntl API, if it isn't already initialized. This is
	// needed because we have to ensure that the ukr func_t objects have
	// been created (and thus contain valid function pointers).
	bli_cntl_init();

	// Avoid dereferencing NULL pointers. (A NULL pointer indicates that
	// there is no kernel for the requested kernel type and method.)
	if ( p == NULL ) return NULL;
	else             return *p;
}

func_t* bli_ukr_get_ref_funcs( l3ukr_t ukr )
{
	func_t** p = bli_ref_ukrs[ ukr ];

	// Initialize the cntl API, if it isn't already initialized. This is
	// needed because we have to ensure that the ukr func_t objects have
	// been created (and thus contain valid function pointers).
	bli_cntl_init();

	// Avoid dereferencing NULL pointers. (A NULL pointer indicates that
	// there is no reference kernel for the requested kernel type.)
	if ( p == NULL ) return NULL;
	else             return *p;
}

