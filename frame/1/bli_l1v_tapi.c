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

// Guard the function definitions so that they are only compiled when
// #included from files that define the typed API macros.
#ifdef BLIS_ENABLE_TAPI

// -- Optional OpenMP threading for memory-bound level-1v operations ----------
// BLIS runs level-1/2 single-threaded, which leaves most of the memory
// bandwidth on the table for large vectors (a single GB10 X925 core sees
// ~45 GB/s vs ~118 GB/s aggregate). When BLIS_ENABLE_L1_OPENMP is defined,
// large contiguous-enough level-1v calls are split across OpenMP threads.
#ifdef BLIS_ENABLE_L1_OPENMP
#include <omp.h>
#ifndef BLIS_L1_MT_THRESHOLD
#define BLIS_L1_MT_THRESHOLD 200000   // min elements to bother threading
#endif
#ifndef BLIS_L1_MT_MAX
#define BLIS_L1_MT_MAX       256      // cap on partial-sum buffer / threads
#endif
#ifndef BLI_L1V_MT_HELPERS
#define BLI_L1V_MT_HELPERS
BLIS_INLINE bool bli_l1v_mt_ok( dim_t n )
{
	// Large enough, not already nested in a parallel region, and within the
	// fixed partial-sum buffer size used by the reduction path.
	return ( n >= ( dim_t )BLIS_L1_MT_THRESHOLD ) &&
	       ( omp_get_active_level() == 0 ) &&
	       ( omp_get_max_threads() <= BLIS_L1_MT_MAX );
}
// Contiguous [start,len) sub-range for the calling thread -- a plain EQUAL
// split. Even on heterogeneous machines (mixed fast/slow cores) equal work per
// thread is the right split for these ops: they are pure streaming with no data
// reuse, so they are memory-bandwidth bound and every core makes memory
// progress at nearly the same rate regardless of its compute width (e.g. on
// GB10 the fast-core:slow-core per-core throughput is ~1.0-1.1 for axpy). This
// is unlike compute-bound level-3 GEMM, whose data reuse would justify giving
// the faster cores proportionally more work.
BLIS_INLINE void bli_l1v_range( dim_t n, dim_t* start, dim_t* len )
{
	const dim_t nt   = omp_get_num_threads();
	const dim_t tid  = omp_get_thread_num();
	const dim_t base = n / nt;
	const dim_t rem  = n % nt;
	*start = tid * base + ( tid < rem ? tid : rem );
	*len   = base + ( tid < rem ? 1 : 0 );
}
#endif

// axpyv/scal2v: disjoint output; split the range and call the kernel per chunk.
#define bli_l1v_axpyv_kercall( ch, ctype, f, conjx, n, alpha, x, incx, y, incy, cntx ) \
{ \
	if ( bli_l1v_mt_ok( n ) ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			dim_t s_, l_; bli_l1v_range( (n), &s_, &l_ ); \
			if ( l_ > 0 ) \
				f( conjx, l_, ( ctype* )(alpha), ( ctype* )(x) + s_*(incx), (incx), \
				   (y) + s_*(incy), (incy), ( cntx_t* )(cntx) ); \
		} \
	} \
	else \
		f( conjx, (n), ( ctype* )(alpha), ( ctype* )(x), (incx), (y), (incy), ( cntx_t* )(cntx) ); \
}

// dotv: reduction; each thread computes a partial dot, combined afterward.
#define bli_l1v_dotv_kercall( ch, ctype, f, conjx, conjy, n, x, incx, y, incy, rho, cntx ) \
{ \
	if ( bli_l1v_mt_ok( n ) ) \
	{ \
		ctype parts_[ BLIS_L1_MT_MAX ]; \
		dim_t nt_ = 1; \
		_Pragma( "omp parallel" ) \
		{ \
			const dim_t tid_ = omp_get_thread_num(); \
			dim_t s_, l_; bli_l1v_range( (n), &s_, &l_ ); \
			ctype pr_; bli_tset0s( ch, pr_ ); \
			if ( tid_ == 0 ) nt_ = omp_get_num_threads(); \
			if ( l_ > 0 ) \
				f( conjx, conjy, l_, ( ctype* )(x) + s_*(incx), (incx), \
				   ( ctype* )(y) + s_*(incy), (incy), &pr_, ( cntx_t* )(cntx) ); \
			parts_[ tid_ ] = pr_; \
		} \
		ctype acc_; bli_tset0s( ch, acc_ ); \
		for ( dim_t t_ = 0; t_ < nt_; ++t_ ) bli_tadds( ch, ch, ch, parts_[ t_ ], acc_ ); \
		*(rho) = acc_; \
	} \
	else \
		f( conjx, conjy, (n), ( ctype* )(x), (incx), ( ctype* )(y), (incy), (rho), ( cntx_t* )(cntx) ); \
}

// copyv/addv/subv: two-vector, disjoint output; split the range.
#define bli_l1v_copyv_kercall( ch, ctype, f, conjx, n, x, incx, y, incy, cntx ) \
{ \
	if ( bli_l1v_mt_ok( n ) ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			dim_t s_, l_; bli_l1v_range( (n), &s_, &l_ ); \
			if ( l_ > 0 ) \
				f( conjx, l_, ( ctype* )(x) + s_*(incx), (incx), (y) + s_*(incy), (incy), ( cntx_t* )(cntx) ); \
		} \
	} \
	else \
		f( conjx, (n), ( ctype* )(x), (incx), (y), (incy), ( cntx_t* )(cntx) ); \
}

// scalv/invscalv/setv: single in-place vector, disjoint; split the range.
#define bli_l1v_scalv_kercall( ch, ctype, f, conjalpha, n, alpha, x, incx, cntx ) \
{ \
	if ( bli_l1v_mt_ok( n ) ) \
	{ \
		_Pragma( "omp parallel" ) \
		{ \
			dim_t s_, l_; bli_l1v_range( (n), &s_, &l_ ); \
			if ( l_ > 0 ) \
				f( conjalpha, l_, ( ctype* )(alpha), (x) + s_*(incx), (incx), ( cntx_t* )(cntx) ); \
		} \
	} \
	else \
		f( conjalpha, (n), ( ctype* )(alpha), (x), (incx), ( cntx_t* )(cntx) ); \
}
#else
#define bli_l1v_axpyv_kercall( ch, ctype, f, conjx, n, alpha, x, incx, y, incy, cntx ) \
	f( conjx, (n), ( ctype* )(alpha), ( ctype* )(x), (incx), (y), (incy), ( cntx_t* )(cntx) )
#define bli_l1v_dotv_kercall( ch, ctype, f, conjx, conjy, n, x, incx, y, incy, rho, cntx ) \
	f( conjx, conjy, (n), ( ctype* )(x), (incx), ( ctype* )(y), (incy), (rho), ( cntx_t* )(cntx) )
#define bli_l1v_copyv_kercall( ch, ctype, f, conjx, n, x, incx, y, incy, cntx ) \
	f( conjx, (n), ( ctype* )(x), (incx), (y), (incy), ( cntx_t* )(cntx) )
#define bli_l1v_scalv_kercall( ch, ctype, f, conjalpha, n, alpha, x, incx, cntx ) \
	f( conjalpha, (n), ( ctype* )(alpha), (x), (incx), ( cntx_t* )(cntx) )
#endif

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	bli_l1v_copyv_kercall( ch, ctype, f, conjx, n, x, incx, y, incy, cntx ); \
}

INSERT_GENTFUNC_BASIC( addv,  BLIS_ADDV_KER )
INSERT_GENTFUNC_BASIC( copyv, BLIS_COPYV_KER )
INSERT_GENTFUNC_BASIC( subv,  BLIS_SUBV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t  n, \
       const ctype* x, inc_t incx, \
             dim_t* index  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  ( ctype* )x, incx, \
	  index, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( amaxv, BLIS_AMAXV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpbyv, BLIS_AXPBYV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) \
		cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	bli_l1v_axpyv_kercall( ch, ctype, f, conjx, n, alpha, x, incx, y, incy, cntx ); \
}

INSERT_GENTFUNC_BASIC( axpyv,  BLIS_AXPYV_KER )
INSERT_GENTFUNC_BASIC( scal2v, BLIS_SCAL2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	bli_l1v_dotv_kercall( ch, ctype, f, conjx, conjy, n, x, incx, y, incy, rho, cntx ); \
}

INSERT_GENTFUNC_BASIC( dotv, BLIS_DOTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
       const ctype* beta, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  conjy, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	  ( ctype* )beta, \
	            rho, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotxv, BLIS_DOTXV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  x, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invertv, BLIS_INVERTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	bli_l1v_scalv_kercall( ch, ctype, f, conjalpha, n, alpha, x, incx, cntx ); \
}

INSERT_GENTFUNC_BASIC( invscalv, BLIS_INVSCALV_KER )
INSERT_GENTFUNC_BASIC( scalv, BLIS_SCALV_KER )
INSERT_GENTFUNC_BASIC( setv,  BLIS_SETV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx, \
       ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  x, incx, \
	  y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( swapv, BLIS_SWAPV_KER )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( xpbyv, BLIS_XPBYV_KER )


#endif

