#include "blis_test.h"
#include "blis_utils.h"
#include "test_amaxv.h"

// Local prototypes.
void libblis_test_amaxv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_amaxv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    index
);

double libblis_test_amaxv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         index
);

void bli_amaxv_test (
  obj_t*  x,
  obj_t*  index
);

double cblas_amaxv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  gint_t*    idx,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      *idx = cblas_isamax( m, xp, incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      *idx = cblas_idamax( m, xp, incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      *idx = cblas_icamax( m, xp, incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      *idx = cblas_izamax( m, xp, incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_amaxv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  gint_t*    idx,
  num_t      dt
){
  gint_t index = 1;
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      index = isamax_( &m, xp, &incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      index = idamax_( &m, xp, &incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      index = icamax_( &m, xp, &incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      index = izamax_( &m, xp, &incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  *idx = (index - 1);
  return 0;
}

void libblis_api_amaxv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         index,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_amaxv_impl( iface, x, index );
  }
  else { /*CLBAS  || BLAS */
    dim_t  m     = bli_obj_vector_dim( x );
    f77_int incx = bli_obj_vector_inc( x );
    gint_t *idx  = (gint_t *)bli_obj_buffer( index );

      if( params->api == API_CBLAS ) {
        cblas_amaxv( m, x, incx, idx, dt );
      } else {
        blas_amaxv( m, x, incx, idx, dt );;
      }
  }
  return ;
}

double libblis_ref_amaxv(
  test_params_t* params,
  obj_t*  x,
  obj_t*  index
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_amaxv_check( params, x, index );
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iamaxv_check( params, x, index );
    }
    else {
      resid = libblis_test_vector_check(params, x);
    }
  }

  return resid;
}

double libblis_test_bitrp_amaxv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         index,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_obj_scalar_init_detached( BLIS_INT, r );
    libblis_test_amaxv_impl( iface, x, r );
    resid = libblis_test_bitrp_vector(index, r, dt);
  }
  return resid;
}

double libblis_test_op_amaxv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m;
  obj_t        x;
  obj_t        index;
  double resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Create test scalars.
  bli_obj_scalar_init_detached( BLIS_INT, &index );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );

  // Randomize x.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, FALSE, &x );
  } else {
    libblis_test_vobj_irandomize( params, &x );
  }

  libblis_api_amaxv( params, iface, &x, &index, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    resid = libblis_test_bitrp_amaxv( params, iface, &x, &index, &r, datatype );

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_amaxv( params, &x, &index );
  }
#endif

  // Zero out performance and residual if input vector is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_amaxv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    index
) {

	switch ( iface )
	{
		 case BLIS_TEST_SEQ_FRONT_END:
		   bli_amaxv( x, index );
		 break;

		default:
		  libblis_test_printf_error( "Invalid interface type.\n" );
	 }
}

double libblis_test_amaxv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         index
) {
  obj_t index_test;
  obj_t chi_i;
  obj_t chi_i_test;
  dim_t i;
  dim_t i_test;

  double i_d, junk;
  double i_d_test;

  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is randomized.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   index := amaxv( x )
  //
  // is functioning correctly if
  //
  //   x[ index ] = max( x )
  //
  // where max() is implemented via the bli_?amaxv_test() function.
  //

  // The following two calls have already been made by the caller. That
  // is, the index object has already been created and the library's
  // amaxv implementation has already been tested.
  //bli_obj_scalar_init_detached( BLIS_INT, &index );
  //bli_amaxv( x, &index );
  bli_getsc( index, &i_d, &junk ); i = i_d;

  // If x is length 0, then we can't access any elements, and so we
  // return early with a good residual.
  if ( bli_obj_vector_dim( x ) == 0 ) { resid = 0.0; return resid; }

  bli_acquire_vi( i, x, &chi_i );

  bli_obj_scalar_init_detached( BLIS_INT, &index_test );
  bli_amaxv_test( x, &index_test );
  bli_getsc( &index_test, &i_d_test, &junk ); i_test = i_d_test;
  bli_acquire_vi( i_test, x, &chi_i_test );

  // Verify that the values referenced by index and index_test are equal.
  if ( bli_obj_equals( &chi_i, &chi_i_test ) ) resid = 0.0;
  else                                         resid = 1.0;

  return resid;
}

// -----------------------------------------------------------------------------

//
// Prototype BLAS-like interfaces with typed operands for a local amaxv test
// operation
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t           n, \
       ctype* restrict x, inc_t incx, \
       dim_t* restrict index  \
     ); \

INSERT_GENTPROT_BASIC0( amaxv_test )


//
// Prototype function pointer query interface.
//

#undef  GENPROT
#define GENPROT( tname, opname ) \
\
PASTECH(tname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt );

GENPROT( amaxv, amaxv_test )


//
// Define function pointer query interfaces.
//

#undef  GENFRONT
#define GENFRONT( tname, opname ) \
\
GENARRAY_FPA( PASTECH(tname,_vft), \
              opname ); \
\
PASTECH(tname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt ) \
{ \
    return PASTECH(opname,_fpa)[ dt ]; \
}

GENFRONT( amaxv, amaxv_test )


//
// Define object-based interface for a local amaxv test operation.
//

#undef  GENFRONT
#define GENFRONT( tname, opname ) \
\
void PASTEMAC0(opname) \
     ( \
       obj_t*  x, \
       obj_t*  index  \
     ) \
{ \
    num_t     dt        = bli_obj_dt( x ); \
\
    dim_t     n         = bli_obj_vector_dim( x ); \
    void*     buf_x     = bli_obj_buffer_at_off( x ); \
    inc_t     incx      = bli_obj_vector_inc( x ); \
\
    dim_t*    buf_index = (dim_t*)bli_obj_buffer_at_off( index ); \
\
/*
	FGVZ: Disabling this code since bli_amaxv_check() is supposed to be a
	non-public API function, and therefore unavailable unless all symbols
	are scheduled to be exported at configure-time (which is not currently
	the default behavior).

    if ( bli_error_checking_is_enabled() ) \
        bli_amaxv_check( x, index ); \
*/ \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(tname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
       n, \
       buf_x, incx, \
       buf_index  \
    ); \
}

GENFRONT( amaxv, amaxv_test )


//
// Define BLAS-like interfaces with typed operands for a local amaxv test
// operation.
// NOTE: This is based on a simplified version of the bli_?amaxv_ref()
// reference kernel.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       dim_t*   index  \
     ) \
{ \
	ctype_r* minus_one = PASTEMAC(chr,m1); \
	dim_t*   zero_i    = PASTEMAC(i,0); \
\
	ctype_r  chi1_r; \
	ctype_r  chi1_i; \
	ctype_r  abs_chi1; \
	ctype_r  abs_chi1_max; \
	dim_t    index_l; \
	dim_t    i; \
\
	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */ \
	if ( bli_zero_dim1( n ) ) \
	{ \
		PASTEMAC(i,copys)( *zero_i, *index ); \
		return; \
	} \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( *zero_i, index_l ); \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	PASTEMAC(chr,copys)( *minus_one, abs_chi1_max ); \
\
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			ctype* chi1 = x + (i  )*incx; \
\
			/* Get the real and imaginary components of chi1. */ \
			PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
			PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			PASTEMAC(chr,set0s)( abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				index_l       = i; \
			} \
		} \
	} \
\
	/* Store the final index to the output variable. */ \
	PASTEMAC(i,copys)( index_l, *index ); \
}
INSERT_GENTFUNCR_BASIC0( amaxv_test )