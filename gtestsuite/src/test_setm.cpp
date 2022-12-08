#include "blis_test.h"
#include "blis_utils.h"

// Local prototypes.
void libblis_test_setm_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_setm_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    x
);

double libblis_test_setm_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x
);

double libblis_test_bitrp_setm(
  test_params_t* params,
  iface_t        iface,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         r,
  num_t          dt
){

  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    libblis_test_setm_impl( iface, beta, r );
    resid = libblis_test_bitrp_matrix(x, r, dt);
  }

  return resid;
}

double libblis_test_op_setm (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m, n;
  obj_t        beta;
  obj_t        x;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, n, &x );

  // Initialize beta to unit.
  bli_copysc( &BLIS_ONE, &beta );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    // Randomize x.
    libblis_test_mobj_randomize( params, FALSE, &x );
  }
  else {
    libblis_test_mobj_irandomize( params, &x );
  }

  libblis_test_setm_impl( iface, &beta, &x );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                sc_str[0], m, n, &r );

    resid = libblis_test_bitrp_setm( params, iface, &beta, &x, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_test_setm_check( params, &beta, &x );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_setm_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    x
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_setm( beta, x );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_setm_check(
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x
){
  num_t dt_x     = bli_obj_dt( x );
  dim_t m_x      = bli_obj_length( x );
  dim_t n_x      = bli_obj_width( x );
  inc_t rs_x     = bli_obj_row_stride( x );
  inc_t cs_x     = bli_obj_col_stride( x );
  void* buf_x    = (void*)bli_obj_buffer_at_off( x );
  void* buf_beta = (void*)bli_obj_buffer_for_1x1( dt_x, beta );
  dim_t i, j;
  double resid   = 0.0;

  // The easiest way to check that setm was successful is to confirm
  // that each element of x is equal to beta.

  if      ( bli_obj_is_float( x ) ) {
    float*    beta_cast  = (float*)buf_beta;
    float*    buf_x_cast = (float*)buf_x;
    float*    chi1;

    for ( j = 0; j < n_x; ++j ) {
      for ( i = 0; i < m_x; ++i ) {
        chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;
        if ( !bli_seq( *chi1, *beta_cast ) ) {
          resid = 1.0;
          return resid;
        }
      }
    }
  }
  else if ( bli_obj_is_double( x ) ) {
    double*   beta_cast  = (double*)buf_beta;
    double*   buf_x_cast = (double*)buf_x;
    double*   chi1;

    for ( j = 0; j < n_x; ++j ) {
      for ( i = 0; i < m_x; ++i ) {
        chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;
        if ( !bli_deq( *chi1, *beta_cast ) ) {
          resid = 1.0;
          return resid;
        }
      }
    }
  }
  else if ( bli_obj_is_scomplex( x ) ) {
    scomplex* beta_cast  = (scomplex*)buf_beta;
    scomplex* buf_x_cast = (scomplex*)buf_x;
    scomplex* chi1;

    for ( j = 0; j < n_x; ++j ) {
      for ( i = 0; i < m_x; ++i ) {
        chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;
        if ( !bli_ceq( *chi1, *beta_cast ) ) {
          resid = 1.0;
          return resid;
        }
      }
    }
  }
  else /* if ( bli_obj_is_dcomplex( x ) )*/  {
    dcomplex* beta_cast  = (dcomplex*)buf_beta;
    dcomplex* buf_x_cast = (dcomplex*)buf_x;
    dcomplex* chi1;

    for ( j = 0; j < n_x; ++j ) {
      for ( i = 0; i < m_x; ++i ) {
        chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;
        if ( !bli_zeq( *chi1, *beta_cast ) ) {
          resid = 1.0;
          return resid;
        }
      }
    }
  }
  return resid;
}