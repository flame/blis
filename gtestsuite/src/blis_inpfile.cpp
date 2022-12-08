#include "blis_utils.h"
#include "blis_api.h"
#include "blis_inpfile.h"

int libblis_test_read_randv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld \n",
         api_name, &dt, &m) == 3) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "randv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_randv( params, iface, params->dc_str[0],
                                                params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_randm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld %ld\n",
         api_name, &dt, &m, &n) == 4) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "randm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_randm( params, iface, params->dc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_addv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld\n",
         api_name, &dt, &transx, &m ) == 4) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "addv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_addv( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_amaxv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;

   // Variables extracted from the logs which are used by bench
  char stor_scheme;
  inc_t incx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld %ld \n", api_name, &dt, &m, &incx) == 4)
  {
    params->sc_str[0][0] = stor_scheme;

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "amaxv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_amaxv( params, iface, params->dc_str[0],
                                 params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_axpbyv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

	inc_t incx, incy;
	double alpha_r, alpha_i, beta_r, beta_i;

  if(sscanf(str, "%s %c %ld %lf %lf %ld %lf %lf %ld\n",
			         api_name, &dt, &m,	&alpha_r, &alpha_i, &incx,
            &beta_r, &beta_i, &incy ) == 9 ){

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real  = beta_r;
    params->beta->imag  = beta_i;
  }

  double resid       = 0.0;
  const char* op_str = "axpbyv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_axpbyv( params, iface, params->dc_str[0], params->pc_str[0],
                                  params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_axpyv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char  stor_scheme, conjx = 'n';
  stor_scheme = 'c';
  inc_t incx, incy;
	 double alpha_r, alpha_i;

  if(sscanf(str, "%s %c %ld %lf %lf %ld %ld \n",
         api_name, &dt, &m, &alpha_r, &alpha_i, &incx, &incy ) == 7) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(conjx) ){
      params->pc_str[0][0]  = tolower(conjx);
    } else {
      params->pc_str[0][0]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;

    params->dim[0].m  = m;
    params->dim[0].n  = 0;
    params->dim[0].k  = 0;
  }

  double resid       = 0.0;
  const char* op_str = "axpyv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_axpyv( params, iface, params->dc_str[0], params->pc_str[0],
                                                params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_copyv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t incx, incy;

  if(sscanf(str, "%s %c %ld %ld %ld\n",
        api_name, &dt, &m, &incx, &incy) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = 0;
    params->dim[0].k  = 0;
  }

  double resid       = 0.0;
  const char* op_str = "copyv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_copyv( params, iface, params->dc_str[0],
                                 params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_dotv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';
  inc_t incx, incy;

  if(sscanf(str, "%s %c %ld %ld %ld\n",
    api_name, &dt, &m, &incx, &incy) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    params->pc_str[0][0]  = 'n';
    params->pc_str[0][1]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "dotv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_dotv( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh,
                        params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_dotxv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, conjx, conjy;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld\n",
    api_name, &dt, &conjx, &conjy, &m) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(conjx) ){
      params->pc_str[0][0]  = tolower(conjx);
    } else {
      params->pc_str[0][0]  = conjx;
    }

    if (isalpha(conjy) ){
      params->pc_str[0][1]  = tolower(conjy);
    } else {
      params->pc_str[0][1]  = conjy;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "dotxv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_dotxv( params, iface, params->dc_str[0], params->pc_str[0],
                                 params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_normfv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld\n",
    api_name, &dt, &m) == 3) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "normfv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_normfv( params, iface, params->dc_str[0],
                                  params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh,
                          params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_scal2v_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld \n",
         api_name, &dt, &transx, &m ) == 4) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "scal2v";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_scal2v( params, iface, params->dc_str[0], params->pc_str[0],
                                                 params->sc_str[0], &t,  params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params,op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_scalv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  double alpha_r, alpha_i;
  inc_t incx;

  if(sscanf(str, "%s %c %lf %lf %ld %ld\n",
      api_name, &dt, &alpha_r, &alpha_i, &m, &incx) == 6){

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
  }

  double resid       = 0.0;
  const char* op_str = "scalv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_scalv( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_setv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld \n",
         api_name, &dt, &m) == 3) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "setv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_setv( params, iface, params->dc_str[0],
                                               params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_subv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld\n",
         api_name, &dt, &transx, &m ) == 4) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "subv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_subv( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_xpbyv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  char transx;

  if(sscanf(str, "%s %c %c %ld\n",
                 api_name, &dt, &transx, &m) == 4){

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "xpbyv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_xpbyv( params, iface, params->dc_str[0],
                                 params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_axpyf_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char  stor_scheme, conja, conjx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld\n",
         api_name, &dt, &conja, &conjx, &m ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(conja) ){
      params->pc_str[0][0]  = tolower(conja);
    } else {
      params->pc_str[0][0]  = conja;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "axpyf";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_axpyf( params, op, iface, params->dc_str[0],
                                 params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_axpy2v_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char  stor_scheme, conjx, conjy;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld\n",
         api_name, &dt, &conjx, &conjy, &m ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(conjx) ){
      params->pc_str[0][0]  = tolower(conjx);
    } else {
      params->pc_str[0][0]  = conjx;
    }

    if (isalpha(conjy) ){
      params->pc_str[0][1]  = tolower(conjy);
    } else {
      params->pc_str[0][1]  = conjy;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "axpy2v";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_axpy2v( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_dotxf_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, conjat, conjx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld\n",
    api_name, &dt, &conjat, &conjx, &m) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(conjat) ){
      params->pc_str[0][0]  = tolower(conjat);
    } else {
      params->pc_str[0][0]  = conjat;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "dotxf";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_dotxf( params, op, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_dotaxpyv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, conjxt, conjx, conjy;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %c %ld\n",
    api_name, &dt, &conjxt, &conjx, &conjy, &m) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(conjxt) ){
      params->pc_str[0][0]  = tolower(conjxt);
    } else {
      params->pc_str[0][0]  = conjxt;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(conjy) ){
      params->pc_str[0][1]  = tolower(conjy);
    } else {
      params->pc_str[0][1]  = conjy;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "dotaxpyv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_dotaxpyv( params, iface, params->dc_str[0],
                 params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_dotxaxpyf_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, conjat, conja, conjw, conjx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %c %c %ld\n",
    api_name, &dt, &conjat, &conja, &conjw, &conjx, &m) == 7) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;
    params->sc_str[0][3] = stor_scheme;

    if (isalpha(conjat) ){
      params->pc_str[0][0]  = tolower(conjat);
    } else {
      params->pc_str[0][0]  = conjat;
    }

    if (isalpha(conja) ){
      params->pc_str[0][1]  = tolower(conja);
    } else {
      params->pc_str[0][1]  = conja;
    }

    if (isalpha(conjw) ){
      params->pc_str[0][2]  = tolower(conjw);
    } else {
      params->pc_str[0][2]  = conjw;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][3]  = tolower(conjx);
    } else {
      params->pc_str[0][3]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "dotxaxpyf";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_dotxaxpyf( params, op, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_addm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "addm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_addm( params, iface, params->dc_str[0],
                               params->pc_str[0], params->sc_str[0], &t );

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_axpym_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "axpym";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_axpym( params, iface, params->dc_str[0],
                 params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_copym_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';


  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "copym";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_copym( params, iface, params->dc_str[0],
                                 params->pc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_normfm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m,n;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld %ld\n",
    api_name, &dt, &m, &n) == 4) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "normfm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_normfm( params, iface, params->dc_str[0],
                                  params->pc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_scal2m_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "scal2m";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_scal2m( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t,  params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_scalm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "scalm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_scalm( params, iface, params->dc_str[0],
                 params->pc_str[0], params->sc_str[0], &t,  params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_setm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %ld %ld\n",
         api_name, &dt, &m, &n) == 4) {

    params->sc_str[0][0] = stor_scheme;

    params->pc_str[0][0]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "setm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_setm( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_subm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char  stor_scheme, transx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld\n",
         api_name, &dt, &transx, &m, &n ) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "subm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_subm( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_xpbym_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m,n;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  char transx;

  if(sscanf(str, "%s %c %c %ld %ld\n",
                 api_name, &dt, &transx, &m, &n) == 5){

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(transx) ){
      params->pc_str[0][0]  = tolower(transx);
    } else {
      params->pc_str[0][0]  = transx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
  }

  double resid       = 0.0;
  const char* op_str = "xpbym";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_xpbym( params, iface, params->dc_str[0],
                          params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;

   // Variables extracted from the logs which are used by bench
  char stor_scheme, transA;
  double alpha_r, beta_r, alpha_i, beta_i;
  inc_t lda;
  inc_t incx, incy;

  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
      api_name, &dt, &transA, &m, &n,  &alpha_r, &alpha_i, &lda,
                  &incx, &beta_r, &beta_i, &incy) == 12)
  {
    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(transA) ){
      params->pc_str[0][0]  = tolower(transA);
    } else {
      params->pc_str[0][0]  = transA;
    }

    if(transA == 'C' || transA == 'c')
      params->pc_str[0][1]  = 'c';
    else /*if(transA == 'N' || transA == 'n')*/
      params->pc_str[0][1]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real  = beta_r;
    params->beta->imag  = beta_i;

    params->ld[0] = lda;
  }

  double resid       = 0.0;
  const char* op_str = "gemv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_gemv( params, iface, params->dc_str[0], params->pc_str[0],
                                params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_ger_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  double alpha_r, alpha_i;
  inc_t lda;
  inc_t incx, incy;

  if(sscanf(str, "%s %c %ld %ld %lf %lf %ld %ld %ld\n",
      api_name, &dt, &m, &n, &alpha_r, &alpha_i, &incx, &incy, &lda) == 9) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][0]  = 'n';
    params->pc_str[0][1]  = 'n';

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;

    params->ld[0] = lda;
  }

  double resid       = 0.0;
  const char* op_str = "ger";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_ger( params, iface, params->dc_str[0],
                  params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_hemv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, uploa, conja, conjx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %c %ld\n",
    api_name, &dt, &uploa, &conja, &conjx, &m) == 6) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uploa) ){
      params->pc_str[0][0]  = tolower(uploa);
    } else {
      params->pc_str[0][0]  = uploa;
    }

    if (isalpha(conja) ){
      params->pc_str[0][1]  = tolower(conja);
    } else {
      params->pc_str[0][1]  = conja;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][2]  = tolower(conjx);
    } else {
      params->pc_str[0][2]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "hemv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_hemv( params, iface, params->dc_str[0], params->pc_str[0],
                                params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_her_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, uploa, conjx;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld\n",
    api_name, &dt, &uploa, &conjx, &m) == 5) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(uploa) ){
      params->pc_str[0][0]  = tolower(uploa);
    } else {
      params->pc_str[0][0]  = uploa;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "her";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_her( params, iface, params->dc_str[0],
                   params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_her2_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme, uploa, conjx, conjy;
  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %c %ld\n",
    api_name, &dt, &uploa, &conjx, &conjy, &m) == 6) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uploa) ){
      params->pc_str[0][0]  = tolower(uploa);
    } else {
      params->pc_str[0][0]  = uploa;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(conjy) ){
      params->pc_str[0][2]  = tolower(conjy);
    } else {
      params->pc_str[0][2]  = conjy;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
  }

  double resid       = 0.0;
  const char* op_str = "her2";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_her2( params, iface, params->dc_str[0],
                                params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_symv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda;
  char uplo, conja, conjx;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str, "%s %c %c %c %c %ld %lf %lf %lu %lf %lf \n",
		   api_name, &dt, &uplo, &conja, &conjx, &m, &alpha_r, &alpha_i,
                    &lda, &beta_r, &beta_i) == 11) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(conja) ){
      params->pc_str[0][1]  = tolower(conja);
    } else {
      params->pc_str[0][1]  = conja;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][2] = tolower(dt);
    } else {
      params->dc_str[0][2] = dt;
    }

    params->dim[0].m  = m;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
  }

  double resid       = 0.0;
  const char* op_str = "symv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_symv( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_syr_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  char uplo, conjx;
  double alpha_r, alpha_i;

  if(sscanf(str, "%s %c %c %c %ld %lf %lf \n",
		   api_name, &dt, &uplo, &conjx, &m, &alpha_r, &alpha_i) == 7) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
  }

  double resid       = 0.0;
  const char* op_str = "syr";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_syr( params, iface, params->dc_str[0],
                 params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_syr2_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  char uplo, conjx;
  double alpha_r, alpha_i;

  if(sscanf(str, "%s %c %c %c %ld %lf %lf \n",
		   api_name, &dt, &uplo, &conjx, &m, &alpha_r, &alpha_i) == 7) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(conjx) ){
      params->pc_str[0][1]  = tolower(conjx);
    } else {
      params->pc_str[0][1]  = conjx;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
  }

  double resid       = 0.0;
  const char* op_str = "syr2";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_syr2( params, iface, params->dc_str[0],
                  params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params,op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_trmv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda;
  char uploa, transA, diaga;
  inc_t incx;

  if(sscanf(str, "%s %c %c %c %c %ld %ld %ld\n",
    api_name, &dt, &uploa, &transA, &diaga, &m, &lda, &incx) == 8){

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(uploa) ){
      params->pc_str[0][0]  = tolower(uploa);
    } else {
      params->pc_str[0][0]  = uploa;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(diaga) ){
      params->pc_str[0][2]  = tolower(diaga);
    } else {
      params->pc_str[0][2]  = diaga;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->ld[0] = lda;
  }

  double resid       = 0.0;
  const char* op_str = "trmv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_trmv( params, iface, params->dc_str[0],
                  params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_trsv_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda;
  char uploa, transA, diaga;
  inc_t incx;

  if(sscanf(str, "%s %c %c %c %c %ld %ld %ld\n",
    api_name, &dt, &uploa, &transA, &diaga, &m, &lda, &incx) == 8){

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(uploa) ){
      params->pc_str[0][0]  = tolower(uploa);
    } else {
      params->pc_str[0][0]  = uploa;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(diaga) ){
      params->pc_str[0][2]  = tolower(diaga);
    } else {
      params->pc_str[0][2]  = diaga;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;

    params->ld[0] = lda;
  }

  double resid       = 0.0;
  const char* op_str = "trsv";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_trsv( params, iface, params->dc_str[0],
                   params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n, k;
  char  dt;

   // Variables extracted from the logs which are used by bench
  char stor_scheme, transA, transB;
  double alpha_r, beta_r, alpha_i, beta_i;
  inc_t lda, ldb, ldc;

  stor_scheme = 'c';

  if(sscanf(str, "%s %c %c %c %ld %ld %ld %lf %lf %ld %ld %lf %lf %ld[^\n]",
         api_name, &dt, &transA, &transB, &m, &n, &k, &alpha_r, &alpha_i,
          &lda, &ldb, &beta_r, &beta_i, &ldc) == 14) {

    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';


    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(transA) ){
      params->pc_str[0][0]  = tolower(transA);
    } else {
      params->pc_str[0][0]  = transA;
    }

    if (isalpha(transB) ){
      params->pc_str[0][1]  = tolower(transB);
    } else {
      params->pc_str[0][1]  = transB;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "gemm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_gemm( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemmt_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda, ldb, ldc;
  char transA, transB, uplo;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str,"%s %c %c %ld %ld %lu %lu %lu %c %c %lf %lf %lf %lf\n",\
    api_name, &dt, &uplo, &m, &n, &lda, &ldb, &ldc, &transA, &transB,
    &alpha_r, &alpha_i, &beta_r, &beta_i) == 14) {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if(isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(transB) ){
      params->pc_str[0][2]  = tolower(transB);
    } else {
      params->pc_str[0][2]  = transB;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real  = beta_r;
    params->beta->imag  = beta_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "gemmt";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_gemmt( params, iface, params->dc_str[0], params->pc_str[0],
                           params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_hemm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda, ldb, ldc;
  char side, uploa ;
  double alpha_r, alpha_i, beta_r, beta_i;

  if(sscanf(str, "%s %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
        api_name, &dt, &side, &uploa, &m, &n, &alpha_r, &alpha_i,
        &lda, &ldb, &beta_r, &beta_i, &ldc) == 13) {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(side) ){
      params->pc_str[0][0]  = tolower(side);
    } else {
      params->pc_str[0][0]  = side;
    }

    if (isalpha(uploa) ){
      params->pc_str[0][1]  = tolower(uploa);
    } else {
      params->pc_str[0][1]  = uploa;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "hemm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_hemm( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_herk_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda, ldc;
  char transA, uplo;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str, "%s %c %c %c %ld %ld %lf %lf %lu %lf %lf %lu\n",
		   api_name, &dt, &uplo, &transA, &m, &n, &alpha_r, &alpha_i,
                    &lda, &beta_r, &beta_i, &ldc) == 12) {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "herk";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_herk( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_her2k_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda, ldc;
  char transA, transB, uplo;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str, "%s %c %c %c %c %ld %ld %lf %lf %lu %lf %lf %lu\n",
		   api_name, &dt, &uplo, &transA, &transB, &m, &n,
     &alpha_r, &alpha_i, &lda, &beta_r, &beta_i, &ldc) == 12)
  {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(transB) ){
      params->pc_str[0][1]  = tolower(transB);
    } else {
      params->pc_str[0][1]  = transB;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "her2k";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_her2k( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_symm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda, ldb, ldc;
  char side, uploa ;
  double alpha_r, alpha_i, beta_r, beta_i;

  if(sscanf(str, "%s %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
        api_name, &dt, &side, &uploa, &m, &n, &alpha_r, &alpha_i,
        &lda, &ldb, &beta_r, &beta_i, &ldc) == 13) {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(side) ){
      params->pc_str[0][0]  = tolower(side);
    } else {
      params->pc_str[0][0]  = side;
    }

    if (isalpha(uploa) ){
      params->pc_str[0][1]  = tolower(uploa);
    } else {
      params->pc_str[0][1]  = uploa;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "symm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_symm( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_syrk_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda, ldc;
  char transA, uplo;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str, "%s %c %c %c %ld %ld %lf %lf %lu %lf %lf %lu\n",
		   api_name, &dt, &uplo, &transA, &m, &n, &alpha_r, &alpha_i,
                    &lda, &beta_r, &beta_i, &ldc) == 12) {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "syrk";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_syrk( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_syr2k_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  inc_t lda, ldc;
  char transA, transB, uplo;
  double alpha_r, beta_r, alpha_i, beta_i;

  if(sscanf(str, "%s %c %c %c %c %ld %ld %lf %lf %lu %lf %lf %lu\n",
		   api_name, &dt, &uplo, &transA, &transB, &m, &n,
     &alpha_r, &alpha_i, &lda, &beta_r, &beta_i, &ldc) == 12)
  {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    if (isalpha(uplo) ){
      params->pc_str[0][0]  = tolower(uplo);
    } else {
      params->pc_str[0][0]  = uplo;
    }

    if (isalpha(transA) ){
      params->pc_str[0][1]  = tolower(transA);
    } else {
      params->pc_str[0][1]  = transA;
    }

    if (isalpha(transB) ){
      params->pc_str[0][1]  = tolower(transB);
    } else {
      params->pc_str[0][1]  = transB;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "syr2k";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_syr2k( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_trmm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda, ldb;
  char side, uploa, transa, diaga;
  double alpha_r, alpha_i;

  if(sscanf(str, "%s %c %c %c %c %c %ld %ld %ld %ld %lf %lf\n",
        api_name, &dt, &side, &uploa, &transa, &diaga, &m, &n,
        &lda, &ldb, &alpha_r, &alpha_i) == 12) {

    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(side) ){
      params->pc_str[0][0]  = tolower(side);
    } else {
      params->pc_str[0][0]  = side;
    }

    if (isalpha(uploa) ){
      params->pc_str[0][1]  = tolower(uploa);
    } else {
      params->pc_str[0][1]  = uploa;
    }

    if (isalpha(transa) ){
      params->pc_str[0][2]  = tolower(transa);
    } else {
      params->pc_str[0][2]  = transa;
    }

    if (isalpha(diaga) ){
      params->pc_str[0][3]  = tolower(diaga);
    } else {
      params->pc_str[0][3]  = diaga;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
  }

  double resid       = 0.0;
  const char* op_str = "trmm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_trmm( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_trmm3_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda, ldb, ldc;
  char side, uploa, transa, transb, diaga;
  double alpha_r, alpha_i, beta_r, beta_i;

  if(sscanf(str, "%s %c %c %c %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
        api_name, &dt, &side, &uploa, &transa, &transb, &diaga, &m, &n,
        &alpha_r, &alpha_i, &lda, &ldb, &beta_r, &beta_i, &ldc) == 16)
  {
    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(side) ){
      params->pc_str[0][0]  = tolower(side);
    } else {
      params->pc_str[0][0]  = side;
    }

    if (isalpha(uploa) ){
      params->pc_str[0][1]  = tolower(uploa);
    } else {
      params->pc_str[0][1]  = uploa;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;
    params->beta->real   = beta_r;
    params->beta->imag   = beta_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;
  }

  double resid       = 0.0;
  const char* op_str = "trmm3";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_trmm3( params, iface, params->dc_str[0], params->pc_str[0],
                             params->sc_str[0], &t, params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_trsm_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[10];
  dim_t m, n;
  char  dt;
  char stor_scheme;
  stor_scheme = 'c';

  dim_t lda, ldb;
  char side, uploa, transa, diaga;
  double alpha_r, alpha_i;

  if(sscanf(str, "%s %c %c %c %c %c %ld %ld %ld %ld %lf %lf\n",
        api_name, &dt, &side, &uploa, &transa, &diaga, &m, &n,
        &lda, &ldb, &alpha_r, &alpha_i) == 12) {

    if( m == lda )
      stor_scheme = 'c';
    if( n == lda )
      stor_scheme = 'r';

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;

    if (isalpha(side) ){
      params->pc_str[0][0]  = tolower(side);
    } else {
      params->pc_str[0][0]  = side;
    }

    if (isalpha(uploa) ){
      params->pc_str[0][1]  = tolower(uploa);
    } else {
      params->pc_str[0][1]  = uploa;
    }

    if (isalpha(transa) ){
      params->pc_str[0][2]  = tolower(transa);
    } else {
      params->pc_str[0][2]  = transa;
    }

    if (isalpha(diaga) ){
      params->pc_str[0][3]  = tolower(diaga);
    } else {
      params->pc_str[0][3]  = diaga;
    }

    if (isalpha(dt) ){
      params->dc_str[0][0] = tolower(dt);
    } else {
      params->dc_str[0][0] = dt;
    }

    params->dim[0].m  = m;
    params->dim[0].n  = n;

    params->alpha->real  = alpha_r;
    params->alpha->imag  = alpha_i;

    params->ld[0] = lda;
    params->ld[1] = ldb;
  }

  double resid       = 0.0;
  const char* op_str = "trsm";
  ind_t mt           = BLIS_NAT;
  iface_t iface      = BLIS_TEST_SEQ_FRONT_END;
  tensor_t *dim      = params->dim;
  tensor_t t         = dim[0];

  resid = libblis_test_op_trsm( params, iface, params->dc_str[0],
          params->pc_str[0], params->sc_str[0], &t, params->alpha[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_u8s8s32os32_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_u8s8s32os32";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_u8s8s32os32( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_u8s8s32os8_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_u8s8s32os8";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_u8s8s32os8( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_f32f32f32of32_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_f32f32f32of32";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_f32f32f32of32( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_u8s8s16os16_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_u8s8s16os16";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_u8s8s16os16( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_u8s8s16os8_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_u8s8s16os8";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_u8s8s16os8( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_bf16bf16f32of32_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_bf16bf16f32of32";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_bf16bf16f32of32( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

int libblis_test_read_gemm_bf16bf16f32obf16_params( char* str, test_op_t* op,
                                    test_params_t* params, printres_t* pfr) {
  char api_name[25];
  dim_t m, n, k;
  char stor_scheme = 'c';

   // Variables extracted from the logs which are used by bench
  inc_t lda, ldb, ldc;
  char op_t;

  if( sscanf( str, "%s %c %c %ld %ld %ld %ld %ld %ld\n",
    api_name, &stor_scheme, &op_t, &m, &n, &k,
    &lda, &ldb, &ldc ) == 9 ) {

    params->sc_str[0][0] = stor_scheme;
    params->sc_str[0][1] = stor_scheme;
    params->sc_str[0][2] = stor_scheme;

    params->pc_str[0][1]  = 'n';
    params->pc_str[0][2]  = 'n';

    params->dim[0].m  = m;
    params->dim[0].n  = n;
    params->dim[0].k  = k;

    params->ld[0] = lda;
    params->ld[1] = ldb;
    params->ld[2] = ldc;

    params->op_t = op_t;

    params->alpha->real  = 2;
    params->alpha->imag  = 0;
    params->beta->real   = 9;
    params->beta->imag   = 0;

  }

  double resid         = 0.0;
  const char* op_str   = "gemm_bf16bf16f32obf16";
  params->dc_str[0][0] = 's';
  ind_t mt             = BLIS_NAT;
  tensor_t *dim        = params->dim;
  tensor_t t           = dim[0];

  resid = libblis_test_op_gemm_bf16bf16f32obf16( params, params->pc_str[0],
                         params->sc_str[0], &t,  params->alpha[0], params->beta[0]);

  char* res_str = libblis_test_result (resid, thresh, params->dc_str[0], params );

  char buffer[125];
  libblis_build_function_string(params, op->opid, op_str, mt, 0, 0, 0, buffer);

  displayProps(buffer, params, op, &t, resid, res_str, pfr);

  return 0;
}

void libblis_read_api(test_ops_t* ops, opid_t opid, dimset_t dimset,
                                   unsigned int n_params, test_op_t* op ) {

  if ( op->op_switch == ENABLE_ONLY ){
    return;
  }

  // Initialize the operation type field.
  op->opid = opid;

  op->op_switch = ENABLE_ONLY ;

  // Check the op_switch for the individual override value.
  if ( op->op_switch == ENABLE_ONLY )	{
    ops->indiv_over = TRUE;
  }

  op->n_dims = libblis_test_get_n_dims_from_dimset( dimset );
  op->dimset = dimset;

  if ( op->n_dims > MAX_NUM_DIMENSIONS ) {
     libblis_test_printf_error( "Detected too many dimensions (%u) in input file to store.\n",
                              op->n_dims );
  }

  if ( n_params > 0 ) {
    op->n_params = n_params;
  }
  else {
    op->n_params = 0;
    strcpy( op->params, "" );
  }

  // Initialize the "test done" switch.
  op->test_done = FALSE;

  // Initialize the parent pointer.
  op->ops = ops;

  return;
}

void libblis_read_inpops(string ss, test_params_t* params, test_ops_t* ops,
                                                    string api, printres_t* pfr){
  char str[125];
  strcpy( str, ss.c_str() );

  	/* Utility operations */
  if       (api ==  "randv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->randv) );
    libblis_test_read_randv_params(str, &(ops->randv), params, pfr);
  } else if(api ==  "randm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->randm) );
    libblis_test_read_randm_params(str, &(ops->randm), params, pfr);
  } else
  	/* 	Level-1v */
  if( (api ==  "addv") || (api ==  "add") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->addv) );
    libblis_test_read_addv_params(str, &(ops->addv), params, pfr);
  } else if( (api ==  "amaxv") || (api ==  "amax") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->amaxv) );
    libblis_test_read_amaxv_params(str, &(ops->amaxv), params, pfr);
  } else if( (api ==  "axpbyv") || (api ==  "axpby") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->axpbyv) );
    libblis_test_read_axpbyv_params(str, &(ops->axpbyv), params, pfr);
  } else if( (api ==  "axpyv") || (api ==  "axpy") ){
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->axpyv) );
    libblis_test_read_axpyv_params(str, &(ops->axpyv), params, pfr);
  } else if((api ==  "copyv") || (api ==  "copy") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->copyv) );
    libblis_test_read_copyv_params(str, &(ops->copyv), params, pfr);
  } else if( (api ==  "dotv") || (api ==  "dot")) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->dotv) );
    libblis_test_read_dotv_params(str, &(ops->dotv), params, pfr);
  } else if( (api ==  "dotxv") || (api ==  "dotx") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->dotxv) );
    libblis_test_read_dotxv_params(str, &(ops->dotxv), params, pfr);
  } else if( (api ==  "normfv") || (api ==  "nrm2") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->normfv) );
    libblis_test_read_normfv_params(str, &(ops->normfv), params, pfr);
  } else if( (api ==  "scalv") || (api ==  "scal") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->scalv) );
    libblis_test_read_scalv_params(str, &(ops->scalv), params, pfr);
  } else if( (api ==  "scal2v") || (api ==  "scal2") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->scal2v) );
    libblis_test_read_scal2v_params(str, &(ops->scal2v), params, pfr);
  } else if( (api ==  "setv")  || (api ==  "set") ){
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->setv) );
    libblis_test_read_setv_params(str, &(ops->setv), params, pfr);
  } else if( (api ==  "subv") || (api ==  "sub") ) {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->subv) );
    libblis_test_read_subv_params(str, &(ops->subv), params, pfr);
  } else if( (api ==  "xpbyv") || (api ==  "xpby") ){
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->xpbyv) );
    libblis_test_read_xpbyv_params(str, &(ops->xpbyv), params, pfr);
  } else
  	/* 	Level-1m */
  if(api ==  "addm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->addm) );
    libblis_test_read_addm_params(str, &(ops->addm), params, pfr);
  } else if(api ==  "axpym") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->axpym) );
    libblis_test_read_axpym_params(str, &(ops->axpym), params, pfr);
  } else if(api ==  "copym") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->copym) );
    libblis_test_read_copym_params(str, &(ops->copym), params, pfr);
  } else if(api ==  "normfm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->normfm) );
    libblis_test_read_normfm_params(str, &(ops->normfm), params, pfr);
  } else if(api ==  "scalm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->scalm) );
    libblis_test_read_scal2m_params(str, &(ops->scalm), params, pfr);
  } else if(api ==  "scal2m") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->scal2m) );
    libblis_test_read_scalm_params(str, &(ops->scal2m), params, pfr);
  } else if(api ==  "setm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->setm) );
    libblis_test_read_setm_params(str, &(ops->setm), params, pfr);
  } else if(api ==  "subm") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->subm) );
    libblis_test_read_subm_params(str, &(ops->subm), params, pfr);
  } else if(api ==  "xpbym") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->xpbym) );
    libblis_test_read_xpbym_params(str, &(ops->xpbym), params, pfr);
  } else
  	/* 	Level-1f */
  if(api ==  "axpy2v") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->axpy2v) );
    libblis_test_read_axpy2v_params(str, &(ops->axpy2v), params, pfr);
  } else if(api ==  "dotaxpyv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->dotaxpyv) );
    libblis_test_read_dotaxpyv_params(str, &(ops->dotaxpyv), params, pfr);
  } else if(api ==  "axpyf") {
	   libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MF,  2, &(ops->axpyf) );
    libblis_test_read_axpyf_params(str, &(ops->axpyf), params, pfr);
  } else if(api ==  "dotxf") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MF,  2, &(ops->dotxf) );
    libblis_test_read_dotxf_params(str, &(ops->dotxf), params, pfr);
  } else if(api ==  "dotxaxpyf") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MF,  4, &(ops->dotxaxpyf) );
    libblis_test_read_dotxaxpyf_params(str, &(ops->dotxaxpyf), params, pfr);
  } else
	  /* Level-2 */
   if(api ==  "gemv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  2, &(ops->gemv) );
    libblis_test_read_gemv_params(str, &(ops->gemv), params, pfr);
  } else if(api ==  "ger") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_MN,  2, &(ops->ger) );
    libblis_test_read_ger_params(str, &(ops->ger), params, pfr);
  } else if(api ==  "hemv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->hemv) );
    libblis_test_read_hemv_params(str, &(ops->hemv), params, pfr);
  } else if(api ==  "her") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->her) );
    libblis_test_read_her_params(str, &(ops->her), params, pfr);
  } else if(api ==  "her2") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->her2) );
    libblis_test_read_her2_params(str, &(ops->her2), params, pfr);
  } else if(api ==  "symv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->symv) );
    libblis_test_read_symv_params(str, &(ops->symv), params, pfr);
  } else if(api ==  "syr") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->syr) );
    libblis_test_read_syr_params(str, &(ops->syr), params, pfr);
  } else if(api ==  "syr2") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->syr2) );
    libblis_test_read_syr2_params(str, &(ops->syr2), params, pfr);
  } else if(api ==  "trmv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->trmv) );
    libblis_test_read_trmv_params(str, &(ops->trmv), params, pfr);
  } else if(api ==  "trsv") {
    libblis_read_api( ops, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->trsv) );
    libblis_test_read_trsv_params(str, &(ops->trsv), params, pfr);
  } else
	  /* Level-3 */
   if(api ==  "gemm") {
    libblis_read_api( ops, BLIS_GEMM,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm) );
    libblis_test_read_gemm_params(str, &(ops->gemm), params, pfr);
  } else if(api ==  "gemmt") {
    libblis_read_api( ops, BLIS_GEMMT, BLIS_TEST_DIMS_MK,  3, &(ops->gemmt) );
    libblis_test_read_gemmt_params(str, &(ops->gemmt), params, pfr);
  } else if(api ==  "hemm") {
    libblis_read_api( ops, BLIS_HEMM,  BLIS_TEST_DIMS_MN,  2, &(ops->hemm) );
    libblis_test_read_hemm_params(str, &(ops->hemm), params, pfr);
  }  else if(api ==  "herk") {
    libblis_read_api( ops, BLIS_HERK,  BLIS_TEST_DIMS_MK,  2, &(ops->herk) );
    libblis_test_read_herk_params(str, &(ops->herk), params, pfr);
  }  else if(api ==  "her2k") {
    libblis_read_api( ops, BLIS_HER2K, BLIS_TEST_DIMS_MK,  2, &(ops->her2k) );
    libblis_test_read_her2k_params(str, &(ops->her2k), params, pfr);
  } else if(api ==  "symm") {
    libblis_read_api( ops, BLIS_SYMM,  BLIS_TEST_DIMS_MN,  2, &(ops->symm) );
    libblis_test_read_symm_params(str, &(ops->symm), params, pfr);
  }  else if(api ==  "syrk") {
    libblis_read_api( ops, BLIS_SYRK,  BLIS_TEST_DIMS_MK,  2, &(ops->syrk) );
    libblis_test_read_syrk_params(str, &(ops->syrk), params, pfr);
  }  else if(api ==  "syr2k") {
    libblis_read_api( ops, BLIS_SYR2K, BLIS_TEST_DIMS_MK,  2, &(ops->syr2k) );
    libblis_test_read_syr2k_params(str, &(ops->syr2k), params, pfr);
  } else if(api ==  "trmm") {
    libblis_read_api( ops, BLIS_TRMM,  BLIS_TEST_DIMS_MN,  4, &(ops->trmm) );
    libblis_test_read_trmm_params(str, &(ops->trmm), params, pfr);
  } else if(api ==  "trmm3") {
    libblis_read_api( ops, BLIS_TRMM3, BLIS_TEST_DIMS_MN,  5, &(ops->trmm3) );
    libblis_test_read_trmm3_params(str, &(ops->trmm3), params, pfr);
  } else if(api ==  "trsm") {
    libblis_read_api( ops, BLIS_TRSM,  BLIS_TEST_DIMS_MN,  4, &(ops->trsm) );
    libblis_test_read_trsm_params(str, &(ops->trsm), params, pfr);
  } else
	  /* LPGEMM */
  if(api ==  "gemm_u8s8s32os32") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s32os32) );
    libblis_test_read_gemm_u8s8s32os32_params(str, &(ops->gemm_u8s8s32os32), params, pfr);
  }
  else if(api ==  "gemm_u8s8s32os8") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s32os8) );
    libblis_test_read_gemm_u8s8s32os8_params(str, &(ops->gemm_u8s8s32os8), params, pfr);
  }
  else if(api ==  "gemm_f32f32f32of32") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_f32f32f32of32) );
    libblis_test_read_gemm_f32f32f32of32_params(str, &(ops->gemm_f32f32f32of32), params, pfr);
  }
  else if(api ==  "gemm_u8s8s16os16") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s16os16) );
    libblis_test_read_gemm_u8s8s16os16_params(str, &(ops->gemm_u8s8s16os16), params, pfr);
  }
  else if(api ==  "gemm_u8s8s16os8") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s16os8) );
    libblis_test_read_gemm_u8s8s16os8_params(str, &(ops->gemm_u8s8s16os8), params, pfr);
  }
  else if(api ==  "gemm_bf16bf16f32of32") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_bf16bf16f32of32) );
    libblis_test_read_gemm_bf16bf16f32of32_params(str, &(ops->gemm_bf16bf16f32of32), params, pfr);
  }
  else if(api ==  "gemm_bf16bf16f32obf16") {
    libblis_read_api( ops, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_bf16bf16f32obf16) );
    libblis_test_read_gemm_bf16bf16f32obf16_params(str, &(ops->gemm_bf16bf16f32obf16), params, pfr);
  }
  else {
    printf("Invalid api option : ");
    cout << ss << endl;
  }
  return;
}

void libblis_read_inpprms(string str, test_params_t* params, test_ops_t* ops, printres_t* pfr) {

  stringstream ss;
  string api, wrd;

  ss << str;

  str = "";
  ss >> wrd;
  api = wrd.substr(1, (wrd.length()-2));
//  api = wrd.substr(0, wrd.length());

  str = str + api;

  // Running loop till end of stream
  while (!ss.eof()) {

    str = str + ' ';

   // Extracting word by word from stream
    ss >> wrd;

    // Concatenating in the string to be returned
    str = str + wrd;
  }

  libblis_read_inpops(str, params, ops, api, pfr);
}
