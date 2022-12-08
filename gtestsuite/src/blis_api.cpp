#include "blis_utils.h"
#include "blis_inpfile.h"
#include "blis_api.h"

/*****************************Utility Operations******************************/
void* libblis_test_randv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;
  unsigned int pci = 0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        if ( tdata->xc % tdata->nt != tdata->id ) {
          tdata->xc++;
          continue;
        }
        resid = libblis_test_op_randv( params, iface, params->dc_str[dci],
                                            params->sc_str[sci], &m);

        pfr->tcnt++;
        char* res_str = libblis_test_result (resid, thresh,
                              params->dc_str[dci], params );

        char buffer[125];
        libblis_build_function_string(params, op->opid, op_str,
                                         mt, dci, pci, sci, buffer);

        displayProps(buffer, params, op, &m, resid, res_str, pfr);

        tdata->xc += 1;
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_randm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;
  unsigned int pci = 0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        if ( tdata->xc % tdata->nt != tdata->id ) {
          tdata->xc++;
          continue;
        }
        resid = libblis_test_op_randm( params, iface, params->dc_str[dci],
                                            params->sc_str[sci], &mn);

        pfr->tcnt++;
        char* res_str = libblis_test_result (resid, thresh,
                              params->dc_str[dci], params );

        char buffer[125];
        libblis_build_function_string(params, op->opid, op_str, mt,
                                              dci, pci, sci, buffer);

        displayProps(buffer, params, op, &mn, resid, res_str, pfr);

        tdata->xc += 1;
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Utility Operations *****************************/
/*                                                                            */
/*****************************Level-1V Operations******************************/
void* libblis_test_addv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_addv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m);

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str, mt,
                                                dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_amaxv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_amaxv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str, mt,
                                                dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_axpbyv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_axpbyv( params, iface, params->dc_str[dci],
                     params->pc_str[pci], params->sc_str[sci], &m,
                        params->alpha[a], params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_axpyv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_axpyv( params, iface, params->dc_str[dci],
            params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_copyv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_copyv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str, mt,
                                                dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_dotv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_dotv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str, mt,
                                                dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_dotxv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_dotxv( params, iface, params->dc_str[dci],
                     params->pc_str[pci], params->sc_str[sci], &m,
                        params->alpha[a], params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_normfv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_normfv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_scal2v_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_scal2v( params, iface, params->dc_str[dci],
             params->pc_str[pci], params->sc_str[sci], &m,  params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_scalv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_scalv( params, iface, params->dc_str[dci],
             params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_setv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
         // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_setv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_subv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_subv( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &m );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &m, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Level-1V Operations *****************************/
/*                                                                            */
/*****************************Level-1F Operations******************************/
void* libblis_test_xpbyv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_xpbyv( params, iface, params->dc_str[dci],
             params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }


  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_axpy2v_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_axpy2v( params, iface,
                    params->dc_str[dci], params->pc_str[pci],
                    params->sc_str[sci], &m, params->alpha[a],
                                              params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_dotaxpyv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
           // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_dotaxpyv( params, iface,
                  params->dc_str[dci], params->pc_str[pci],
                  params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_axpyf_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_axpyf( params, op, iface,
                  params->dc_str[dci], params->pc_str[pci],
                  params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_dotxf_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
             // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_dotxf( params, op, iface,
                    params->dc_str[dci], params->pc_str[pci],
                    params->sc_str[sci], &m, params->alpha[a],
                                                params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_dotxaxpyf_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
             // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_dotxaxpyf( params, op, iface,
                    params->dc_str[dci], params->pc_str[pci],
                    params->sc_str[sci], &m, params->alpha[a],
                                                params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Level-1F Operations *****************************/
/*                                                                            */
/*****************************Level-1M Operations******************************/
void* libblis_test_addm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_addm( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &mn );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &mn, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_axpym_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_axpym( params, iface, params->dc_str[dci],
                   params->pc_str[pci], params->sc_str[sci], &mn,
                               params->alpha[a] );

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &mn, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_copym_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_copym( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &mn );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &mn, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_normfm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_normfm( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &mn );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &mn, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_scal2m_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_scal2m( params, iface, params->dc_str[dci],
              params->pc_str[pci], params->sc_str[sci], &mn,  params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &mn, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_scalm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_scalm( params, iface, params->dc_str[dci],
              params->pc_str[pci], params->sc_str[sci], &mn, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &mn, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_setm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_setm( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &mn );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &mn, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_subm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          if ( tdata->xc % tdata->nt != tdata->id ) {
            tdata->xc++;
            continue;
          }
          resid = libblis_test_op_subm( params, iface, params->dc_str[dci],
                 params->pc_str[pci], params->sc_str[sci], &mn );

          pfr->tcnt++;
          char* res_str = libblis_test_result (resid, thresh,
                                params->dc_str[dci], params );

          char buffer[125];
          libblis_build_function_string(params, op->opid, op_str,
                                                mt, dci, pci, sci, buffer);

          displayProps(buffer, params, op, &mn, resid, res_str, pfr);

          tdata->xc += 1;
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_xpbym_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_xpbym( params, iface, params->dc_str[dci],
             params->pc_str[pci], params->sc_str[sci], &mn, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &mn, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Level-1M Operations *****************************/
/*                                                                            */
/*****************************Level-2 Operations*******************************/
void* libblis_test_gemv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_gemv( params, iface, params->dc_str[dci],
                      params->pc_str[pci], params->sc_str[sci], &mn,
                      params->alpha[a], params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str, mt,
                                                        dci, pci, sci, buffer);

              displayProps(buffer, params, op, &mn, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_ger_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_ger( params, iface, params->dc_str[dci],
              params->pc_str[pci], params->sc_str[sci], &mn, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &mn, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_hemv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_hemv( params, iface, params->dc_str[dci],
                      params->pc_str[pci], params->sc_str[sci], &m,
                      params->alpha[a], params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_her_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_her( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_her2_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_her2( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_symv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            // Loop over the beta values.
            for ( unsigned int b = 0; b < params->nab; ++b ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_symv( params, iface, params->dc_str[dci],
                      params->pc_str[pci], params->sc_str[sci], &m,
                      params->alpha[a], params->beta[b]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &m, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_syr_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_syr( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_syr2_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_syr2( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_trmv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_trmv( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_trsv_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t* op         = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t m;
  double resid = 0.0;

  if( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    m = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        // Loop over the requested parameter combinations.
        for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
          // Loop over the alpha values.
          for ( unsigned int a = 0; a < params->nab; ++a ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_trsv( params, iface, params->dc_str[dci],
                    params->pc_str[pci], params->sc_str[sci], &m, params->alpha[a]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[dci], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str,
                                                  mt, dci, pci, sci, buffer);

            displayProps(buffer, params, op, &m, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Level-2 Operations *****************************/
/*                                                                           */
/*****************************Level-3 Operations******************************/
void* libblis_test_gemm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_gemm( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mnk,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemmt_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t  nk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    nk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_gemmt( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &nk,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str,
                                                      mt, dci, pci, sci, buffer);

                displayProps(buffer, params, op, &nk, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_hemm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_hemm( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mn,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mn, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_herk_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t  nk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    nk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_herk( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &nk,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                        dci, pci, sci, buffer);

                displayProps(buffer, params, op, &nk, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_her2k_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_her2k( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mn,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mn, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_symm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_symm( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mn,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mn, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_syrk_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t  nk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    nk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_syrk( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &nk,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                        dci, pci, sci, buffer);

                displayProps(buffer, params, op, &nk, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_syr2k_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_syr2k( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mn,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mn, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_trmm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_trmm( params, iface, params->dc_str[dci],
                      params->pc_str[pci], params->sc_str[sci], &mn,
                      params->alpha[a]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &mn, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_trmm3_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              // Loop over the beta values.
              for ( unsigned int b = 0; b < params->nab; ++b ) {
                if ( tdata->xc % tdata->nt != tdata->id ) {
                  tdata->xc++;
                  continue;
                }
                resid = libblis_test_op_trmm3( params, iface, params->dc_str[dci],
                        params->pc_str[pci], params->sc_str[sci], &mn,
                        params->alpha[a], params->beta[b]);

                pfr->tcnt++;
                char* res_str = libblis_test_result (resid, thresh,
                                      params->dc_str[dci], params );

                char buffer[125];
                libblis_build_function_string(params, op->opid, op_str, mt,
                                                      dci, pci, sci, buffer);

                displayProps(buffer, params, op, &mn, resid, res_str, pfr);

                tdata->xc += 1;
              }
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_trsm_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  iface_t iface         = tdata->iface;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mn;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  tensor_t *dim = params->dim;
  unsigned int ndim = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mn = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested datatypes.
      for ( unsigned int dci = 0; dci < params->n_dt_combos; ++dci ) {
        unsigned int n = params->indn[dci];
        // Loop over induced methods (or just BLIS_NAT).
        for( unsigned int ind = 0 ; ind < n ; ++ind) {
          mt = ind_enable_get_str(params, dci, ind, op);
          // Loop over the requested parameter combinations.
          for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
            // Loop over the alpha values.
            for ( unsigned int a = 0; a < params->nab; ++a ) {
              if ( tdata->xc % tdata->nt != tdata->id ) {
                tdata->xc++;
                continue;
              }
              resid = libblis_test_op_trsm( params, iface, params->dc_str[dci],
                      params->pc_str[pci], params->sc_str[sci], &mn,
                      params->alpha[a]);

              pfr->tcnt++;
              char* res_str = libblis_test_result (resid, thresh,
                                    params->dc_str[dci], params );

              char buffer[125];
              libblis_build_function_string(params, op->opid, op_str,
                                                    mt, dci, pci, sci, buffer);

              displayProps(buffer, params, op, &mn, resid, res_str, pfr);

              tdata->xc += 1;
            }
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}
/********************* End Of Level-3 Operations *****************************/
/*                                                                           */
/*****************************LPGEMM Operations ******************************/
void* libblis_test_gemm_u8s8s32os32_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_u8s8s32os32( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_u8s8s32os8_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_u8s8s32os8( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_f32f32f32of32_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_f32f32f32of32( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                                 params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                            0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_u8s8s16os8_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_u8s8s16os8( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_u8s8s16os16_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_u8s8s16os16( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_bf16bf16f32obf16_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_bf16bf16f32obf16( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

void* libblis_test_gemm_bf16bf16f32of32_thread_entry( void* tdata_void ) {
  thread_data_t* tdata  = (thread_data_t*)tdata_void;
  test_params_t* params = tdata->params;
  test_op_t*     op     = tdata->op;
  const char* op_str    = tdata->str;
  printres_t* pfr       = tdata->pfr;
  ind_t mt              = BLIS_NAT;
  char label_str[128];
  tensor_t mnk;
  double resid = 0.0;

  if ( tdata->id == 0 ) {
    libblis_test_build_col_labels_string( params, op, label_str );
    libblis_test_fprintf( stdout, "\n%s\n", label_str );
  }

  params->dc_str[0][0] = 's';
  tensor_t *dim        = params->dim;
  unsigned int ndim    = params->ndim;

  // Loop over the requested problem sizes.
  for(unsigned int i = 0 ; i < ndim ; i++) {
    mnk = dim[i];
    // Loop over the requested storage schemes.
    for ( unsigned int sci = 0; sci < params->n_store_combos; ++sci ) {
      // Loop over the requested parameter combinations.
      for ( unsigned int pci = 0; pci < params->n_param_combos; ++pci ) {
        // Loop over the alpha values.
        for ( unsigned int a = 0; a < params->nab; ++a ) {
          // Loop over the beta values.
          for ( unsigned int b = 0; b < params->nab; ++b ) {
            if ( tdata->xc % tdata->nt != tdata->id ) {
              tdata->xc++;
              continue;
            }
            resid = libblis_test_op_gemm_bf16bf16f32of32( params, params->pc_str[pci],
                params->sc_str[sci], &mnk,  params->alpha[a], params->beta[b]);

            pfr->tcnt++;
            char* res_str = libblis_test_result (resid, thresh,
                                  params->dc_str[0], params );

            char buffer[125];
            libblis_build_function_string(params, op->opid, op_str, mt,
                                                  0, pci, sci, buffer);

            displayProps(buffer, params, op, &mnk, resid, res_str, pfr);

            tdata->xc += 1;
          }
        }
      }
    }
  }

  // Wait for all other threads so that the output stays organized.
  bli_pthread_barrier_wait( tdata->barrier );

  return 0;
}

/*************************** End Of Operations *******************************/
/*                           END OF OPERATIONS                               */
/*****************************************************************************/