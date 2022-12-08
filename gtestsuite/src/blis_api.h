#ifndef BLIS_API_H
#define BLIS_API_H

#include "blis_utils.h"
#include "blis_inpfile.h"

char* libblis_test_get_result
     (
       double          resid,
       const thresh_t* thresh,
       char*           dc_str,
       test_params_t*  params
     );

void fill_string_with_n_spaces( char* str, unsigned int n_spaces );

void libblis_test_build_function_string
     (
       char*        prefix_str,
       opid_t       opid,
       ind_t        method,
       char*        ind_str,
       const char*  op_str,
       unsigned int is_mixed_dt,
       char*        dc_str,
       unsigned int n_param_combos,
       char*        pc_str,
       char*        sc_str,
       char*        funcname_str
     );

void libblis_test_build_dims_string(test_op_t* op, tensor_t* dim, char* dims_str);

static char* libblis_test_result( double resid, const thresh_t* thresh,
                                char* dc_str, test_params_t* params )  {
    char* r_val;
    return r_val = libblis_test_get_result ( resid, thresh, dc_str, params );
}

static void libblis_build_function_string( test_params_t* params,
  opid_t opid, const char *op_str, ind_t method, unsigned int dci,
  unsigned int pci, unsigned int sci, char* fucnptr ) {

    char* ind_str = NULL;
    char* str     = NULL;
    if( params->api == API_CBLAS )
      str = (char*)CBLAS_FILEDATA_PREFIX_STR;
    else if( params->api == API_BLAS )
      str = (char*)BLAS_FILEDATA_PREFIX_STR;
    else
      str = (char*)BLIS_FILEDATA_PREFIX_STR;

    if ( method != BLIS_NAT ) {
        ind_str = bli_ind_get_impl_string( method );
    }

    // Build a string unique to the operation, datatype combo,
    // parameter combo, and storage combo being tested.
    libblis_test_build_function_string( str,
         opid, method, ind_str, op_str, params->is_mixed_dt,
         params->dc_str[dci], params->n_param_combos, params->pc_str[pci],
         params->sc_str[sci], fucnptr );
}

static void displayProps( const char* fucnptr, test_params_t* prms, test_op_t* op,
    tensor_t* dim, double& resid, char *ps, printres_t *ptr)
{
    char blank_str[32];
    char dims_str[64];
    string sas = ps ;
    string sfs = BLIS_TEST_FAIL_STRING ;
    string sos = BLIS_TEST_OVERFLOW_STRING;
    string sus = BLIS_TEST_UNDERFLOW_STRING;
    string sps = BLIS_TEST_PASS_STRING;
    string sws = BLIS_TEST_WARN_STRING;

    // Compute the number of spaces we have left to fill given
    // length of our operation's name.
    unsigned int  n_spaces = MAX_FUNC_STRING_LENGTH - strlen( fucnptr );
    fill_string_with_n_spaces( blank_str, n_spaces );

    // Print all dimensions to a single string.
    libblis_test_build_dims_string( op, dim, dims_str );

/*   if(( prms->passflag && ( strcmp(ps, BLIS_TEST_FAIL_STRING) != 0 )) ||
       ( prms->oruflw && (( strcmp(ps, BLIS_TEST_OVERFLOW_STRING) != 0 ) ||
       ( strcmp(ps, BLIS_TEST_UNDERFLOW_STRING) != 0 ))))*/
   if( prms->passflag && (( sas == sps) || ( sas == sws)) )
    {
        fprintf( stdout,
                 "%s%s      %s  %8.2le   %s\n",
                 fucnptr, blank_str,
                 dims_str, resid,
                 ps );
    }

/*    if(( strcmp(ps, BLIS_TEST_FAIL_STRING) == 0 ) ||
       ( prms->oruflw && (( strcmp(ps, BLIS_TEST_OVERFLOW_STRING) == 0 ) ||
       ( strcmp(ps, BLIS_TEST_UNDERFLOW_STRING) == 0 ))))*/
    if(( sas == sfs) || ( prms->oruflw && (( sas == sos) || ( sas == sus))))
    {
        fprintf( stdout,
                 "%s%s      %s  %8.2le   %s\n",
                 fucnptr, blank_str,
                 dims_str, resid,
                 ps );

        ptr->cntf++;
    }
}

double libblis_test_op_randv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_randm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_addv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_amaxv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_axpbyv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_axpyv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_copyv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_dotv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_dotxv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_normfv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_scal2v
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_scalv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_setv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_xpbyv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_subv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_axpy2v
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_dotaxpyv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_axpyf
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_dotxf
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_dotxaxpyf
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_addm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_axpym
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_copym
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_normfm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_scal2m
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_scalm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_setm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_subm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim
     );

double libblis_test_op_xpbym
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_gemv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_ger
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_hemv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_her
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_her2
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_symv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_syr
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_syr2
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_trmv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_trsv
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_gemm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemmt
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_hemm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_herk
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_her2k
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_symm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_syrk
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_syr2k
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_trmm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_trmm3
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_trsm
     (
       test_params_t* params,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha
     );

double libblis_test_op_gemm_u8s8s32os32
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_u8s8s32os8
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_f32f32f32of32
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_u8s8s16os8
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_u8s8s16os16
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_bf16bf16f32obf16
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

double libblis_test_op_gemm_bf16bf16f32of32
     (
       test_params_t* params,
       char*          pc_str,
       char*          sc_str,
       tensor_t*      dim,
       atom_t         alpha,
       atom_t         beta
     );

#endif  // BLIS_API_H

