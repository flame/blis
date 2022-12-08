#ifndef BLIS_TEST_H
#define BLIS_TEST_H

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cstdio>

#include "blis.h"
#include "gtest/gtest.h"

using namespace std;

// --- System headers ---------------------------------------------------------
// For va_* functions.
#include <stdarg.h>
// For string manipulation functions.
#include <string.h>
// For other string manipulation functions (e.g. isspace()).
#include <ctype.h>

// For POSIX stuff.
#ifndef _MSC_VER
#include <unistd.h>
#endif

// --- Constants and types ----------------------------------------------------
#define PARAMETERS_FILENAME          "input.general"
#define OPERATIONS_FILENAME          "input.operations"
#define ALPHA_BETA_FILENAME          "alphabeta.dat"
#define INPUT_COMMENT_CHAR           '#'
#define OUTPUT_COMMENT_CHAR          '%'

#define BLIS_FILE_PREFIX_STR         "libblis_test"
#define BLIS_FILEDATA_PREFIX_STR     "blis"
#define BLAS_FILEDATA_PREFIX_STR     "blas"
#define CBLAS_FILEDATA_PREFIX_STR    "cblas"

#define INPUT_BUFFER_SIZE            256
#define MAX_FILENAME_LENGTH          1000
#define MAX_BINARY_NAME_LENGTH       256
#define MAX_FUNC_STRING_LENGTH       36
#define FLOPS_PER_UNIT_PERF          1e9

#define MAX_NUM_MSTORAGE             4
#define MAX_NUM_VSTORAGE             5
#define MAX_NUM_DATATYPES            4
#define MAX_NUM_PARAMETERS           7
#define MAX_NUM_DIMENSIONS           3
#define MAX_NUM_OPERANDS             5

#define MAX_PASS_STRING_LENGTH       32
#define BLIS_TEST_FAIL_STRING        "FAILURE"
#define BLIS_TEST_WARN_STRING        "MARGINAL"
#define BLIS_TEST_PASS_STRING        "PASS"
#define BLIS_TEST_OVERFLOW_STRING    "OVERFLOW"
#define BLIS_TEST_UNDERFLOW_STRING   "UNDERFLOW"

#define ON_FAILURE_IGNORE_CHAR       'i'
#define ON_FAILURE_SLEEP_CHAR        's'
#define ON_FAILURE_ABORT_CHAR        'a'

#define SECONDS_TO_SLEEP             3

#define DISABLE                      0
#define ENABLE                       1
#define ENABLE_ONLY                  2

#define MAX_PARAM_VALS_PER_TYPE      4
#define BLIS_TEST_PARAM_SIDE_CHARS   "lr"
#define BLIS_TEST_PARAM_UPLO_CHARS   "lu"
#define BLIS_TEST_PARAM_UPLODE_CHARS "dlu"
#define BLIS_TEST_PARAM_TRANS_CHARS  "ncth"
#define BLIS_TEST_PARAM_CONJ_CHARS   "nc"
#define BLIS_TEST_PARAM_DIAG_CHARS   "nu"

#define BLIS_INIT_SUCCESS            0
#define BLIS_INIT_FAILURE           -1
#define NUM_PARAM_TYPES              6
#define MAX_NUM_ABVALUES             5

/*Allocating buffers with malloc in gtestsuite */
#define __GTESTSUITE_MALLOC_BUFFER__

typedef enum
{
    BLIS_TEST_PARAM_SIDE      = 0,
    BLIS_TEST_PARAM_UPLO      = 1,
    BLIS_TEST_PARAM_UPLODE    = 2,
    BLIS_TEST_PARAM_TRANS     = 3,
    BLIS_TEST_PARAM_CONJ      = 4,
    BLIS_TEST_PARAM_DIAG      = 5,
} param_t;

#define MAX_STORE_VALS_PER_TYPE      4
#define BLIS_TEST_MSTORE_CHARS       "crg"
#define BLIS_TEST_VSTORE_CHARS       "crji"

#define NUM_OPERAND_TYPES           2
typedef enum
{
    BLIS_TEST_MATRIX_OPERAND  = 0,
    BLIS_TEST_VECTOR_OPERAND  = 1
} operand_t;

typedef enum
{
    API_BLIS  = 0,
    API_CBLAS = 1,
    API_BLAS  = 2
} api_t;

typedef enum
{
    BLIS_DEFAULT    = 0,
    BLIS_OVERFLOW   = 1,
    BLIS_UNDERFLOW  = 2
} vflg_t;

typedef enum
{
    BLIS_TEST_DIMS_MNK        = 0,
    BLIS_TEST_DIMS_MN         = 1,
    BLIS_TEST_DIMS_MK         = 2,
    BLIS_TEST_DIMS_M          = 3,
    BLIS_TEST_DIMS_MF         = 4,
    BLIS_TEST_DIMS_K          = 5,
    BLIS_TEST_NO_DIMS         = 6
} dimset_t;

typedef enum
{
    BLIS_TEST_SEQ_UKERNEL     = 0,
    BLIS_TEST_SEQ_FRONT_END   = 1,
    BLIS_TEST_MT_FRONT_END    = 2
} iface_t;


typedef enum
{
    BLIS_TEST_RAND_REAL_VALUES = 0,
    BLIS_TEST_RAND_NARROW_POW2 = 1
} rand_t;

typedef struct
{
    double failwarn;
    double warnpass;
} thresh_t;

const thresh_t thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                             { 1e-04, 1e-05 },   // warn, pass for c
                                             { 1e-13, 1e-14 },   // warn, pass for d
                                             { 1e-13, 1e-14 } }; // warn, pass for z

#define SIZE 4
typedef dcomplex atom_t;

typedef struct
{
    dim_t m;
    dim_t n;
    dim_t k;
} tensor_t;

typedef struct
{
    unsigned int  n_repeats;
    unsigned int  n_mstorage;
    unsigned int  n_vstorage;
    char          storage[ NUM_OPERAND_TYPES ][ MAX_NUM_MSTORAGE + 1 ];
    unsigned int  mix_all_storage;
    unsigned int  alignment;
    unsigned int  rand_method;
    unsigned int  gs_spacing;
    unsigned int  n_datatypes;
    char          datatype_char[ MAX_NUM_DATATYPES + 1 ];
    num_t         datatype[ MAX_NUM_DATATYPES + 1 ];
    unsigned int  mixed_domain;
    unsigned int  mixed_precision;
    unsigned int  p_first;
    unsigned int  p_max;
    unsigned int  p_inc;
    unsigned int  ind_enable[ BLIS_NUM_IND_METHODS ];
    unsigned int  n_app_threads;
    char          reaction_to_failure;
    unsigned int  output_matlab_format;
    unsigned int  output_files;
    unsigned int  error_checking_level;
    unsigned int  is_mixed_dt;
    unsigned int  n_param_combos;
    unsigned int  n_store_combos;
    unsigned int  n_dt_combos;
    char**        pc_str;
    char**        sc_str;
    char**        dc_str;
    unsigned int  indim[MAX_NUM_DATATYPES][BLIS_NAT+1];
    unsigned int  indn[MAX_NUM_DATATYPES];
    num_t         dt[MAX_NUM_DATATYPES];
    bool          initparams;

    api_t         api;
    unsigned int  abf;
    atom_t       *alpha;
    atom_t       *beta;
    unsigned int  nab;
    unsigned int  ldf;
    unsigned int  ld[3];
    unsigned int  bitextf;
    unsigned int  dimf;
    unsigned int  ndim;
    unsigned int  nanf;
    tensor_t      *dim;

    unsigned int  passflag;
    vflg_t        oruflw;
    unsigned int  bitrp;

    char          op_t;

} test_params_t;

typedef struct
{
    char libblis_test_parameters_filename[ MAX_FILENAME_LENGTH + 1 ];
    char libblis_test_operations_filename[ MAX_FILENAME_LENGTH + 1 ];
    char libblis_test_alphabeta_parameter[ MAX_FILENAME_LENGTH + 1 ];
} blis_string_t;

typedef struct
{
    // parent test_ops_t struct
    struct test_ops_s*   ops;

    opid_t        opid;
    dimset_t      dimset;

    int           op_switch;
    unsigned int  n_dims;

    int           dim_spec[ MAX_NUM_DIMENSIONS ];
    int           dim_aux[ MAX_NUM_DIMENSIONS ];
    unsigned int  n_params;
    char          params[ MAX_NUM_PARAMETERS ];
    bool          test_done;
} test_op_t;

typedef struct test_ops_s
{
    // individual override
    int       indiv_over;

    // section overrides
    int       util_over;
    int       l1v_over;
    int       l1m_over;
    int       l1f_over;
    int       l2_over;
    int       l3ukr_over;
    int       l3_over;

    // util
    test_op_t randv;
    test_op_t randm;

    // level-1v
    test_op_t addv;
    test_op_t amaxv;
    test_op_t axpbyv;
    test_op_t axpyv;
    test_op_t copyv;
    test_op_t dotv;
    test_op_t dotxv;
    test_op_t normfv;
    test_op_t scalv;
    test_op_t scal2v;
    test_op_t setv;
    test_op_t subv;
    test_op_t xpbyv;

    // level-1m
    test_op_t addm;
    test_op_t axpym;
    test_op_t copym;
    test_op_t normfm;
    test_op_t scalm;
    test_op_t scal2m;
    test_op_t setm;
    test_op_t subm;
    test_op_t xpbym;

    // level-1f
    test_op_t axpy2v;
    test_op_t dotaxpyv;
    test_op_t axpyf;
    test_op_t dotxf;
    test_op_t dotxaxpyf;

    // level-2
    test_op_t gemv;
    test_op_t ger;
    test_op_t hemv;
    test_op_t her;
    test_op_t her2;
    test_op_t symv;
    test_op_t syr;
    test_op_t syr2;
    test_op_t trmv;
    test_op_t trsv;

    // level-3 micro-kernels
    test_op_t gemm_ukr;
    test_op_t trsm_ukr;
    test_op_t gemmtrsm_ukr;

    // level-3
    test_op_t gemm;
    test_op_t gemmt;
    test_op_t hemm;
    test_op_t herk;
    test_op_t her2k;
    test_op_t symm;
    test_op_t syrk;
    test_op_t syr2k;
    test_op_t trmm;
    test_op_t trmm3;
    test_op_t trsm;

    test_op_t gemm_u8s8s32os32;
    test_op_t gemm_u8s8s32os8;
    test_op_t gemm_f32f32f32of32;
    test_op_t gemm_u8s8s16os16;
    test_op_t gemm_u8s8s16os8;
    test_op_t gemm_bf16bf16f32of32;
    test_op_t gemm_bf16bf16f32obf16;

} test_ops_t;

typedef struct
{
    uint32_t tcnt;
    uint32_t cntf;
} printres_t;

typedef struct
{
    test_params_t*         params;
    test_op_t*             op;
    const char*            str;
    unsigned int           nt;
    unsigned int           id;
    iface_t                iface;
    unsigned int           xc;
    bli_pthread_barrier_t* barrier;
    printres_t*            pfr;
} thread_data_t;

typedef struct
{
    char inputfile[ MAX_FILENAME_LENGTH ];
    int  fileread;
} input_file_t;

typedef struct
{
    test_params_t *params;
    test_ops_t    *ops;
    input_file_t  *pfile;
    bli_pthread_t *pthread;
    thread_data_t *tdata;
} input_data_t;

void* libblis_test_randv_thread_entry( void* tdata_void );
void* libblis_test_randm_thread_entry( void* tdata_void );

void* libblis_test_addv_thread_entry( void* tdata_void );
void* libblis_test_amaxv_thread_entry( void* tdata_void );
void* libblis_test_axpbyv_thread_entry( void* tdata_void );
void* libblis_test_axpyv_thread_entry( void* tdata_void );
void* libblis_test_copyv_thread_entry( void* tdata_void );
void* libblis_test_dotv_thread_entry( void* tdata_void );
void* libblis_test_dotxv_thread_entry( void* tdata_void );
void* libblis_test_normfv_thread_entry( void* tdata_void );
void* libblis_test_scal2v_thread_entry( void* tdata_void );
void* libblis_test_scalv_thread_entry( void* tdata_void );
void* libblis_test_setv_thread_entry( void* tdata_void );
void* libblis_test_subv_thread_entry( void* tdata_void );

void* libblis_test_xpbyv_thread_entry( void* tdata_void );
void* libblis_test_axpy2v_thread_entry( void* tdata_void );
void* libblis_test_dotaxpyv_thread_entry( void* tdata_void );
void* libblis_test_axpyf_thread_entry( void* tdata_void );
void* libblis_test_dotxf_thread_entry( void* tdata_void );
void* libblis_test_dotxaxpyf_thread_entry( void* tdata_void );

void* libblis_test_addm_thread_entry( void* tdata_void );
void* libblis_test_axpym_thread_entry( void* tdata_void );
void* libblis_test_copym_thread_entry( void* tdata_void );
void* libblis_test_normfm_thread_entry( void* tdata_void );
void* libblis_test_scal2m_thread_entry( void* tdata_void );
void* libblis_test_scalm_thread_entry( void* tdata_void );
void* libblis_test_setm_thread_entry( void* tdata_void );
void* libblis_test_subm_thread_entry( void* tdata_void );
void* libblis_test_xpbym_thread_entry( void* tdata_void );

void* libblis_test_gemv_thread_entry( void* tdata_void );
void* libblis_test_ger_thread_entry( void* tdata_void );
void* libblis_test_hemv_thread_entry( void* tdata_void );
void* libblis_test_her_thread_entry( void* tdata_void );
void* libblis_test_her2_thread_entry( void* tdata_void );
void* libblis_test_symv_thread_entry( void* tdata_void );
void* libblis_test_syr_thread_entry( void* tdata_void );
void* libblis_test_syr2_thread_entry( void* tdata_void );
void* libblis_test_trmv_thread_entry( void* tdata_void );
void* libblis_test_trsv_thread_entry( void* tdata_void );

void* libblis_test_gemm_thread_entry( void* tdata_void );
void* libblis_test_gemmt_thread_entry( void* tdata_void );
void* libblis_test_hemm_thread_entry( void* tdata_void );
void* libblis_test_herk_thread_entry( void* tdata_void );
void* libblis_test_her2k_thread_entry( void* tdata_void );
void* libblis_test_symm_thread_entry( void* tdata_void );
void* libblis_test_syrk_thread_entry( void* tdata_void );
void* libblis_test_syr2k_thread_entry( void* tdata_void );
void* libblis_test_trmm_thread_entry( void* tdata_void );
void* libblis_test_trmm3_thread_entry( void* tdata_void );
void* libblis_test_trsm_thread_entry( void* tdata_void );

void* libblis_test_gemm_u8s8s32os32_thread_entry( void* tdata_void );
void* libblis_test_gemm_f32f32f32of32_thread_entry( void* tdata_void );
void* libblis_test_gemm_u8s8s16os8_thread_entry( void* tdata_void );
void* libblis_test_gemm_u8s8s32os8_thread_entry( void* tdata_void );
void* libblis_test_gemm_u8s8s16os16_thread_entry( void* tdata_void );
void* libblis_test_gemm_bf16bf16f32of32_thread_entry( void* tdata_void );
void* libblis_test_gemm_bf16bf16f32obf16_thread_entry( void* tdata_void );

/*
 * The derived class for Blis Test Suite
 * where all the data members and member functions are
 * declared and defined
 */
class AoclBlisTestFixture : public ::testing::TestWithParam<input_data_t>
{
    public:
    void SetUp() override
    {
        params    = GetParam().params;
        ops       = GetParam().ops;
        pthread   = GetParam().pthread;
        tdata     = GetParam().tdata;
        pfile     = GetParam().pfile;

        if(pfile->fileread != 1)
        {
            nt      = ( unsigned int )params->n_app_threads;
            barrier =
            (bli_pthread_barrier_t*)bli_malloc_user( sizeof( bli_pthread_barrier_t ) );
            bli_pthread_barrier_init( barrier, NULL, nt );
        }
        pfr     = (printres_t*)bli_malloc_user( sizeof( printres_t ) );
        memset(pfr, 0, sizeof(printres_t));
    }

    void TearDown() override
    {
        if(pfile->fileread != 1)
        {
            bli_pthread_barrier_destroy( barrier );
            bli_free_user( barrier );
        }
        bli_free_user( pfr );
    }

    bool libblis_test_preprocess_params( test_params_t* params, test_op_t* op,
                      iface_t iface, const char* p_types, const char* o_types);

    bool create_params(test_params_t *params);

    bool destroy_params(test_params_t *params);

    int libblis_test_read_params_inpfile( char* filename, test_params_t* params,
                                             test_ops_t* ops, printres_t* pfr);
    protected:
      unsigned int     nt;
      input_data_t*    inData;
      test_params_t*   params;
      test_ops_t*      ops;
      tensor_t*        dim;
      bli_pthread_t*   pthread;
      thread_data_t*   tdata;
      printres_t*      pfr;
      input_file_t*    pfile;
      bli_pthread_barrier_t* barrier;
};

class BlisTestSuite
{
    private:
        blis_string_t  blis_string;
        input_file_t   pfile;
        test_params_t  params;
        test_ops_t     ops;
    public:
        ~BlisTestSuite( );
        test_params_t* getParamsStr() { return &(this->params); }
        blis_string_t* getStgStr() { return &(this->blis_string); }
        test_ops_t* getOpsStr() { return &(this->ops); }
        input_file_t* getfileStr() { return &(this->pfile); }

        int libblis_test_init_strings(blis_string_t *test_data);
        int libblis_test_inpfile( char* input_filename, input_file_t* pfile);
        int libblis_test_read_params_file( char* input_filename,
                                           test_params_t* params, char *abpf);
        int libblis_test_read_ops_file( char* input_filename, test_ops_t* ops );
        void CreateGtestFilters(test_ops_t* ops, string& str);
        void CreateGtestFilters_api(input_file_t* pfile, string& str);
};
#endif  // BLIS_TEST_H
