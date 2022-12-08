#include "blis_test.h"
#include "gtest_pthread.h"

extern vector<input_data_t>	inputData;

using namespace std;

/*****************************Utility Operations******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_RANDV )
{
    unsigned int id = 0;
    const char* op_str    = "randv";
    const char* o_types   = "v";    // x
    const char* p_types   = "";     // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->randv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_randv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for randv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_RANDM )
{
    unsigned int id = 0;
    const char* op_str    = "randm";
    const char* o_types   = "m";    // a
    const char* p_types   = "";     // transa transb
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->randm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_randm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for randm : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}
/********************* End Of Utility Operations *****************************/

/*****************************Level-1V Operations******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_ADDV )
{
    unsigned int id = 0;
    const char* op_str    = "addv";
    const char* o_types   = "vv";    // x y
    const char* p_types   = "c";     // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->addv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_addv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for addv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AMAXV )
{
    unsigned int id = 0;
    const char* op_str    = "amaxv";
    const char* o_types   = "v";    // x
    const char* p_types   = "";     // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->amaxv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_amaxv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for amaxv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AXPBYV )
{
    unsigned int id = 0;
    const char* op_str    = "axpbyv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->axpbyv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_axpbyv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for axpbyv : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AXPYV )
{
    unsigned int id = 0;
    const char* op_str    = "axpyv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->axpyv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_axpyv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for axpyv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_COPYV )
{
    unsigned int id = 0;
    const char* op_str    = "copyv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->copyv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_copyv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for copyv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_DOTV )
{
    unsigned int id = 0;
    const char* op_str    = "dotv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "cc";  // conjx conjy
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->dotv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_dotv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for dotv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }

TEST_P( AoclBlisTestFixture, AOCL_BLIS_DOTXV )
{
    unsigned int id = 0;
    const char* op_str    = "dotxv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "cc";  // conjx conjy
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->dotxv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_dotxv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for dotxv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_NORMFV )
{
    unsigned int id = 0;
    const char* op_str    = "normfv";
    const char* o_types   = "v";  // x
    const char* p_types   = "";   // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->normfv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_normfv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for normfv : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SCAL2V )
{
    unsigned int id = 0;
    const char* op_str    = "scal2v";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->scal2v);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_scal2v_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for scal2v : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
 }

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SCALV )
{
    unsigned int id = 0;
    const char* op_str    = "scalv";
    const char* o_types   = "v";   // x
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->scalv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_scalv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for scalv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SETV )
{
    unsigned int id = 0;
    const char* op_str    = "setv";
    const char* o_types   = "v";  // x
    const char* p_types   = "";   // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->setv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_setv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for setv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SUBV )
{
    unsigned int id = 0;
    const char* op_str    = "subv";
    const char* o_types   = "vv";    // x y
    const char* p_types   = "c";     // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->subv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_subv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for subv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}
  /********************* End Of Level-1V Operations *****************************/

  /*****************************Level-1F Operations******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_XPBYV )
{
    unsigned int id = 0;
    const char* op_str    = "xpbyv";
    const char* o_types   = "vv";  // x y
    const char* p_types   = "c";   // conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->xpbyv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_xpbyv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for xpbyv : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AXPY2V )
{
    unsigned int id = 0;
    const char* op_str    = "axpy2v";
    const char* o_types   = "vvv";  // x y z
    const char* p_types   = "cc";   // conjx conjy
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->axpy2v);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_axpy2v_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for axpy2v : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_DOTAXPYV )
{
    unsigned int id = 0;
    const char* op_str    = "dotaxpyv";
    const char* o_types   = "vvv";  // x y z
    const char* p_types   = "ccc";  // conjxt conjx conjy
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->dotaxpyv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_dotaxpyv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for dotaxpyv : %d\n", pfr->tcnt );
    printf( "Total test cases passed       : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed       : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AXPYF )
{
    unsigned int id = 0;
    const char* op_str    = "axpyf";
    const char* o_types   = "mvv";  // A x y
    const char* p_types   = "cc";   // conja conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->axpyf);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_axpyf_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for axpyf : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_DOTXF )
{
    unsigned int id = 0;
    const char* op_str    = "dotxf";
    const char* o_types   = "mvv";  // A x y
    const char* p_types   = "cc";   // conjat conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->dotxf);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_dotxf_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for dotxf : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_DOTXAXPYF )
{
    unsigned int id = 0;
    const char* op_str    = "dotxaxpyf";
    const char* o_types   = "mvvvv";  // A w x y z
    const char* p_types   = "cccc";   // conjat conja conjw conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->dotxaxpyf);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_dotxaxpyf_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for dotxaxpyf : %d\n", pfr->tcnt );
    printf( "Total test cases passed        : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed        : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }
  /********************* End Of Level-1F Operations *****************************/

  /*****************************Level-1M Operations******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_ADDM )
{
    unsigned int id = 0;
    const char* op_str    = "addm";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->addm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_addm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for addm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_AXPYM )
{
    unsigned int id = 0;
    const char* op_str    = "axpym";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->axpym);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_axpym_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for axpym : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }

TEST_P( AoclBlisTestFixture, AOCL_BLIS_COPYM )
{
    unsigned int id = 0;
    const char* op_str    = "copym";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->copym);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_copym_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for copym : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_NORMFM )
{
    unsigned int id = 0;
    const char* op_str    = "normfm";
    const char* o_types   = "m";  // x
    const char* p_types   = "";   // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->normfm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_normfm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for normfm : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SCAL2M )
{
    unsigned int id = 0;
    const char* op_str    = "scal2m";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->scal2m);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_scal2m_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for scal2m : %d\n", pfr->tcnt );
    printf( "Total test cases passed     : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed     : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SCALM )
{
    unsigned int id = 0;
    const char* op_str    = "scalm";
    const char* o_types   = "m";  // x
    const char* p_types   = "c";  // conjbeta
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->scalm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_scalm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for scalm : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SETM )
{
    unsigned int id = 0;
    const char* op_str    = "setm";
    const char* o_types   = "m";  // x
    const char* p_types   = "";   // (no parameters)
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->setm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_setm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for setm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SUBM )
{
    unsigned int id = 0;
    const char* op_str    = "subm";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->subm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_subm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for subm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_XPBYM )
{
    unsigned int id = 0;
    const char* op_str    = "xpbym";
    const char* o_types   = "mm";  // x y
    const char* p_types   = "h";   // transx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->xpbym);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_xpbym_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for xpbym : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
  }
  /********************* End Of Level-1M Operations *****************************/
  /*                                                                            */
  /*****************************Level-2 Operations*******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_GEMV )
{
    unsigned int id = 0;
    const char* op_str    = "gemv";
    const char* o_types   = "mvv";   // a x y
    const char* p_types   = "hc";    // transa conjx
    iface_t iface         = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op         = &(ops->gemv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_GER )
{
    unsigned int id;
    const char* op_str  = "ger";
    const char* o_types = "vvm"; // x y a
    const char* p_types = "cc";  // conjx conjy
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->ger);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_ger_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for ger : %d\n", pfr->tcnt );
    printf( "Total test cases passed  : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed  : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_HEMV )
{
    unsigned int id;
    const char* op_str  = "hemv";
    const char* o_types = "mvv";  // a x y
    const char* p_types = "ucc";  // uploa conja conjx
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->hemv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_hemv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for hemv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L2HER )
{
    unsigned int id;
    const char* op_str  = "her";
    const char* o_types = "vm";  // x a
    const char* p_types = "uc";  // uploa conjx
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->her);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_her_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for her : %d\n", pfr->tcnt );
    printf( "Total test cases passed  : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed  : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L2HER2 )
{
    unsigned int id;
    const char* op_str  = "her2";
    const char* o_types = "vvm";  // x y a
    const char* p_types = "ucc";  // uploa conjx conjy
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->her2);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_her2_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for her2 : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SYMV )
{
    unsigned int id;
    const char* op_str  = "symv";
    const char* o_types = "mvv";  // a x y
    const char* p_types = "ucc";  // uploa conja conjx
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->symv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_symv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for symv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L2SYR )
{
    unsigned int id;
    const char* op_str  = "syr";
    const char* o_types = "vm";  // x a
    const char* p_types = "uc";  // uploa conjx
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->syr);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_syr_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for syr : %d\n", pfr->tcnt );
    printf( "Total test cases passed  : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed  : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L2SYR2 )
{
    unsigned int id;
    const char* op_str  = "syr2";
    const char* o_types = "vvm";  // x y a
    const char* p_types = "ucc";  // uploa conjx conjy
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->syr2);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_syr2_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for syr2 : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_TRMV )
{
    unsigned int id;
    const char* op_str  = "trmv";
    const char* o_types = "mv";   // a x
    const char* p_types = "uhd";  // uploa transa diaga
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->trmv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_trmv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for trmv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_TRSV )
{
    unsigned int id;
    const char* op_str  = "trsv";
    const char* o_types = "mv";   // a x
    const char* p_types = "uhd";  // uploa transa diaga
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->trsv);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_trsv_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for trsv : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}
/********************* End Of Level-2 Operations *****************************/
/*                                                                           */
/*****************************Level-3 Operations******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_GEMM )
{
    unsigned int id;
    const char* op_str  = "gemm";
    const char* o_types = "mmm";  // a b c
    const char* p_types = "hh";   // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L3GEMMT )
{
    unsigned int id;
    const char* op_str  = "gemmt";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "uhh";   // uploc transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemmt);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemmt_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemmt: %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_HEMM )
{
    unsigned int id;
    const char* op_str  = "hemm";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "su";    // side uploa
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->hemm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_hemm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for hemm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_HERK )
{
    unsigned int id;
    const char* op_str  = "herk";
    const char* o_types = "mm";    // a c
    const char* p_types = "uc";    // uploc trans
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->herk);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_herk_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for herk : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_HER2K )
{
    unsigned int id;
    const char* op_str  = "her2k";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "uc";    // uploc trans
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->her2k);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_her2k_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for her2k : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SYMM )
{
    unsigned int id;
    const char* op_str  = "symm";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "su";    // side uploa
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->symm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_symm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for symm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SYRK )
{
    unsigned int id;
    const char* op_str  = "syrk";
    const char* o_types = "mm";    // a c
    const char* p_types = "uh";    // uploc trans
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->syrk);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_syrk_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for syrk : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_SYR2K )
{
    unsigned int id;
    const char* op_str  = "syr2k";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "uh";    // uploc trans
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->syr2k);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_syr2k_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for syr2k : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_L3TRMM )
{
    unsigned int id;
    const char* op_str  = "trmm";
    const char* o_types = "mm";   // a b
    const char* p_types = "suhd"; // side uploa transa diaga
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->trmm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_trmm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for trmm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_TRMM3 )
{
    unsigned int id;
    const char* op_str  = "trmm3";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "suhdh";  // side uploa transa diaga transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->trmm3);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_trmm3_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for trmm3 : %d\n", pfr->tcnt );
    printf( "Total test cases passed    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_BLIS_TRSM )
{
    unsigned int id;
    const char* op_str  = "trsm";
    const char* o_types = "mm";   // a b
    const char* p_types = "suhd"; // side uploa transa diaga
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->trsm);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_trsm_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for trsm : %d\n", pfr->tcnt );
    printf( "Total test cases passed   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}
/********************* End Of Level-3 Operations *****************************/
/*                                                                           */
/***************************** LPGEMM Operations *****************************/
TEST_P( AoclBlisTestFixture, AOCL_GEMM_S32S32 )
{
    unsigned int id;
    const char* op_str  = "gemm_u8s8s32os32";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "hh";     // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_u8s8s32os32);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_u8s8s32os32_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_u8s8s32os32 : %d\n", pfr->tcnt );
    printf( "Total test cases passed               : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed               : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_S8S32 )
{
    unsigned int id;
    const char* op_str  = "gemm_u8s8s32os8";
    const char* o_types = "mmm";   // a b c
    const char* p_types = "hh";    // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_u8s8s32os8);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_u8s8s32os8_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_u8s8s32os8   : %d\n", pfr->tcnt );
    printf( "Total test cases passed                : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed                : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_F32F32 )
{
    unsigned int id;
    const char* op_str  = "gemm_f32f32f32of32";
    const char* o_types = "mmm";  // a b c
    const char* p_types = "hh";   // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_f32f32f32of32);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_f32f32f32of32_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_f32f32f32of32 : %d\n", pfr->tcnt );
    printf( "Total test cases passed                 : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed                 : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_S8S16 )
{
    unsigned int id;
    const char* op_str  = "gemm_u8s8s16os8";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "hh";     // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_u8s8s16os8);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_u8s8s16os8_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_u8s8s16os8  : %d\n", pfr->tcnt );
    printf( "Total test cases passed               : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed               : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_S16S16 )
{
    unsigned int id;
    const char* op_str  = "gemm_u8s8s16os16";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "hh";     // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_u8s8s16os16);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_u8s8s16os16_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_u8s8s16os16  : %d\n", pfr->tcnt );
    printf( "Total test cases passed                : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed                : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_BF16BF16 )
{
    unsigned int id;
    const char* op_str  = "gemm_bf16bf16f32obf16";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "hh";     // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_bf16bf16f32obf16);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_bf16bf16f32obf16_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_bf16bf16f32obf16 : %d\n", pfr->tcnt );
    printf( "Total test cases passed                    : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed                    : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}

TEST_P( AoclBlisTestFixture, AOCL_GEMM_F32BF16 )
{
    unsigned int id;
    const char* op_str  = "gemm_bf16bf16f32of32";
    const char* o_types = "mmm";    // a b c
    const char* p_types = "hh";     // transa transb
    iface_t iface       = BLIS_TEST_SEQ_FRONT_END;
    test_op_t* op       = &(ops->gemm_bf16bf16f32of32);

    libblis_test_preprocess_params( params, op, iface, p_types, o_types );

    for( id = 0 ; id < nt ; id++ )
    {
        tdata[id].params  = params;
        tdata[id].op      = op;
        tdata[id].str     = op_str;
        tdata[id].nt      = nt;
        tdata[id].id      = id;
        tdata[id].iface   = iface;
        tdata[id].xc      = 0;
        tdata[id].barrier = barrier;
        tdata[id].pfr     = pfr;
        gtest_pthread_create( &pthread[id], NULL,
                              libblis_test_gemm_bf16bf16f32of32_thread_entry,
                              (void *)&tdata[id] );
    }

    // Thread 0 waits for additional threads to finish.
    for( id = 0 ; id < nt ; id++ )
    {
        gtest_pthread_join( pthread[id], NULL );
    }

    printf( "\n" );
    printf( "Total test cases for gemm_bf16bf16f32of32 : %d\n", pfr->tcnt );
    printf( "Total test cases passed                   : %d\n", (pfr->tcnt - pfr->cntf) );
    printf( "Total test cases failed                   : %d\n", pfr->cntf );

    AoclBlisTestFixture::destroy_params( params );
}
/********************* End Of Level-3 Operations *****************************/
/*                                                                           */
/********************* FILE_READ from InputFile ******************************/
TEST_P( AoclBlisTestFixture, AOCL_BLIS_READ_INPUTFILE )
{
    AoclBlisTestFixture::create_params( params );

    libblis_test_read_params_inpfile( pfile->inputfile, params, ops, pfr );

    AoclBlisTestFixture::destroy_params( params );
}
/******************************************************************************************/
/*                                   END OF OPERATIONS                                    */
/******************************************************************************************/


/******************************************************************************************/
/***								INSTANTIATE_TEST_SUITE_P							***/
/******************************************************************************************/
INSTANTIATE_TEST_SUITE_P( AoclBlisTests,
                          AoclBlisTestFixture,
                          ::testing::ValuesIn( inputData ) );
/******************************************************************************************/
