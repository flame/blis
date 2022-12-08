#include <cstdio>
#include <thread>

#include <iostream>
#include <cfenv>
#include <cmath>
#include <cerrno>

#include "blis_test.h"

vector<input_data_t>	inputData;

int main(int argc, char **argv)
{
    BlisTestSuite	bts;
    test_params_t* params  = NULL;
    blis_string_t* strData = NULL;
    test_ops_t*        ops = NULL;
    input_file_t*    pfile = NULL;
    input_data_t inData;
    string filter_data( "" );

    params  = bts.getParamsStr();
    strData = bts.getStgStr();
    ops     = bts.getOpsStr();
    pfile   = bts.getfileStr();

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    memset( &inData, 0, sizeof( input_data_t ) );
    memset( params, 0, sizeof( test_params_t ) );
    memset( strData, 0, sizeof( blis_string_t ) );
    memset( ops, 0, sizeof( test_ops_t ) );
    memset( pfile, 0, sizeof( input_file_t ) );

    // Initialize some strings.
    bts.libblis_test_init_strings( strData );

    if(argc <= 1)
    {
         // Read the global parameters file.
        bts.libblis_test_read_params_file( strData->libblis_test_parameters_filename,
                           params, strData->libblis_test_alphabeta_parameter);

       // Read the operations parameter file.
        bts.libblis_test_read_ops_file(strData->libblis_test_operations_filename, ops);

        bts.CreateGtestFilters(ops, filter_data);
    }
    else
    {
        // Read the global parameters file.
        bts.libblis_test_inpfile(argv[1], pfile);

        bts.CreateGtestFilters_api(pfile, filter_data);
    }

    inData.params  = params;
    inData.ops     = ops;
    inData.pfile   = pfile;
    if(pfile->fileread != 1) {
        inData.pthread = (bli_pthread_t *)malloc( sizeof( bli_pthread_t ) * params->n_app_threads );
        inData.tdata   = (thread_data_t *)malloc( sizeof( thread_data_t ) * params->n_app_threads );
    }

    inputData.push_back(inData);

    ::testing::GTEST_FLAG(filter) = filter_data.c_str();
    testing::InitGoogleTest(&argc, argv);
    int retval = RUN_ALL_TESTS();

    if(pfile->fileread != 1) {
        free( inData.pthread );
        free( inData.tdata );
    }

    return retval;
}
