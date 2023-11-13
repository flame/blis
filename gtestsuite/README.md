# Dependencies
* GoogleTest is used as the tool for testing. The library is fetched and build at configuration time. No installation necessary.
* An installation location for BLIS needs to be passed in as an argument during cmake invocation.
* A path to a reference library needs to be passed during cmake invocation. Currently, MKL, OpenBLAS and Netlib are supported.

# Layout
The repo is organized as follows
* testinghelpers
    * inc
        * common
        * level1
        * level2
        * level3
        * util
    * src
        * common
        * level1
        * level2
        * level3
        * util
* testsuite
    * inc
    * level1
        * addv
        * axpyv
    * level2
    * level3
    * utils

where the code architecture is separated into two parts.

First, from testinghelpers directory a library testinghelpers.a is created. This library holds helper functionality for the tests and functions to compute reference results using the CBLAS interface from the library that was provided as a reference during the CMake invokation.

Testsuite includes headers from testinghelpers/inc and links testinghelpers.a.
* `inc` directory has the comparison functionalit to determine SUCCESS or FAILURE of tests.
* The subdirectories are named after the BLAS level or util (as described in BLIS documentation) and each level holds directories depending on the functionality. Then each of those, for example axpyv, consists of cpp files. This is the main directory where developers add their unit tests - on the corresponding functionality directory. There is one executable per functionality directory. For example, the executable with all tests for axpy, is named as testsuite/level1/axpy.There is the option to build and/or run the tests only for a specific API, or a specific level.

# Basic CMake Configuration
First create and `build` directory using
```console
$ mkdir build
$ cd build
```
## Configure BLIS GTestSuite with OpenBLAS as reference
```console
$ cmake .. -DBLIS_PATH=/path_to_blis_installation -DREF_CBLAS=OpenBLAS -DOPENBLAS_PATH=/path_to_openblas_lib
```
## Configure BLIS GTestSuite with Netlib as reference
```console
$ cmake .. -DBLIS_PATH=/path_to_blis_installation -DREF_CBLAS=Netlib -DNETLIB_PATH=/path_to_netlib_lib
```
## Configure BLIS GTestSuite with MKL as reference
```console
$ cmake .. -DBLIS_PATH=/path_to_blis_installation -DREF_CBLAS=MKL [-DMKL_PATH=/path_to_mkl_lib]
```
MKL_PATH is an optional argument. The default is `$ENV{MKLROOT}/lib/intel64`.

## Configure BLIS GTestSuite with any dynamic BLAS as reference
```console
$ cmake .. -DBLIS_PATH=/path_to_blis_installation -DREF_LIB_PATH=/path_to_blas_lib/anyblas.so
```

# Additional CMake Configuration Options
There are multiple configuration options to chose from when invoking CMake. Those can be used in addition to the basic configuration above.
## Compiler Options
* `-DCMAKE_CXX_COMPILER=path_to_preferred_compiler` can be used to specify the compiler.
* For example, to compile with Clang, use `-DCMAKE_CXX_COMPILER=clang++`.
## Threading Options
* For single threaded BLIS, use `-DENABLE_THREADING=no`.
* For multithreaded BLIS that uses pthreads, use `-DENABLE_THREADING=pthreads` (Linux only).
* For multithreaded BLIS that uses OpenMP, use `-DENABLE_THREADING=openmp`. [**Default**]
    * GNU OpenMP runtime is used by default on Linux.
    * LLVM OpenMP runtime is used by default on Windows, except if MKL is used as a reference, in which case Intel OpenMP runtime is used.
## Threading Options for MKL (if used as reference)
In general, the variable `MKL_ENABLE_THREADING` gets its value from `ENABLE_THREADING` defined above, but can be overwritten, especially if we want to test single-threaded BLIS with multi-threaded MKL.
* For threaded MKL version, use `-DMKL_ENABLE_THREADING=openmp`.
For threaded MKL the following OpenMP runtimes are used:
* GNU is used by default on Linux.
* Intel is used by default on Windows.

## BLIS Linking Type
* To link static BLIS, use `-DBLIS_LINKING_TYPE=static`. [**Default**]
* To link shared BLIS, use `-DBLIS_LINKING_TYPE=shared`.
## Integer Size
* For testing a 32-bit integer BLIS library, use `-DINT_SIZE=32`. [**Default**"]
* For testing a 64-bit integer BLIS library, use `-DINT_SIZE=64`.
## Address Sanitizer (Linux Only)
* To build using address sanitizer, configure using `-DENABLE_ASAN=ON`. [**OFF by default**]
* An installation to BLIS which was build with ASAN flags[CFLAGS="-O0 -g -fsanitize=address"] needs to be provided.
## Code Coverage (Only GCC Compiler)
* BLIS : Configure BLIS Library with code coverage flags[CFLAGS="-O0 -fprofile-arcs -ftest-coverage"], compile and install.
* Gtestsuite : To build for code coverage, configure cmake with `-DENABLE_COVERAGE=ON`. [**OFF by default**] and then compile and run the executable.
* CodeCoverage : in gtestsuite folder, run the below mentioned steps or bash script - to generate html LCOV-code coverage report.
                 Run the bash script : bash codecov.sh <blis_obj_path> <out_dir_name>
                                      or
                 Steps to generate html LCOV-code coverage report.
                 1. lcov --capture --directory <obj_path> --output-file <out_dir>.info
                 2. lcov --remove <out_dir>.info -o <out_dir_fir>.info '/usr/*' '/*/_deps/*'
                 3. genhtml <out_dir_fir>.info --output-directory <out_dir>
                 4. In <out_dir>, open index.html file
## BLIS Library Interface to be Tested
* To build the testsuite using BLAS interface, configure using `-DTEST_INTERFACE=BLAS`. [**Default**]
* To build the testsuite using CBLAS interface, configure using `-DTEST_INTERFACE=CBLAS`.
* To build the testsuite using BLIS-typed interface, configure using `-DTEST_INTERFACE=BLIS_TYPED`. Note that more tests are built for this option, due to the extended APIs.

# Building the Tests
After the successful configuration of CMake, we can build the tests. The following steps are taken by the building process:
1. Building testinghelpers.a.
2. Getting and building GoogleTest libraries.
3. Building the tests in testsuite.
The code is modular and we can build and run all of the executables at once or only specific parts of the testsuite. The targets recursively, so that only tests related to specific functionality can be build and run.
### To build everything use
```console
$ make -j
```
### To build all tests for a specific level use:
```console
$ make -j testsuite.level1
```
### To build all tests for a specific API use:
```console
$ make -j testsuite.level1.axpyv
```
## To build only the testing library use:
```console
$ make -j testinghelpers
```
This can be helpful if you are looking to understand how things are set up in testinghelpers.a.

# Running Tests
## Using CTest
CTest is a test driver program; an executable that comes with CMake and handles running the tests for the project. CTest views each executable as a test. In reality, each executable has many GoogleTest tests implemented and if any of those fails, CTest will give a failure as well. To get a detailed report from CTest, please look into the log files in build/Testing/Temporary directory, which is created automatically.

### To run all tests use:
```console
$ ctest
```
### To run all tests in parallel use:
```console
$ ctest -j3
```
The above command will run all tests using 3 threads.
### To run tests of a specific level use:
```console
$ ctest -R level1
```
The above command will run only the level1 tests.
### To run tests of a specific API use:
```console
$ ctest -R gemm
```
The above command will run only the gemm tests.
## Other CTest options
There are several other options that can be used when running CTest.
One good example is using --test-load which is particularly helpful when the code is being tested is parallel and the option of running tests in parallel (e.g., `-j12`) has been used as well.
To see what is available use:
```console
ctest --help
```
You can also find more details in [CMake Documentation](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

## Using the Executables
As we mentioned earlier, all cpp files of each API directory are compiled into one executable. This executable can be run separately which can be very useful while developing or debugging.
When MKL is used as a reference, the following environment variables need to be set before calling the executables, depending on the configuration.
* MKL_INTERFACE_LAYER=LP64 or MKL_INTERFACE_LAYER=ILP64 depending on whether 32 or 64 bit integers are used, respectivelly.
* MKL_THREADING_LAYER=SEQUENTIAL for sequential MKL.
* MKL_THREADING_LAYER=INTEL or MKL_THREADING_LAYER=GNU depending on whether we execute on Windows or on Linux, respectivelly.

### To run all addv tests use:
```console
$ ./testsuite.level1.addv
```
### To run a more specific tests, say the snrm2 tests of nrm2, use:
```console
$ ./testuite.util.nrm2 --gtest_filter="*snrm2*"
```
## Running tests using Valgrind
We can run any executable using valgrind as usual. For example, use the following command
```console
$ OMP_NUM_THREADS=1 valgrind ./testsuite.level3.gemm
```

## Clean cmake generated files
```console
$ make distclean
```

## Other GoogleTest options
A list of useful options:
### Test Selection
--gtest_list_tests
      List the names of all tests instead of running them. The name of
      TEST(Foo, Bar) is "Foo.Bar".
--gtest_filter=POSITIVE_PATTERNS[-NEGATIVE_PATTERNS]
      Run only the tests whose name matches one of the positive patterns but
      none of the negative patterns. '?' matches any single character; '*'
      matches any substring; ':' separates two patterns.

### Test Execution
--gtest_repeat=[COUNT]
      Run the tests repeatedly; use a negative count to repeat forever.

### Test Output
--gtest_brief=1
      Only print test failures.
--gtest_output=(json|xml)[:DIRECTORY_PATH/|:FILE_PATH]
      Generate a JSON or XML report in the given directory or with the given
      file name. FILE_PATH defaults to test_detail.xml.
### Assertion Behavior
--gtest_break_on_failure
      Turn assertion failures into debugger break-points.
--gtest_throw_on_failure
      Turn assertion failures into C++ exceptions for use by an external
      test framework.

There are several other options that can be used when running an executable which has GoogleTests implemented. To see what is available use:
```console
./testsuite.util.nrm2 --help
```

# How to Add New Tests
There are two ways to add new tests.
### Modify an existing cpp file
* Add any of the GoogleTest testing API, e.g., `TEST()` in any of the existing cpp files.
* Rebuild.
* Rerun.

### Add a new cpp file
* Add a cpp file which has any of the GoogleTest testing API, e.g., `TYPED_TEST()` calls in it.
* Reconfigure cmake as mentioned above.
* Rebuild.
* Rerun.

# Wrong Input Testing
When testing for wrong input parameter values, then the code is meant to return early and the data of vectors and/or matrices should not be accessed. Therefore, the values of the elements of vectors and/or matrices are not important and thus we follow the methodolody of typed-testing. Since there are no error codes returned by the APIs, we write the tests doing the following:
* check return value to be as expected. For example, in nrm2, the default returned value is 0. For gemm, C should not be modified so we check against a copy of C, before calling into gemm. Note that for APIs where there is pure output (e.g., norm from nrm2), since the default initialization of scalars is zero, it's a better practice to initialize the output to a nonzero value prior calling the API. That way we can ensure that the returned value was modified correctly by the function and it's not using the default value.
* since the checks are expected to be done before any computation, use nullptr as inputs for the other F.P. data. For example, in nrm2, pass nullptr as x and in gemm, pass nullptr as A and B. If those get accessed, then the code would crush so that would show bugs.

Currently, we have the following behaviour in the different interfaces:
* BLIS-typed prints and aborts.
* BLAS prints and returns.
* CBLAS prints and exits.
For that reason, we currently test only for BLAS APIs, so ensure to add the #ifdef's as appropriate. Note that printing seems to be inconsistent.

A test program would be looking like the following:
```cpp
#include <gtest/gtest.h>
#include "common/testing_helpers.h"
#include "gemm.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

/**
 * Testing invalid/incorrect input parameters.
 *
 * storage : 'c', 'r', note BLAS is 'c' only.
 * transa, transb : 'n', 't', 'c'
 * m, n, k >= 0
 * lda, ldb, ldc >= max(m/n/k, 1)
*/
template <typename T>
class gemm_IIT : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(gemm_IIT, TypeParam);

// Adding namespace to get default parameters from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#ifdef TEST_BLAS
TYPED_TEST(gemm_IIT, wrong_transa)
{
  using T = TypeParam;
  // Create a vector with some default value.
  std::vector<T> c(M*N, T{2.0});
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS gemm with a wrong value for transa.
  gemm<T>( STORAGE, 'k', TRANS, M, N, K, nullptr, nullptr, LDA,
                              nullptr, LDB, nullptr, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, M, N, c.data(), c_ref.data(), LDC);
}
#endif
```

# A short explanation on how lda increments for matrices work
Say we have an m-by-n matrix A, which is stored in column major order. Then to access the elements we go through the matrix column by column. In this case, to store the full matrix in an array we need to store m * n elements and to access an element A(i,j), we need to know the size of the column, which is in this case m. Now let's assume that the matrix A is part of a bigger k-by-n matrix B. In this case, A is part of an array with k * n and to access an element A(i,j), we need to know the size of the column of B, which is in this case k. The leading dimension shows to us how many elements we need to go through, to be able to access the next column of a matrix. So, the requirement is to have lda >= max(1, m).

     __________________                         __________________
A = | |  |             |                  B =  | |  |             |
    | |  |             |                       | |  |             |
  m | |  |             |                    k  | |  |             |
    | |  V             |                       | |  V             |
    | |                |                       | |                |
    | |  a             |                       | |  a             |
    |_V________________|                       |_|________________|
            n                                  | |      n         |
                                               | |                |
                                               | |                |
                                               |_V________________|

For an m-by-n matrix A, stored in row major order, we traverse the matrix row by row. We need to store full matrix A into an array of size m * n (as before) but now we use the number of columns to move to the next row as we iterate through the elements, which is n. Similarly to above, if A is part of a bigger matrix B of size m-by-l, in order to access an element A(i,j), we need to know the number of columns, l. In this case, the leading dimension shows to us how many elements we need to go through while traversing the matrix to access the next row of A. So, the requirement in this case is lda >= max(1, n).
     __________________                         _________________________
A = |                  |                  B =  |                  |      |
    |                  |                       |                  |      |
  m |                  |                    m  |                  |      |
    |                  |                       |                  |      |
    | -------------->  |                       |  ---------------------> |
    | ---> a           |                       |  ---> a          |      |
    |__________________|                       |__________________|______|
            n                                               l

Since in generic testing we generate tests for matrices with arbitrary sizes m and n and we need to check for column-major and row-major order in a generic way, using a fixed value for lda is not trivial. Consider the case where m and n take values from ::testing::Values(30, 40), and storage takes values from ::testing::Values('c','r').
Then, we generate the following test combinations:
1. m = 30, n = 30, storage = 'c'
2. m = 30, n = 30, storage = 'r'
3. m = 40, n = 40, storage = 'c'
4. m = 40, n = 40, storage = 'r'
5. m = 30, n = 40, storage = 'c'
6. m = 30, n = 40, storage = 'r'
7. m = 40, n = 30, storage = 'c'
8. m = 40, n = 30, storage = 'r'

If we want to test for different lda combinations as well, especially for the case where m != n, this would cause a problem as follows:
If lda = 30, for the cases 3, 4, 7, 8 above, lda < max(1, m), so the requirement is not satisfied. Another issue is that lda depends on the storage type and on whether we test for non transpose, transpose or conjugate transpose matrices.

To overcome this issue and generate tests which fullfill the requirements for the correct value of the leading dimension of a matrix we use **lda increments** and do the lda computation as follows:
* Depending on the parameters storage and trans, compute lda = max(1, k), where k is m or n, depending on the requirements.
* Add the lda_inc parameter: lda += lda_inc

To test an m-by-n matrix A (column-major), stored in an array a, use lda_inc = 0 as a parameter to the test generator. To test for the case where A is a submatrix of k-by-n matrix B, use lda_inc = k-m.

# BLIS GTestSuite on Windows
Building and runing GTestSuite on Windows is somewhat similar to Linux. In this section we focus on what is different, so please read the previous sections so that you have the complete picture.

The instructions are given for building and running through the terminal, but using cmake-gui is also possible. An x64 Native Toolbox Command Prompt can be used so that the environment is set and the necessary compilers are available.

## Build System Generators
On the descriptions above we assumed that Make will be used to build the libraries. The same instructions can be modified so that Ninja is used instead in a straigthforward manner. On Windows, where Make cannot be used, you can invoke CMake in one of the two ways below. Beware that Windows environment needs to be set correctly otherwise there might be some libraries missing.

### Generate using Ninja
```console
$ cmake .. -G "Ninja" -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl [more variables that we'll explain later]
```
We specify the compilers to clang-cl, so that it's the same as the default option for BLIS library builds.

### Generate using Visual Studio
```console
$ cmake .. -G "Visual Studio 17 2022" -TClangCl [more variables that we'll explain later]
```
-TClangCl sets the toolbox to be used for the generation. To see what VS generators you have available, type
```console
$ cmake --help
```

## Configuring CMake
On Windows currently the BLIS repo does not have a CMake target for the library and the headers, so to configure properly we need to replace the variable BLIS_PATH that was used on Linux with the variables BLIS_LIB_PATH and BLIS_INCLUDE.
So, we can invoke cmake using
```console
$ cmake ..  -G "Visual Studio 17 2022" -TClangCl -DBLIS_LIB_PATH=/path_to_blis_libraries -DBLIS_INCLUDE=/path_to_blis_headers -DREF_CBLAS=OpenBLAS -DOPENBLAS_PATH=/path_to_openblas_dll
```
## Additional CMake Configuration Opions
The configuration is similar to Linux. In this section we only mention the specifics for Windows.
### BLIS Linking Type
* `-DBLIS_LINKING_TYPE=static` implies that AOCL-LibBlis-Win.lib (or AOCL-LibBlis-Win-MT.lib) will be tested.
* `-DBLIS_LINKING_TYPE=shared` implies that AOCL-LibBlis-Win-dll.lib (or AOCL-LibBlis-Win-MT-dll.lib) will be tested. Windows needs to find the coresponding dlls (AOCL-LibBlis-Win-dll.dll or AOCL-LibBlis-Win-MT-dll.dll) to be able to run the tests. The CMake system uses the prepends the Environment's PATH to the path provided during configuration, so that `ctest` can find the dll. To run the executables separately, you need to copy the dll manually, or specify the PATH.
### Threading Options
The path to the OpenMP runtime needs to be passed using `-DOpenMP_libomp_LIBRARY=/path_to_openmp_runtime`.

## Building the Tests
The building process is similar to Windows with the main difference that a testinghelpers.lib is built.
### Building with Ninja
To build with Ninja, replace the word `make` with `ninja`.
### Building with Visual Studio
To build everything use
```console
$ cmake --build . --config Release
```
To build a specific target use
```console
$ cmake --build . --config Release --target testsuite.level1
```

## To run tests with Visual Studio
The process is similar to Linux, apart from the modification below. For parallel builds etc. you can add the options after `Release`.
```console
$ ctest -C Release
```

## Using the executables using Visual Studio
Visual Studio is a multiconfig generator. That means that it can build for `Release`, `Debug`, etc. simultaneously. For that reason VS will create a directory named Release, where it puts all the executables. So, to runn all addv tests, use
```console
$ cd Release
$ testsuite.level1.addv.exe
```
Then, you can use filters in the same way if you need to.
