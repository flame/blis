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
## Threading Options (Linux Only)
* For single threaded BLIS, use `-DENABLE_THREADING=no`.
* For multithreaded BLIS that uses pthreads, use `-DENABLE_THREADING=pthreads`.
* For multithreaded BLIS that uses OpenMP, use `-DENABLE_THREADING=openmp`. [**Default**]
    * In addition, to use Intel OpenMP runtime, use `-DOpenMP_LIBRARY=Intel`.
    * For GNU OpenMP runtime, use `-DOpenMP_LIBRARY=GNU`. [**Default**]
## BLIS Linking Type (Linux Only) 
* To link static BLIS, use `-DBLIS_LINKING_TYPE=static`. [**Default**]
* To link shared BLIS, use `-DBLIS_LINKING_TYPE=shared`. 
## Address Sanitizer
* To build using address sanitizer, configure using `-DENABLE_ASAN=ON`. [**OFF by default**]
* An installation to BLIS which was build with ASAN flags needs to be provided.
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
### To run all addv tests use:
```console
$ ./test.level1.addv
```
### To run a more specific tests, say the snrm2 tests of nrm2, use:
```console
$ ./test.util.nrm2 --gtest_filter="*snrm2*"
```
## Running tests using Valgrind
We can run any executable using valgrind as usual. For example, use the following command
```console
$ OMP_NUM_THREADS=1 valgrind ./testsuite.level3.gemm
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

