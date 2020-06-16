##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

add_executable(TestAxpyv test_axpyv.c)
target_link_libraries(TestAxpyv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestAxpyv optimized "${PROJECT_NAME}.lib")

add_executable(TestCopyv test_copyv.c)
target_link_libraries(TestCopyv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestCopyv optimized "${PROJECT_NAME}.lib")

add_executable(TestDotv test_dotv.c)
target_link_libraries(TestDotv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestDotv optimized "${PROJECT_NAME}.lib")

add_executable(TestGemm test_gemm.c)
target_link_libraries(TestGemm debug "${PROJECT_NAME}.lib")
target_link_libraries(TestGemm optimized "${PROJECT_NAME}.lib")

add_executable(TestGemv test_gemv.c)
target_link_libraries(TestGemv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestGemv optimized "${PROJECT_NAME}.lib")

add_executable(TestGer test_ger.c)
target_link_libraries(TestGer debug "${PROJECT_NAME}.lib")
target_link_libraries(TestGer optimized "${PROJECT_NAME}.lib")

add_executable(TestHemm test_hemm.c)
target_link_libraries(TestHemm debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHemm optimized "${PROJECT_NAME}.lib")

add_executable(TestHemv test_hemv.c)
target_link_libraries(TestHemv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHemv optimized "${PROJECT_NAME}.lib")

add_executable(TestHer test_her.c)
target_link_libraries(TestHer debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHer optimized "${PROJECT_NAME}.lib")

add_executable(TestHer2 test_her2.c)
target_link_libraries(TestHer2 debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHer2 optimized "${PROJECT_NAME}.lib")

add_executable(TestHer2k test_her2k.c)
target_link_libraries(TestHer2k debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHer2k optimized "${PROJECT_NAME}.lib")

add_executable(TestHerk test_herk.c)
target_link_libraries(TestHerk debug "${PROJECT_NAME}.lib")
target_link_libraries(TestHerk optimized "${PROJECT_NAME}.lib")

add_executable(TestScalv test_scalv.c)
target_link_libraries(TestScalv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestScalv optimized "${PROJECT_NAME}.lib")

add_executable(TestSwapv test_swapv.c)
target_link_libraries(TestSwapv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestSwapv optimized "${PROJECT_NAME}.lib")

add_executable(TestTrmm test_trmm.c)
target_link_libraries(TestTrmm debug "${PROJECT_NAME}.lib")
target_link_libraries(TestTrmm optimized "${PROJECT_NAME}.lib")

add_executable(TestTrmv test_trmv.c)
target_link_libraries(TestTrmv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestTrmv optimized "${PROJECT_NAME}.lib")

add_executable(TestTrsm test_trsm.c)
target_link_libraries(TestTrsm debug "${PROJECT_NAME}.lib")
target_link_libraries(TestTrsm optimized "${PROJECT_NAME}.lib")

add_executable(TestTrsv test_trsv.c)
target_link_libraries(TestTrsv debug "${PROJECT_NAME}.lib")
target_link_libraries(TestTrsv optimized "${PROJECT_NAME}.lib")


