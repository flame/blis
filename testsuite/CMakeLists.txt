##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(TestSuite "")

add_subdirectory(src)

target_link_libraries(TestSuite debug "${PROJECT_NAME}.lib")
target_link_libraries(TestSuite optimized "${PROJECT_NAME}.lib")



