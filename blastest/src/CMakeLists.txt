##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

add_executable(cblat1 cblat1.c)
target_link_libraries(cblat1 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(cblat2 cblat2.c)
target_link_libraries(cblat2 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(cblat3 cblat3.c)
target_link_libraries(cblat3 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(dblat1 dblat1.c)
target_link_libraries(dblat1 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(dblat2 dblat2.c)
target_link_libraries(dblat2 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(dblat3 dblat3.c)
target_link_libraries(dblat3 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(sblat1 sblat1.c)
target_link_libraries(sblat1 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(sblat2 sblat2.c)
target_link_libraries(sblat2 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(sblat3 sblat3.c)
target_link_libraries(sblat3 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(zblat1 zblat1.c)
target_link_libraries(zblat1 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(zblat2 zblat2.c)
target_link_libraries(zblat2 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )

add_executable(zblat3 zblat3.c)
target_link_libraries(zblat3 PRIVATE  "${F2C_LIB}" "${LIB_NAME}.lib" )
