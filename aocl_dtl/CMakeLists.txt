##Copyright (C) 2020, Advanced Micro Devices, Inc ##

target_sources("${PROJECT_NAME}" 
     PRIVATE 
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtlcf.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl_blis.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl_blis.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclfal.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclfal.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclflist.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclflist.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclos.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclos.h
     ${CMAKE_CURRENT_SOURCE_DIR}/aocltpdef.h
     )
