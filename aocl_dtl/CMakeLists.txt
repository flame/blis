##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources("${PROJECT_NAME}"
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aocldtl_blis.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclfal.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclflist.c
     ${CMAKE_CURRENT_SOURCE_DIR}/aoclos.c
     )
