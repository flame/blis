##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntx_init_haswell.c
   )

set(SUBDIRECTORIES "")
set(RELATIVE_PATH "haswell")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()

if(FILES)
    #Add source files to target
    target_sources("${PROJECT_NAME}" PRIVATE ${FILES})

    #Install our source files
    install(FILES ${FILES} DESTINATION ${RELATIVE_PATH})
endif()
