##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

add_subdirectory(haswell)
add_subdirectory(zen)

if(${TARGET_ARCH} STREQUAL zen4 OR 
   ${TARGET_ARCH} STREQUAL amdzen)
   add_subdirectory(skx)
   add_subdirectory(zen4)
endif()