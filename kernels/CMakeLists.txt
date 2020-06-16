##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

if(${TARGET_ARCH} STREQUAL zen2)
add_subdirectory(haswell)
add_subdirectory(zen)
add_subdirectory(zen2)
endif()