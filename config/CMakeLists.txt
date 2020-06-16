##Copyright (C) 2020, Advanced Micro Devices, Inc ##

if(${TARGET_ARCH} STREQUAL zen2)
message("The configuration is : ${TARGET_ARCH}")
add_subdirectory(zen2)
elseif(${TARGET_ARCH} STREQUAL zen)
message("The configuration is : ${TARGET_ARCH}")
add_subdirectory(zen)
else(${TARGET_ARCH} STREQUAL haswell)
message("The configuration is : ${TARGET_ARCH}")
add_subdirectory(haswell)
endif()