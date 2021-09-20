##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##


set(SUBDIRECTORIES "1" "1f" "2" "3" "util")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()


