##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

# Import modules
import os
import sys

def check_blastest():
    results_file_path = sys.argv[1]
    results_directory = os.listdir(results_file_path)
    has_failure = False
    is_empty = False
    for fname in results_directory:
        if os.path.isfile(results_file_path + os.sep + fname) and "out" in fname:
            file = open(results_file_path + os.sep + fname, 'r')
            # read all content of a file
            content = file.read()
            if content == "":
                is_empty = True
            # check if string present in a file
            if "*****" in content:
                has_failure = True
    if has_failure:
        print("\033[0;31m At least one BLAS test failed. :( \033[0m")
        print("\033[0;31m Please see the corresponding out.* for details. \033[0m")
    elif is_empty:
        print("\033[0;31m At least one BLAS test resulted without a PASS. :( \033[0m")
        print("\033[0;31m Please ensure that the corresponding out.* was generated correctly. \033[0m")
    else:
        print("\033[0;32m All BLAS tests passed! \033[0m")

check_blastest()
