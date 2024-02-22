##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

# Import modules
import os
import sys

def check_blistest():
    results_file = sys.argv[1]
    with open(results_file, 'r') as file:
        # read all content of a file
        content = file.read()
        # check if string present in a file
        if "FAILURE" in content:
            print("\033[0;31m At least one BLIS test failed. :( \033[0m")
            print("\033[0;31m Please see the corresponding output.testsuite* for details. \033[0m")
            exit(1)
        elif not "PASS" in content:
            print("\033[0;31m No BLIS test resulted in PASS. :( \033[0m")
            print("\033[0;31m Please ensure that the corresponding output.testsuite* was generated correctly. \033[0m")
            exit(1)
        else:
            print("\033[0;32m All BLIS tests passed! \033[0m")
            exit(0)
check_blistest()
