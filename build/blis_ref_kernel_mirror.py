"""Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved."""

################################################################################
# This file is used to mirroring the refkernels folder data into to zen, zen2, #
# zen3, zen4 and generic folder.                                               #
# Rename all .c files by adding zen, zen2, zen3, zen4 and generic for the      #
# corresponding folder .c files and update the corresponding CMakeLists.txt    #
# file for amdzen (dynamic dispatcher) config option.                          #
#                                                                              #
# Usage:                                                                       #
#       python blis_ref_kernel_mirror.py <project build directory name>        #
#                                                                              #
# Author: Chandrashekara K R <chandrkr@amd.com>                                #
#                                                                              #
################################################################################
import os
import shutil
import subprocess
import sys


def create_folder(path):
    """ Function to create the folder in an given path.

    Args:
        path:- Folder path to create.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def remove_folder(path):
    """ Function to delete folder in a given path.

    Args:
        path:- Folder path to delete.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def execute_and_check(cmd):
    """ Function to run power shell command in windows and bash command
    in linux.

    Arg:
        cmd:- Power shell/ bash command.

    Return:
         Returns command output on success and terminates the execution
         on failure.
    """
    print('********************************************************')
    print('Started execution of {} command...\n'.format(cmd))
    print('********************************************************')

    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output, err = proc.communicate()

    if not proc.returncode:
        print('********************************************************')
        print('Execution of command : {} - was successful'.format(cmd))
        print('command {} output: {}'.format(cmd,
                                             output.decode('ASCII')))
        print('********************************************************')
        return output.decode('ASCII')
    else:
        print('########################################################')
        print('Execution of command : {} - was failed'.format(cmd))
        print('command {} output: {}\n{}\n'.format(cmd, output.decode(
            'ASCII'), err.decode('ASCII')))
        exit(1)


def remove_lines_in_file(filename):
    with open(filename, 'r') as fd:
        file_content = fd.read()
    file_content = file_content.replace(
        'if(${TARGET_ARCH} STREQUAL amdzen)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/generic '
        '${CMAKE_BINARY_DIR}/ref_kernels/generic)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen '
        '${CMAKE_BINARY_DIR}/ref_kernels/zen)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen2 '
        '${CMAKE_BINARY_DIR}/ref_kernels/zen2)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen3 '
        '${CMAKE_BINARY_DIR}/ref_kernels/zen3)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen4 '
        '${CMAKE_BINARY_DIR}/ref_kernels/zen4)\nelse()', '\n')
    data = file_content.replace('endif()', '\n')
    with open(filename, 'w') as fd:
        fd.write(data + '\n')


def write_to_file(filename, data):
    with open(filename, 'r') as fd:
        file_content = fd.read()
    file_content = file_content.split('#include "blis.h"')
    data = '\n'.join([file_content[0], '#include "blis.h"', data] +
                     file_content[1:])

    with open(filename, 'w') as fd:
        fd.write(data + '\n')


def update_cmakelists_contents(cmakefiles, replacement_str):
    for cmakefile in cmakefiles:
        if os.path.exists(cmakefile):
            # Updating the modified .c files name in CMakeLists.txt
            with open(cmakefile, 'r') as fd:
                file_content = fd.read()
                file_content = file_content.replace(
                    'ref.c', replacement_str + '_ref.c')
            with open(cmakefile, 'w') as fd:
                fd.write(file_content)


def add_macro_to_cfiles(cfiles, macro):
    for cfile in cfiles:
        if os.path.exists(cfile):
            write_to_file(cfile, macro)
            # Renaming the .c files name to incorporate with linux
            os.rename(cfile,  cfile.split('ref.c')[0] + macro.split(' ')[
                -1].split('\n')[0][1:] + '_ref.c')


if __name__ == '__main__':
    cwd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    source_path = os.path.join(cwd, 'ref_kernels')
    build_path = sys.argv[1].replace('/', '\\')
    dest_path = os.path.join(build_path, 'ref_kernels')
    if os.path.exists(dest_path):
        remove_folder(dest_path)

    # Creating all the required folders
    temp = os.path.join(cwd, 'temp')
    create_folder(temp)
    execute_and_check('XCOPY {} {} /E'.format(source_path, temp))
    create_folder(os.path.join(dest_path, 'zen'))
    create_folder(os.path.join(dest_path, 'zen2'))
    create_folder(os.path.join(dest_path, 'zen3'))
    create_folder(os.path.join(dest_path, 'zen4'))
    create_folder(os.path.join(dest_path, 'generic'))
    # Mirroring refkernels folder data to zen, zen2, zen3, zen4 and generic folder
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen2')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen3')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen4')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'generic')))
    remove_folder(temp)
    remove_lines_in_file(os.path.join(
        dest_path, 'generic', 'CMakeLists.txt'))
    remove_lines_in_file(os.path.join(
        dest_path, 'zen', 'CMakeLists.txt'))
    remove_lines_in_file(os.path.join(
        dest_path, 'zen2', 'CMakeLists.txt'))
    remove_lines_in_file(os.path.join(
        dest_path, 'zen3', 'CMakeLists.txt'))
    remove_lines_in_file(os.path.join(
        dest_path, 'zen4', 'CMakeLists.txt'))
    cfiles_in_generic = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                          .format(os.path.join(dest_path,
                                                               'generic')))
    cfiles_in_generic = cfiles_in_generic.split('\r\n')
    add_macro_to_cfiles(cfiles_in_generic,
                        '\n#define BLIS_CNAME_INFIX _generic\n')
    # Listing all CMakelists.txt file from generic folder and updating them.
    cmake_files_in_generic = execute_and_check(
        'cd {} && dir / s / b / o: gn CMakeLists.txt'.format(
            os.path.join(dest_path, 'generic')))
    cmake_files_in_generic = cmake_files_in_generic.split('\r\n')
    update_cmakelists_contents(cmake_files_in_generic, 'generic')
    cfiles_in_zen = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                      .format(os.path.join(dest_path, 'zen')))
    cfiles_in_zen = cfiles_in_zen.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen,
                        '\n#define BLIS_CNAME_INFIX _zen\n')
    # Listing all CMakelists.txt file from zen folder and updating them.
    cmake_files_in_zen = execute_and_check(
        'cd {} && dir / s / b / o: gn CMakeLists.txt'.format(
            os.path.join(dest_path, 'zen')))
    cmake_files_in_zen = cmake_files_in_zen.split('\r\n')
    update_cmakelists_contents(cmake_files_in_zen, 'zen')
    cfiles_in_zen2 = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                       .format(os.path.join(dest_path, 'zen2')))
    cfiles_in_zen2 = cfiles_in_zen2.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen2,
                        '\n#define BLIS_CNAME_INFIX _zen2\n')
    # Listing all CMakelists.txt file from zen2 folder and updating them.
    cmake_files_in_zen2 = execute_and_check(
        'cd {} && dir / s / b / o: gn CMakeLists.txt'.format(
            os.path.join(dest_path, 'zen2')))
    cmake_files_in_zen2 = cmake_files_in_zen2.split('\r\n')
    update_cmakelists_contents(cmake_files_in_zen2, 'zen2')
    cfiles_in_zen3 = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                       .format(os.path.join(dest_path, 'zen3')))
    cfiles_in_zen3 = cfiles_in_zen3.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen3,
                        '\n#define BLIS_CNAME_INFIX _zen3\n')
    # Listing all CMakelists.txt file from zen3 folder and updating them.
    cmake_files_in_zen3 = execute_and_check(
        'cd {} && dir / s / b / o: gn CMakeLists.txt'.format(
            os.path.join(dest_path, 'zen3')))
    cmake_files_in_zen3 = cmake_files_in_zen3.split('\r\n')
    update_cmakelists_contents(cmake_files_in_zen3, 'zen3')
    cfiles_in_zen4 = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                       .format(os.path.join(dest_path, 'zen4')))
    cfiles_in_zen4 = cfiles_in_zen4.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen4,
                        '\n#define BLIS_CNAME_INFIX _zen4\n')
    # Listing all CMakelists.txt file from zen4 folder and updating them.
    cmake_files_in_zen4 = execute_and_check(
        'cd {} && dir / s / b / o: gn CMakeLists.txt'.format(
            os.path.join(dest_path, 'zen4')))
    cmake_files_in_zen4 = cmake_files_in_zen4.split('\r\n')
    update_cmakelists_contents(cmake_files_in_zen4, 'zen4')
