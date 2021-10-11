"""Copyright (C) 2021, Advanced Micro Devices, Inc. All Rights Reserved"""
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
        'if(${TARGET_ARCH} STREQUAL amdzen)\nadd_subdirectory(${CMAKE_BINARY_'
        'DIR}/ref_kernels/generic ${CMAKE_BINARY_DIR}/ref_kernels/generic)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen ${CMAKE_BINARY_'
        'DIR}/ref_kernels/zen)\nadd_subdirectory(${CMAKE_BINARY_DIR}/'
        'ref_kernels/zen2 ${CMAKE_BINARY_DIR}/ref_kernels/zen2)\n'
        'add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen3 '
        '${CMAKE_BINARY_DIR}/ref_kernels/zen3)\nelse()', '\n')
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


def add_macro_to_cfiles(cfiles, macro):
    for cfile in cfiles:
        if os.path.exists(cfile):
            write_to_file(cfile, macro)


if __name__ == '__main__':
    cwd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    source_path = os.path.join(cwd, 'ref_kernels')
    build_path = sys.argv[1].replace('/', '\\')
    dest_path = os.path.join(build_path, 'ref_kernels')
    if os.path.exists(dest_path):
        remove_folder(dest_path)

    temp = os.path.join(cwd, 'temp')
    create_folder(temp)
    execute_and_check('XCOPY {} {} /E'.format(source_path, temp))
    create_folder(os.path.join(dest_path, 'zen'))
    create_folder(os.path.join(dest_path, 'zen2'))
    create_folder(os.path.join(dest_path, 'zen3'))
    create_folder(os.path.join(dest_path, 'generic'))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen2')))
    execute_and_check('XCOPY {} {} /E'.format(
        temp, os.path.join(dest_path, 'zen3')))
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
    cfiles_in_generic = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                          .format(os.path.join(dest_path,
                                                               'generic')))
    cfiles_in_generic = cfiles_in_generic.split('\r\n')
    add_macro_to_cfiles(cfiles_in_generic,
                        '\n#define BLIS_CNAME_INFIX _generic\n')
    cfiles_in_zen = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                          .format(os.path.join(dest_path,
                                                               'zen')))
    cfiles_in_zen = cfiles_in_zen.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen,
                        '\n#define BLIS_CNAME_INFIX _zen\n')
    cfiles_in_zen2 = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                          .format(os.path.join(dest_path,
                                                               'zen2')))
    cfiles_in_zen2 = cfiles_in_zen2.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen2,
                        '\n#define BLIS_CNAME_INFIX _zen2\n')
    cfiles_in_zen3 = execute_and_check('cd {} && dir / s / b / o: gn *.c'
                                          .format(os.path.join(dest_path,
                                                               'zen3')))
    cfiles_in_zen3 = cfiles_in_zen3.split('\r\n')
    add_macro_to_cfiles(cfiles_in_zen3,
                        '\n#define BLIS_CNAME_INFIX _zen3\n')
