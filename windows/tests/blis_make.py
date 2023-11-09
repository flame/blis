"""Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved."""
import re
import subprocess
import yaml
import sys
import os


class BlisCheck:

    @staticmethod
    def check_execution():

        try:
            with open(r'inputs.yaml') as file:
                input_file = yaml.safe_load(file)
                try:
                    if (sys.argv[1] == '') or (sys.argv[1] == "--h") or (sys.argv[1] == "--help"):
                        print("Below options are available \n")
                        print("usage: python blis_make.py ", end='[')
                        for var in input_file.keys():
                            print(var, end=' | ')
                        print('checkcpp | --h | --help]')
                        sys.exit()
                except IndexError:
                    print("Below options are available \n")
                    print("usage: python blis_make.py ", end='[')
                    for var in input_file.keys():
                        print(var, end=' | ')
                    print('checkcpp | --h | --help]')
                    sys.exit()
                if sys.argv[1] == "check":
                    which_check = 'checkblis-fast'
                    command = "test_libblis.exe -g " + input_file['checkblis-fast'][0] + " -o "+input_file['checkblis-fast'][1]
                    BlisCheck.test_checkblis(which_check, command)
                    flag = 0
                    which_check = 'check'
                    for i in range(len(input_file[which_check])):
                        #print(input_file[which_check][i])
                        if '1' in input_file[which_check][i]:
                            command = input_file[which_check][i]+' > out.'+input_file[which_check][i][:6]
                            print("Running ", input_file[which_check][i], " (output to 'out."+input_file[which_check][i][:6]+"')")
                        else:
                            command = input_file[which_check][i]
                            print("Running ", input_file[which_check][i], " (output to 'out."+input_file[which_check][i][:6]+"')")
                        subprocess.check_call(command , shell=True)
                        with open(r"out."+input_file[which_check][i][:6]) as out_file:
                            strings = re.findall(r'FAIL', out_file.read())
                            if strings:
                                flag += 1

                    if flag:
                        print("At lease one BLAS test failed.")
                        print("Please see out.* files for details")
                    else:
                        print("All BLAS tests passed")

                    sys.exit()

                elif sys.argv[1] == 'checkcpp':
                    files = [f for f in os.listdir('.') if re.search('_blis.exe', f)]
                    #print(files)
                    for executable in files:
                        subprocess.check_call(executable, shell=True)

                    sys.exit()
                else:
                    general_file = input_file[sys.argv[1]][0]
                    operations_file = input_file[sys.argv[1]][1]
                    command = "test_libblis.exe -g " + general_file + " -o " + operations_file
                    BlisCheck.test_checkblis(sys.argv[1], command)

        except Exception as error:
            print(error)

    @staticmethod
    def test_checkblis(which_check, command):
        flag = 0
        with open("output.testsuite.txt", 'w') as f:
            if 'md' in which_check:
                print("Running test_libblis.exe {} with output redirected to 'output.testsuite'".format(
                    "(mixed dt)"))
            elif len(which_check) > 9:
                print("Running test_libblis.exe {} with output redirected to 'output.testsuite'".format(
                    "(" + which_check[10:] + ")"))
            else:
                print("Running test_libblis.exe with output redirected to 'output.testsuite'")
            process = subprocess.Popen(command, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, ''):
                if "FAIL" in line:
                    flag = + 1
                f.write(line)
                sys.stdout.flush()
            process.wait()
            errcode = process.returncode
            f.close()
            if flag:
                print("At least one BLIS test failed. :( \n Please see output.testsuite for details.")
            else:
                print("All BLIS tests passed!")


if __name__ == "__main__":
    #which_check = sys.argv[1]
    calling_Object = BlisCheck()
    calling_Object.check_execution()
