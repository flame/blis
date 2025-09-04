#BLIS check execution script

Check execution script covers:
        * checkblis
        * checkblis-fast
        * checkblis-md
        * checkblis-salt

##Requirements
* Install latest version of python from python.org(preferably python 3.5 or greater)
* Add python path and scripts path to the environment variable path

#Copy all the files present in <src>/windows/tests directory to the directory where TestSuite.exe is present
#Open the command prompt and execute the python script and provide an argument(check name)
For example:
python blis_check.py checkblis

#Output can be seen on the command prompt

Note:
   New check execution can be added into the inputs.yaml in the below format
   Ex: new_check: [input.general.filename,input.operations.filename]
