#!/bin/bash

main() 
{
    script_name=${0##*/}
    echo " "
    #echo " "
    #echo " "${script_name}
    #echo " "
    #echo " Objective:"
    #echo "   The script removes ftnlen-related arguments which are not necessary in lapack2flame."
    #echo " "

    files="$(find . -maxdepth 1 -name "*.c")"

    for file in ${files}; do
        echo -ne "   Removing ftnlen from ... ${file}                  "\\r
        tmp_file=$(echo "${file}.back")

        # tr -s '\t\n' ' ' < ${file} \ # remove return and tab leading a single line source 
        # sed 's/;/;\'$'\n/g'            \ # add return after ;
        # sed 's/{/{\'$'\n/g'            \ # add return after {
        # sed 's/}/}\'$'\n/g'            \ # add return after }
        # sed 's/\*\//\*\/\'$'\n/g'      \ # add return after */
        # sed 's/  */\ /g'               \ # remove multiple spaces into a single space
        # sed 's/, ftnlen [0-9a-zA-Z_]*//g' \ # remove ftnlen in the function definition
        # sed 's/, (ftnlen)[0-9]*//g'    \ # remove ftnlen in function arguments
        # > ${tmp_file}                    # wrote it to file

        # 
        # | sed 's/\(\#define *[a--zA-Z] *[()a-zA-Z_0-9]*\)/\1\'$'\n/g' \
        #
        # int s_cat(char *, char **, integer *, integer *, ftnlen);
        #     s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);

        ( tr -s '\t\n' ' ' < ${file} \
            | sed 's/;/;\'$'\n/g' \
            | sed 's/{/{\'$'\n/g' \
            | sed 's/}/}\'$'\n/g' \
            | sed 's/\*\//\*\/\'$'\n/g' \
            | sed 's/  */\ /g' \
            > ${tmp_file} ;
            rm -f ${file}  ;
			echo "" >> ${tmp_file}
			cp ${tmp_file} ${file} ;
            rm -f ${tmp_file} ) 
    done
    return 0
}

main "$@"

#            | sed 's/, ftnlen *[0-9a-zA-Z_]*//g' \
#            | sed 's/, ( *ftnlen) *[0-9]*//g' \

#            | sed 's/\((equiv_[0-9])\)/\1\'$'\n/g' \
#            | sed 's/\(\#undef *[a-zA-Z]*\)/\1\'$'\n/g' \


#        ( tr -s '\t\n' ' ' < ${file} \
#            | sed 's/;/;\'$'\n/g' \
#            | sed 's/\((equiv_[0-9])\)/\1\'$'\n/g' \
#            | sed 's/\(\#undef *[a-zA-Z]*\)/\1\'$'\n/g' \
#            | sed 's/{/{\'$'\n/g' \
#            | sed 's/}/}\'$'\n/g' \
#            | sed 's/\*\//\*\/\'$'\n/g' \
#            | sed 's/  */\ /g' \
#            | sed 's/, ftnlen *[0-9a-zA-Z_]*//g' \
#            | sed 's/, ( *ftnlen) *[0-9]*//g' \
#            > ${tmp_file} ;
#            rm -f ${file}  ;
#            sed 's/, ftnlen//g' < ${tmp_file} > ${file} ;  # remove remainder of ftnlen used alone
#            rm -f ${tmp_file} ) 
