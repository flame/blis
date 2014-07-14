#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#


#
# --- General build system options --------------------------------------------
#

# Uncomment this for verbose output from nmake.
# VERBOSE = 1

# Assign this varible to be the full path to the directory to which you would
# like the BLIS build products to be installed upon running "nmake install".
# The nmake install target will create the install directory and all requisite
# subdirectories if they do not already exist (in which case the user must have
# permission to create these directories).
INSTALL_PREFIX = c:\field\lib


#
# --- Important build system filenames ----------------------------------------
#

# DLL link arguments. The contents of this file should be customized when
# building a dynamically-linked library. The lines of the file should contain
# linker options, library names, and library paths. Note that the library
# paths must be declared in the following form:
#
#   /link /LIBPATH:<path1>
#   /link /LIBPATH:<path2>
#   /link /LIBPATH:<path3>
#
# where <path1>, <path2>, and <path3> are library paths to add to the list
# of paths to search when the linker attempts to locate other libraries
# listed in the file.
LINKARGS_FILENAME = linkargs.txt
LINKARGS_FILEPATH = $(PWD)\$(LINKARGS_FILENAME)

# Various log file names that capture standard output when VERBOSE is undefined.
CC_LOG_FILE   = nmake-cc.log
FC_LOG_FILE   = nmake-fc.log
COPY_LOG_FILE = nmake-copy.log


#
# --- General name and directory definitions -----------------------------------
#

# The relative and absolute locations of the top-level Windows build directory.
# This is the directory in which nmake is run (not the directory named "build").
TOP_BUILD_DIR_REL = .
TOP_BUILD_DIR_ABS = $(PWD)

# The revision string.
REV_STR           = r$(REVISION)

# The names of the libraries.
LIBBLIS_NAME_ONLY = libblis
LIBBLIS           = $(LIBBLIS_NAME_ONLY)-$(ARCH_STR)-$(REV_STR)

# Directories that reside within the top-level Windows directory.
CNF_DIRNAME       = config
INC_DIRNAME       = include
SRC_DIRNAME       = frame
OBJ_DIRNAME       = obj
LIB_DIRNAME       = lib
DLL_DIRNAME       = dll

# Leaves of interest for Windows.

# Relative directory paths to each of the above subdirectories.
INC_DIRPATH       = $(TOP_BUILD_DIR_REL)\$(INC_DIRNAME)
SRC_DIRPATH       = $(TOP_BUILD_DIR_REL)\$(SRC_DIRNAME)
OBJ_DIRPATH       = $(TOP_BUILD_DIR_REL)\$(OBJ_DIRNAME)
LIB_DIRPATH       = $(TOP_BUILD_DIR_REL)\$(LIB_DIRNAME)
DLL_DIRPATH       = $(TOP_BUILD_DIR_REL)\$(DLL_DIRNAME)

# We only have header files for flamec leaves.
INC_BLI_DIRPATH   = $(INC_DIRPATH)

# We have source code for flamec and lapack2flamec leaves.
SRC_BLI_DIRPATH   = $(SRC_DIRPATH)


# And we have object file paths corresponding to those source leaves defined
# above.
OBJ_BLI_DIRPATH   = $(OBJ_DIRPATH)\$(ARCH_STR)\$(BUILD_STR)

# Separate directories into which we'll move object files when we create the
# static libraries.
LIB_LIBBLIS_DIRPATH = $(LIB_DIRPATH)\$(ARCH_STR)\$(BUILD_STR)

# Separate directories into which we'll move object files when we create the
# dynamic libraries.
DLL_LIBBLIS_DIRPATH = $(DLL_DIRPATH)\$(ARCH_STR)\$(BUILD_STR)

# The install subdirectories.
INSTALL_PREFIX_LIB = $(INSTALL_PREFIX)\libblis\lib
INSTALL_PREFIX_DLL = $(INSTALL_PREFIX)\libblis\dll
INSTALL_PREFIX_INC = $(INSTALL_PREFIX)\libblis\include-$(ARCH_STR)-$(REV_STR)

# Definitions for important header files used in the install-headers rule.
BUILD_DIRNAME      = build
BLIS_H             = blis.h


#
# --- General shell definitions ------------------------------------------------
#

CD     = cd
DIR    = dir
COPY   = copy
DEL    = del /F /Q
MKDIR  = mkdir
RMDIR  = rd /S /Q
ECHO   = echo


#
# --- Helper scripts -----------------------------------------------------------
#

NMAKE_HELP = .\build\nmake-help.cmd



#
# --- Compiler-related definitions ---------------------------------------------
#

#!include $(VERSION_FILE)

# --- C compiler definitions ---

WINDOWS_BUILD = BLIS_ENABLE_WINDOWS_BUILD
VERS_STR      = 0.0.9
VERSION       = BLIS_VERSION_STRING=\"$(VERS_STR)\"

!if "$(CCOMPILER_STR)"=="icl"

!if "$(BUILD_STR)"=="debug"
CDEBUG = /Zi
COPTIM = /Od
!elseif "$(BUILD_STR)"=="release"
CDEBUG =
COPTIM = /Ox
!endif

CC            = icl.exe
CMISCFLAGS    = /nologo
CLANGFLAGS    =
CPPROCFLAGS   = /I.\build /I$(INC_BLI_DIRPATH) /D$(WINDOWS_BUILD) /D$(VERSION)
CWARNFLAGS    = /w
CDBGFLAGS     = $(CDEBUG)
COPTFLAGS     = $(COPTIM)
CRTIMEFLAGS   = /MT
CMTHREADFLAGS = /Qopenmp
CFLAGS        = $(CMISCFLAGS) $(CLANGFLAGS) $(CPPROCFLAGS) $(CWARNFLAGS) \
                $(CDBGFLAGS) $(COPTFLAGS) $(CRTIMEFLAGS) $(CMTHREADFLAGS)

!elseif "$(CCOMPILER_STR)"=="cl"

!if "$(BUILD_STR)"=="debug"
CDEBUG = /Zi
COPTIM = /Od
!elseif "$(BUILD_STR)"=="release"
CDEBUG =
COPTIM = /Ox
!endif

CC            = cl.exe
CMISCFLAGS    = /nologo
CLANGFLAGS    =
CPPROCFLAGS   = /I.\build /I$(INC_BLI_DIRPATH) /D$(WINDOWS_BUILD) /D$(VERSION)
CWARNFLAGS    = /w
CDBGFLAGS     = $(CDEBUG)
COPTFLAGS     = $(COPTIM)
CRTIMEFLAGS   = /MT
CMTHREADFLAGS = /openmp
CFLAGS        = $(CMISCFLAGS) $(CLANGFLAGS) $(CPPROCFLAGS) $(CWARNFLAGS) \
                $(CDBGFLAGS) $(COPTFLAGS) $(CRTIMEFLAGS) $(CMTHREADFLAGS)

!endif



#
# --- Library-related definitions ----------------------------------------------
#

# --- Static library definitions ---

LIBBLIS_LIB          = $(LIBBLIS).lib

LIB                   = lib
LIB_OPTIONS           = /nologo
LIB_BLI_OUTPUT_ARG    = /out:$(LIBBLIS_LIB)
LIB_BLI_INPUT_ARGS    = *.obj

# --- Dynamic library definitions ---

LIBBLIS_DLL          = $(LIBBLIS).dll

GENDLL                = $(TOP_BUILD_DIR_ABS)\gendll.cmd
OBJ_LIST_FILE         = libblis-objects.txt

SYM_DEF_FILEPATH      = $(TOP_BUILD_DIR_ABS)\$(BUILD_DIRNAME)\libblis-symbols.def

