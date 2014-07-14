::
::
::  BLIS    
::  An object-based framework for developing high-performance BLAS-like
::  libraries.
::
::  Copyright (C) 2014, The University of Texas at Austin
::
::  Redistribution and use in source and binary forms, with or without
::  modification, are permitted provided that the following conditions are
::  met:
::   - Redistributions of source code must retain the above copyright
::     notice, this list of conditions and the following disclaimer.
::   - Redistributions in binary form must reproduce the above copyright
::     notice, this list of conditions and the following disclaimer in the
::     documentation and/or other materials provided with the distribution.
::   - Neither the name of The University of Texas at Austin nor the names
::     of its contributors may be used to endorse or promote products
::     derived from this software without specific prior written permission.
::
::  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
::  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
::  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
::  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
::  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
::  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
::  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
::  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
::  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
::  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
::  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
::
::

@echo off

:ENVIRONMENT
	set GEN_CHECK_REV_FILE=.\build\gen-check-rev-file.py
	set GATHER_SRC=.\build\gather-src-for-windows.py
	set GEN_CONFIG_FILE=.\build\gen-config-file.py
	set CONFIG_DEFS_TEMPL=.\build\config.mk.in
	set SRC_TREE_DIR=..\frame
	set TOP_BUILD_DIR=.

:PARAMS
	if "%1"=="" (goto USAGE)
	if "%2"=="" (goto USAGE)
	if "%3"=="" (goto USAGE)

	set ARCH=%1
	set BUILD=%2
	set CCOMPILER=%3
	
:TASK_UNIT
	echo %0: Checking/updating revision file.
	%GEN_CHECK_REV_FILE% -v
	echo %0: Gathering source files into local flat directories.
	%GATHER_SRC% %SRC_TREE_DIR% %TOP_BUILD_DIR%
	echo %0: Creating configure definitions file.
	%GEN_CONFIG_FILE% %TOP_BUILD_DIR% %ARCH% %BUILD% %CCOMPILER% %CONFIG_DEFS_TEMPL%
	echo %0: Configuration and setup complete. You may now run nmake. 

	goto END

:USAGE
	echo. 
	echo  configure.cmd
	echo. 
	echo  A wrapper script for various configuration and setup scripts that need
	echo. to be run before nmake when building BLIS for Microsoft Windows.
	echo. 
	echo  USAGE:
	echo     %0 [arch] [build] [cc]
	echo.
	echo        arch     -- The architecture string to build.
	echo                    Supported values: {x86,x64}
	echo        build    -- The kind of build.
	echo                    Supported values: {debug,release}
	echo        cc       -- The C compiler to use.
	echo                    Supported values: {icl,cl}
	echo. 
	echo  examples:
	echo     %0 x86 debug icl
	echo     %0 x64 release cl
	echo.

:END
