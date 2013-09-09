@echo off
@setlocal enabledelayedexpansion

rem --------------------------------------------------------------------
rem Build a dll out of a set of object files specified by the 
rem argument /objlist.
rem
rem The .lib file thus created is an "import" library, which one links
rem with, but the bulk of the code ends up in the associated .dll file.
rem ---------------------------------------------------------------------

set THIS_SCRIPT=%~dp0%~nx0

if "%1"=="" goto USAGE
if "%2"=="" goto USAGE
if "%3"=="" goto USAGE
if "%4"=="" goto USAGE
if "%5"=="" goto USAGE

set gd_lib_name=%1
set gd_link=%gd_lib_name%-static.link
set LINKER=%3
set LINKARGSFILE=%4
set gd_def=%5

:PARSE_ARGS
set IMPORT=
set OBJLIST=
:ARGLOOP
if "%6"=="" goto ENDARGLOOP
if /i not "%6"=="/import" goto OBJARG
set IMPORT=!IMPORT! %7
goto SHIFT
:OBJARG
if /i not "%6"=="/objlist" goto ENDARGLOOP
set OBJLIST=%7
:SHIFT
shift /4
shift /4
goto ARGLOOP
:ENDARGLOOP

if defined OBJLIST goto COMPILER_SETUP
echo Error: must supply /objlist <file with list of object names>
goto USAGE

:COMPILER_SETUP
set gd_path=%2
set gd_dll_path=%gd_path%.dll
set gd_main_c=dll_main__%gd_lib_name%.c
set gd_main_obj=dll_main__%gd_lib_name%.obj

rem create C file for dll_main
for /F "tokens=*" %%i in ("#include <windows.h>") do echo %%i >%gd_main_c%
echo. >>%gd_main_c%
echo BOOLEAN WINAPI DllMain( >>%gd_main_c%
echo 	HINSTANCE hDllHandle, >>%gd_main_c%
echo 	DWORD     nReason,    >>%gd_main_c%
echo 	LPVOID    Reserved){  >>%gd_main_c%
echo.                        >>%gd_main_c%
echo BOOLEAN bSuccess = TRUE;>>%gd_main_c%
echo.                        >>%gd_main_c%
echo	switch (nReason){     >>%gd_main_c%
echo		case DLL_PROCESS_ATTACH: >>%gd_main_c%
echo			DisableThreadLibraryCalls( hDllHandle ); >>%gd_main_c%
echo		break; >>%gd_main_c%
echo		case DLL_PROCESS_DETACH: >>%gd_main_c%
echo		break; >>%gd_main_c%
echo.            >>%gd_main_c%
echo	}; >>%gd_main_c%
echo.   >>%gd_main_c%
echo	return bSuccess; >>%gd_main_c%
echo }; >>%gd_main_c%
echo.>>%gd_main_c%

rem set up link file by specifying dll filepath and main object
echo /Fe%gd_dll_path% > %gd_link%
echo %gd_main_obj% >> %gd_link%

rem add contents of linkargs file; most of the link argument action is
rem in this file
type %LINKARGSFILE% >> %gd_link%

rem add command-line import libraries, if any
if defined IMPORT echo !IMPORT! >> %gd_link%

rem add export specification
echo %gd_def% >> %gd_link%

rem add contents of OBJLIST file
type %OBJLIST% >> %gd_link%

rem create dll, import lib, and export file
%LINKER% /nologo /c /O2 /Fo%gd_main_obj% %gd_main_c% >> gendll-cl.log
%LINKER% @%gd_link%

:CLEANUP
del /F /Q %gd_link% %gd_main_c% %gd_main_obj% gendll-cl.log
goto END


:USAGE
echo. 
echo. gendll.cmd
echo. 
echo. Generate a dynamically-linked library from a set of object files
echo. specified in objlist_file.
echo. 
echo. Usage:
echo.   %0 dllname dllpath linker linkargs_file symbols_file {/import importlib} /objlist objlist_file
echo.
echo.     dllname       -- the name of the DLL being created, with no extension.
echo.     dllpath       -- the path to the DLL being created, with no extension.
echo.     linker        -- the compiler to use to link the DLL.
echo.     linkargs_file -- the path to a file containing a list of all linker
echo.                      arguments--link options, libraries, and library paths--
echo.                      that that may be needed to successfully link the DLL
echo.                      being created.
echo.     symbols_file  -- the path to a file containing a list of symbols to
echo.                      export in the DLL.
echo.     importlib     -- the path to a .lib library that you wish to import into
echo.                      the DLL being created. Optional.
echo.     objlist_file  -- the path to a file containing the list of object files
echo.                      that make up the bulk of the DLL being created.
echo.

:END
endlocal
