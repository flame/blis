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
# --- Include variables determined at configure-time --------------------------
#
CONFIGURE_DEFS = config\config.mk

!if exist ( $(CONFIGURE_DEFS) )
!include $(CONFIGURE_DEFS)
!else
!error nmake: $(CONFIGURE_DEFS) does not exist! Run configure.cmd first.
!endif



#
# --- Include environment- and build-specific definitions ----------------------
#

MAKE_DEFS = build\defs.mk

# Include build definitions
!if exist ( $(MAKE_DEFS) )
!include $(MAKE_DEFS)
!else
!error nmake: $(MAKE_DEFS) does not exist! Your libblis distribution may be incomplete.
!endif



#
# --- Variable modifications ---------------------------------------------------
#



#
# --- High-level rules ---------------------------------------------------------
#

all: libblis

libblis: libblis-lib

libblis-objs: $(BLIS_OBJS)

libblis-lib: $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS_LIB)

libblis-dll: $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS_DLL)

lib: libblis-lib

dll: libblis-dll

install: install-lib install-headers

install-lib: $(INSTALL_PREFIX_LIB)\$(LIBBLIS).lib

install-dll: $(INSTALL_PREFIX_DLL)\$(LIBBLIS).dll \
             $(INSTALL_PREFIX_DLL)\$(LIBBLIS).lib \
             $(INSTALL_PREFIX_DLL)\$(LIBBLIS).exp

install-headers: $(INSTALL_PREFIX_INC)\$(BLIS_H)

clean: clean-build clean-log

distclean: clean-config clean-build clean-log



#
# --- Source code (inference) rules --------------------------------------------
#

# --- C source files in flamec directory ---
{$(SRC_BLI_DIRPATH)}.c{$(OBJ_BLI_DIRPATH)}.obj:
!ifdef VERBOSE
	if not exist $(OBJ_BLI_DIRPATH) \
	   ( $(MKDIR) $(OBJ_BLI_DIRPATH) )
	$(CC) $(CFLAGS) /c $< /Fo$@
!else
	@if not exist $(OBJ_BLI_DIRPATH) \
	 (  ( $(ECHO) nmake: Creating $(OBJ_BLI_DIRPATH) directory ) & \
	    ( $(MKDIR) $(OBJ_BLI_DIRPATH) ) )
	@$(ECHO) nmake: Compiling $<
	@$(CC) $(CFLAGS) /c $< /Fo$@ >> $(CC_LOG_FILE)
!endif



#
# --- Library generation rules -------------------------------------------------
#

# --- Static library ---
$(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS_LIB): libblis-objs
!ifdef VERBOSE
	if not exist $(LIB_LIBBLIS_DIRPATH) \
	   ( $(MKDIR) $(LIB_LIBBLIS_DIRPATH) )
	$(COPY) $(OBJ_BLI_DIRPATH)\*.obj $(LIB_LIBBLIS_DIRPATH)
	$(CD) $(LIB_LIBBLIS_DIRPATH)
	$(LIB) $(LIB_OPTIONS) $(LIB_BLI_OUTPUT_ARG) $(LIB_BLI_INPUT_ARGS)
	$(DEL) *.obj
	$(CD) $(TOP_BUILD_DIR_ABS)
!else
	@if not exist $(LIB_LIBBLIS_DIRPATH) \
	 (  ( $(ECHO) nmake: Creating $(LIB_LIBBLIS_DIRPATH) directory ) & \
	    ( $(MKDIR) $(LIB_LIBBLIS_DIRPATH) ) )
	@$(ECHO) nmake: Creating static library $@
	@$(COPY) $(OBJ_BLI_DIRPATH)\*.obj $(LIB_LIBBLIS_DIRPATH) >> $(COPY_LOG_FILE)
	@$(CD) $(LIB_LIBBLIS_DIRPATH)
	@$(LIB) /VERBOSE $(LIB_OPTIONS) $(LIB_BLI_OUTPUT_ARG) $(LIB_BLI_INPUT_ARGS)
	@$(DEL) *.obj
	@$(CD) $(TOP_BUILD_DIR_ABS)
!endif

# --- Dynamic library (object code file, import library, and export file) ---
$(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS_DLL): libblis-objs
!ifdef VERBOSE
	if not exist $(DLL_LIBBLIS_DIRPATH) \
	   ( $(MKDIR) $(DLL_LIBBLIS_DIRPATH) )
	$(COPY) $(OBJ_BLI_DIRPATH)\*.obj $(DLL_LIBBLIS_DIRPATH) >> $(COPY_LOG_FILE)
	$(CD) $(DLL_LIBBLIS_DIRPATH)
	$(DIR) /B *.obj > $(OBJ_LIST_FILE)
	$(GENDLL) $(LIBBLIS) $(LIBBLIS) $(CC) $(LINKARGS_FILEPATH) $(SYM_DEF_FILEPATH) /objlist $(OBJ_LIST_FILE)
	$(DEL) $(OBJ_LIST_FILE)
	$(DEL) *.obj
	$(CD) $(TOP_BUILD_DIR_ABS)
!else
	@if not exist $(DLL_LIBBLIS_DIRPATH) \
	 (  ( $(ECHO) nmake: Creating $(DLL_LIBBLIS_DIRPATH) directory ) & \
	    ( $(MKDIR) $(DLL_LIBBLIS_DIRPATH) ) )
	@$(ECHO) nmake: Creating dynamic library $@
	@$(COPY) $(OBJ_BLI_DIRPATH)\*.obj $(DLL_LIBBLIS_DIRPATH) >> $(COPY_LOG_FILE)
	@$(CD) $(DLL_LIBBLIS_DIRPATH)
	@$(DIR) /B *.obj > $(OBJ_LIST_FILE)
	@$(GENDLL) $(LIBBLIS) $(LIBBLIS) $(CC) $(LINKARGS_FILEPATH) $(SYM_DEF_FILEPATH) /objlist $(OBJ_LIST_FILE)
	@$(DEL) $(OBJ_LIST_FILE)
	@$(DEL) *.obj
	@$(CD) $(TOP_BUILD_DIR_ABS)
!endif



#
# --- Install rules ------------------------------------------------------------
#

# --- Header files ---
$(INSTALL_PREFIX_INC)\$(BLIS_H): $(INC_BLI_DIRPATH)\$(BLIS_H) \
                                  $(BUILD_DIRNAME)\$(BLI_CONFIG_H)
!ifdef VERBOSE
	if not exist $(INSTALL_PREFIX_INC) \
	   ( $(MKDIR) $(INSTALL_PREFIX_INC) )
    $(COPY) $(BUILD_DIRNAME)\$(BLI_CONFIG_H) $(INSTALL_PREFIX_INC) >> $(COPY_LOG_FILE)
    $(COPY) $(INC_BLI_DIRPATH)\*.h $(INSTALL_PREFIX_INC) >> $(COPY_LOG_FILE)
!else
	@if not exist $(INSTALL_PREFIX_INC) \
	    ( $(MKDIR) $(INSTALL_PREFIX_INC) )
    @$(ECHO) nmake: Installing libblis header files to $(INSTALL_PREFIX_INC)
    @$(COPY) $(BUILD_DIRNAME)\$(BLI_CONFIG_H) $(INSTALL_PREFIX_INC) >> $(COPY_LOG_FILE)
    @$(COPY) $(INC_BLI_DIRPATH)\*.h $(INSTALL_PREFIX_INC) >> $(COPY_LOG_FILE)
!endif

# --- Static library ---
$(INSTALL_PREFIX_LIB)\$(LIBBLIS).lib: $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib
!ifdef VERBOSE
	if not exist $(INSTALL_PREFIX_LIB) ( $(MKDIR) $(INSTALL_PREFIX_LIB) )
	if     exist $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib \
	   ( $(COPY) $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib $(INSTALL_PREFIX_LIB) >> $(COPY_LOG_FILE) )
!else
	@if not exist $(INSTALL_PREFIX_LIB) ( $(MKDIR) $(INSTALL_PREFIX_LIB) )
	@if     exist $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib \
	 (  ( $(ECHO) nmake: Installing $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib to $(INSTALL_PREFIX_LIB) ) & \
	    ( $(COPY) $(LIB_LIBBLIS_DIRPATH)\$(LIBBLIS).lib $(INSTALL_PREFIX_LIB) >> $(COPY_LOG_FILE) ) )
!endif

# --- Dynamic library (object code) ---
$(INSTALL_PREFIX_DLL)\$(LIBBLIS).dll: $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll
!ifdef VERBOSE
	if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll \
	   ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) )
!else
	@if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	@if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll \
	 (  ( $(ECHO) nmake: Installing $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll to $(INSTALL_PREFIX_DLL) ) & \
	    ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).dll $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) ) )
!endif

# --- Dynamic library (import library) ---
$(INSTALL_PREFIX_DLL)\$(LIBBLIS).lib: $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib
!ifdef VERBOSE
	if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib \
	   ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) )
!else
	@if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	@if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib \
	 (  ( $(ECHO) nmake: Installing $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib to $(INSTALL_PREFIX_DLL) ) & \
	    ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).lib $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) ) )
!endif

# --- Dynamic library (export file) ---
$(INSTALL_PREFIX_DLL)\$(LIBBLIS).exp: $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp
!ifdef VERBOSE
	if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp \
	   ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) )
!else
	@if not exist $(INSTALL_PREFIX_DLL) ( $(MKDIR) $(INSTALL_PREFIX_DLL) )
	@if     exist $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp \
	 (  ( $(ECHO) nmake: Installing $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp to $(INSTALL_PREFIX_DLL) ) & \
	    ( $(COPY) $(DLL_LIBBLIS_DIRPATH)\$(LIBBLIS).exp $(INSTALL_PREFIX_DLL) >> $(COPY_LOG_FILE) ) )
!endif



#
# --- Clean rules --------------------------------------------------------------
#

clean-log:
!ifdef VERBOSE
	if exist $(CC_LOG_FILE) \
	   ( $(DEL) $(CC_LOG_FILE) )
	if exist $(FC_LOG_FILE) \
	   ( $(DEL) $(FC_LOG_FILE) )
	if exist $(COPY_LOG_FILE) \
	   ( $(DEL) $(COPY_LOG_FILE) )
!else
	@if exist $(CC_LOG_FILE) \
	 (  ( $(ECHO) nmake: Deleting $(CC_LOG_FILE) ) & \
	    ( $(DEL) $(CC_LOG_FILE) ) )
	@if exist $(FC_LOG_FILE) \
	 (  ( $(ECHO) nmake: Deleting $(FC_LOG_FILE) ) & \
	    ( $(DEL) $(FC_LOG_FILE) ) )
	@if exist $(COPY_LOG_FILE) \
	 (  ( $(ECHO) nmake: Deleting $(COPY_LOG_FILE) ) & \
	    ( $(DEL) $(COPY_LOG_FILE) ) )
!endif

clean-config:
!ifdef VERBOSE
	if exist $(CNF_DIRNAME) \
	   ( $(RMDIR) $(CNF_DIRNAME) )
	if exist $(INC_DIRNAME) \
	   ( $(RMDIR) $(INC_DIRNAME) )
	if exist $(SRC_DIRNAME) \
	   ( $(RMDIR) $(SRC_DIRNAME) )
!else
	@if exist $(CNF_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(CNF_DIRNAME) directory ) & \
	    ( $(RMDIR) $(CNF_DIRNAME) ) )
	@if exist $(INC_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(INC_DIRNAME) directory ) & \
	    ( $(RMDIR) $(INC_DIRNAME) ) )
	@if exist $(SRC_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(SRC_DIRNAME) directory ) & \
	    ( $(RMDIR) $(SRC_DIRNAME) ) )
!endif

clean-build:
!ifdef VERBOSE
	if exist $(OBJ_DIRNAME) \
	   ( $(RMDIR) $(OBJ_DIRNAME) )
	if exist $(LIB_DIRNAME) \
	   ( $(RMDIR) $(LIB_DIRNAME) )
	if exist $(DLL_DIRNAME) \
	   ( $(RMDIR) $(DLL_DIRNAME) )
!else
	@if exist $(OBJ_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(OBJ_DIRNAME) directory ) & \
	    ( $(RMDIR) $(OBJ_DIRNAME) ) )
	@if exist $(LIB_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(LIB_DIRNAME) directory ) & \
	    ( $(RMDIR) $(LIB_DIRNAME) ) )
	@if exist $(DLL_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(DLL_DIRNAME) directory ) & \
	    ( $(RMDIR) $(DLL_DIRNAME) ) )
!endif

# Useful for developing when all we want to do is remove the library products.
clean-lib:
!ifdef VERBOSE
	if exist $(LIB_DIRNAME) \
	   ( $(RMDIR) $(LIB_DIRNAME) )
	if exist $(DLL_DIRNAME) \
	   ( $(RMDIR) $(DLL_DIRNAME) )
!else
	@if exist $(LIB_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(LIB_DIRNAME) directory ) & \
	    ( $(RMDIR) $(LIB_DIRNAME) ) )
	@if exist $(DLL_DIRNAME) \
	 (  ( $(ECHO) nmake: Deleting $(DLL_DIRNAME) directory ) & \
	    ( $(RMDIR) $(DLL_DIRNAME) ) )
!endif



#
# --- Help target --------------------------------------------------------------
#

help:
	@$(NMAKE_HELP)

