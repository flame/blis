#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2019, Advanced Micro Devices, Inc.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
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

# FLAGS that are specific to 'zen2' architecture are added here.
# FLAGS that are common for all the AMD architectures are present in config/zen/amd_config.mk
#

# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := zen2
#CONFIGS_INCL   += $(THIS_CONFIG)

# Include file containing common flags for all AMD architectures
AMD_CONFIG_FILE := amd_config.mk
AMD_CONFIG_PATH := $(BASE_SHARE_PATH)/config/zen
-include $(AMD_CONFIG_PATH)/$(AMD_CONFIG_FILE)
#
# --- Determine the C compiler and related flags ---
#
# Flags specific to optimized kernels.
ifeq ($(CC_VENDOR),gcc)
# gcc 9.0 (clang ?) or later:
GCC_VERSION := $(strip $(shell gcc -dumpversion))
ifeq ($(shell test $(GCC_VERSION) -ge 9; echo $$?),0)
CKVECFLAGS     += -march=znver2
# gcc 6.0 (clang 4.0) or later:
else
CKVECFLAGS     += -march=znver1 -mno-avx256-split-unaligned-store
endif
# gcc 4.9 (clang 3.5) or later:
# possibly add zen-specific instructions: -mclzero -madx -mrdseed -mmwaitx -msha -mxsavec -mxsaves -mclflushopt -mpopcnt
#CKVECFLAGS     := -mavx2 -mfpmath=sse -mfma -march=bdver4 -mno-fma4 -mno-tbm -mno-xop -mno-lwp
endif
# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))

