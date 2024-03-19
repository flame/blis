/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include @PLUGIN_HEADER@

err_t PASTEMAC(plugin_register,BLIS_PNAME_INFIX)
     (
       PASTECH(plugin,BLIS_PNAME_INFIX,_params)
     )
{
	// ------------------------------------------------------------------------>
	// -- Example Plugin Registration  ---------------------------------------->
	// ------------------------------------------------------------------------>

	//
	// Register slots for new microkernels, preferences, and block sizes.
	//

	err_t err;

	err = bli_gks_register_blksz( &bszids[ MY_BLKSZ_1 ] );
	err = bli_gks_register_blksz( &bszids[ MY_BLKSZ_1 ] );
	err = bli_gks_register_ukr( &kerids[ MY_KERNEL_1 ] );
	err = bli_gks_register_ukr( &kerids[ MY_KERNEL_2 ] );
	err = bli_gks_register_ukr_pref( &prefids[ MY_PREF_1 ] );
	err = bli_gks_register_ukr_pref( &prefids[ MY_PREF_2 ] );

	if ( err != BLIS_SUCCESS )
		return err;

	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------

	//
	// Initialize the context for each enabled sub-configuration.
	//

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	PASTEMAC(plugin_init,BLIS_PNAME_INFIX,_,config,BLIS_REF_SUFFIX) \
	( \
	  PASTECH(plugin,BLIS_PNAME_INFIX,_params_only) \
	);

	INSERT_GENTCONF

	return BLIS_SUCCESS;
}

