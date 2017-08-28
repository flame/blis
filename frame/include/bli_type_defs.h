/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#ifndef BLIS_TYPE_DEFS_H
#define BLIS_TYPE_DEFS_H


//
// -- BLIS basic types ---------------------------------------------------------
//

#ifdef __cplusplus
  // For C++, include stdint.h.
  #include <stdint.h>
#elif __STDC_VERSION__ >= 199901L
  // For C99 (or later), include stdint.h.
  #include <stdint.h>
#else
  // When stdint.h is not available, manually typedef the types we will use.
  #ifdef _WIN32
    typedef          __int32  int32_t;
    typedef unsigned __int32 uint32_t;
    typedef          __int64  int64_t;
    typedef unsigned __int64 uint64_t;
  #else
    #error "Attempting to compile on pre-C99 system without stdint.h."
  #endif
#endif

// -- General-purpose integers --

// Define integer types depending on what size integer was requested.
#if   BLIS_INT_TYPE_SIZE == 32
typedef           int32_t  gint_t;
typedef          uint32_t guint_t;
#elif BLIS_INT_TYPE_SIZE == 64
typedef           int64_t  gint_t;
typedef          uint64_t guint_t;
#else
typedef   signed long int  gint_t;
typedef unsigned long int guint_t;
#endif

// -- Boolean type --

typedef  gint_t  bool_t;

// -- Special-purpose integers --

// This cpp guard provides a temporary hack to allow libflame
// interoperability with BLIS.
#ifndef _DEFINED_DIM_T
#define _DEFINED_DIM_T
typedef  gint_t  dim_t;      // dimension type
#endif
typedef  gint_t  inc_t;      // increment/stride type
typedef  gint_t  doff_t;     // diagonal offset type
typedef guint_t  siz_t;      // byte size type
typedef guint_t  objbits_t;  // object information bit field

// -- Real types --

// Define the number of floating-point types supported, and the size of the
// largest type.
#define BLIS_NUM_FP_TYPES   4
#define BLIS_MAX_TYPE_SIZE  sizeof(dcomplex)

// There are some places where we need to use sizeof() inside of a C
// preprocessor #if conditional, and so here we define the various sizes
// for those purposes.
#define BLIS_SIZEOF_S      4  // sizeof(float)
#define BLIS_SIZEOF_D      8  // sizeof(double)
#define BLIS_SIZEOF_C      8  // sizeof(scomplex)
#define BLIS_SIZEOF_Z      16 // sizeof(dcomplex)

// -- Complex types --

#ifdef BLIS_ENABLE_C99_COMPLEX

	#if __STDC_VERSION__ >= 199901L
		#include <complex.h>

		// Typedef official complex types to BLIS complex type names.
		typedef  float complex scomplex;
		typedef double complex dcomplex;
	#else
		#error "Configuration requested C99 complex types, but C99 does not appear to be supported."
	#endif

#else // ifndef BLIS_ENABLE_C99_COMPLEX

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_SCOMPLEX
	#define _DEFINED_SCOMPLEX
	typedef struct
	{
		float  real;
		float  imag;
	} scomplex;
	#endif

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_DCOMPLEX
	#define _DEFINED_DCOMPLEX
	typedef struct
	{
		double real;
		double imag;
	} dcomplex;
	#endif

#endif // BLIS_ENABLE_C99_COMPLEX

// -- Atom type --

// Note: atom types are used to hold "bufferless" scalar object values. Note
// that it needs to be as large as the largest possible scalar value we might
// want to hold. Thus, for now, it is a dcomplex.
typedef dcomplex atom_t;

// -- Fortran-77 types --

// Note: These types are typically only used by BLAS compatibility layer, but
// we must define them even when the compatibility layer isn't being built
// because they also occur in bli_slamch() and bli_dlamch().

// Define f77_int depending on what size of integer was requested.
#if   BLIS_BLAS2BLIS_INT_TYPE_SIZE == 32
typedef int32_t   f77_int;
#elif BLIS_BLAS2BLIS_INT_TYPE_SIZE == 64
typedef int64_t   f77_int;
#else
typedef long int  f77_int;
#endif

typedef char      f77_char;
typedef float     f77_float;
typedef double    f77_double;
typedef scomplex  f77_scomplex;
typedef dcomplex  f77_dcomplex;


//
// -- BLIS info bit field offsets ----------------------------------------------
//

/*
  info field description

  bit(s)   purpose
  -------  -------
   2 ~ 0   Stored numerical datatype
           - 0: domain    (0 == real, 1 == complex)
           - 1: precision (0 == single, 1 == double)
           - 2: special   (100 = int; 101 = const)
       3   Transposition required [during pack]?
       4   Conjugation required [during pack]?
   7 ~ 5   Part of matrix stored:
           - 5: strictly upper triangular
           - 6: diagonal
           - 7: strictly lower triangular
       8   Implicit unit diagonal?
       9   Invert diagonal required [during pack]?
  12 ~ 10  Target numerical datatype
           - 10: domain    (0 == real, 1 == complex)
           - 11: precision (0 == single, 1 == double)
           - 12: unused
  15 ~ 13  Execution numerical datatype
           - 13: domain    (0 == real, 1 == complex)
           - 14: precision (0 == single, 1 == double)
           - 15: unused
  22 ~ 16  Packed type/status
           - 0 0000 00: not packed
           - 1 0000 00: packed (unspecified; by rows, columns, or vector)
           - 1 0000 00: packed by rows
           - 1 0000 01: packed by columns
           - 1 0000 10: packed by row panels
           - 1 0000 11: packed by column panels
           - 1 0001 10: packed by 4m interleaved row panels
           - 1 0001 11: packed by 4m interleaved column panels
           - 1 0010 10: packed by 3m interleaved row panels
           - 1 0010 11: packed by 3m interleaved column panels
           - 1 0011 10: packed by 4m separated row panels (not used)
           - 1 0011 11: packed by 4m separated column panels (not used)
           - 1 0100 10: packed by 3m separated row panels
           - 1 0100 11: packed by 3m separated column panels
           - 1 0101 10: packed real-only row panels
           - 1 0101 11: packed real-only column panels
           - 1 0110 10: packed imag-only row panels
           - 1 0110 11: packed imag-only column panels
           - 1 0111 10: packed real+imag row panels
           - 1 0111 11: packed real+imag column panels
           - 1 1000 10: packed by 1m expanded row panels
           - 1 1000 11: packed by 1m expanded column panels
           - 1 1001 10: packed by 1m reordered row panels
           - 1 1001 11: packed by 1m reordered column panels
       23  Packed panel order if upper-stored
           - 0 == forward order if upper
           - 1 == reverse order if upper
       24  Packed panel order if lower-stored
           - 0 == forward order if lower
           - 1 == reverse order if lower
  26 ~ 25  Packed buffer type
           - 0 == block of A
           - 1 == panel of B
           - 2 == panel of C
           - 3 == general use
  28 ~ 27  Structure type
           - 0 == general
           - 1 == Hermitian
           - 2 == symmetric
           - 3 == triangular
*/

#define BLIS_DATATYPE_SHIFT                0
#define   BLIS_DOMAIN_SHIFT                0
#define   BLIS_PRECISION_SHIFT             1
#define BLIS_CONJTRANS_SHIFT               3
#define   BLIS_TRANS_SHIFT                 3
#define   BLIS_CONJ_SHIFT                  4
#define BLIS_UPLO_SHIFT                    5
#define   BLIS_UPPER_SHIFT                 5
#define   BLIS_DIAG_SHIFT                  6
#define   BLIS_LOWER_SHIFT                 7
#define BLIS_UNIT_DIAG_SHIFT               8
#define BLIS_INVERT_DIAG_SHIFT             9
#define BLIS_TARGET_DT_SHIFT               10
#define BLIS_EXECUTION_DT_SHIFT            13
#define BLIS_PACK_SCHEMA_SHIFT             16
#define   BLIS_PACK_RC_SHIFT               16
#define   BLIS_PACK_PANEL_SHIFT            17
#define   BLIS_PACK_FORMAT_SHIFT           18
#define   BLIS_PACK_SHIFT                  22
#define BLIS_PACK_REV_IF_UPPER_SHIFT       23
#define BLIS_PACK_REV_IF_LOWER_SHIFT       24
#define BLIS_PACK_BUFFER_SHIFT             25
#define BLIS_STRUC_SHIFT                   27

//
// -- BLIS info bit field masks ------------------------------------------------
//

#define BLIS_DATATYPE_BITS                 ( 0x7  << BLIS_DATATYPE_SHIFT )
#define   BLIS_DOMAIN_BIT                  ( 0x1  << BLIS_DOMAIN_SHIFT )
#define   BLIS_PRECISION_BIT               ( 0x1  << BLIS_PRECISION_SHIFT )
#define BLIS_CONJTRANS_BITS                ( 0x3  << BLIS_CONJTRANS_SHIFT )
#define   BLIS_TRANS_BIT                   ( 0x1  << BLIS_TRANS_SHIFT )
#define   BLIS_CONJ_BIT                    ( 0x1  << BLIS_CONJ_SHIFT )
#define BLIS_UPLO_BITS                     ( 0x7  << BLIS_UPLO_SHIFT )
#define   BLIS_UPPER_BIT                   ( 0x1  << BLIS_UPPER_SHIFT )
#define   BLIS_DIAG_BIT                    ( 0x1  << BLIS_DIAG_SHIFT )
#define   BLIS_LOWER_BIT                   ( 0x1  << BLIS_LOWER_SHIFT )
#define BLIS_UNIT_DIAG_BIT                 ( 0x1  << BLIS_UNIT_DIAG_SHIFT )
#define BLIS_INVERT_DIAG_BIT               ( 0x1  << BLIS_INVERT_DIAG_SHIFT )
#define BLIS_TARGET_DT_BITS                ( 0x7  << BLIS_TARGET_DT_SHIFT )
#define BLIS_EXECUTION_DT_BITS             ( 0x7  << BLIS_EXECUTION_DT_SHIFT )
#define BLIS_PACK_SCHEMA_BITS              ( 0x7F << BLIS_PACK_SCHEMA_SHIFT )
#define   BLIS_PACK_RC_BIT                 ( 0x1  << BLIS_PACK_RC_SHIFT )
#define   BLIS_PACK_PANEL_BIT              ( 0x1  << BLIS_PACK_PANEL_SHIFT )
#define   BLIS_PACK_FORMAT_BITS            ( 0xF  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_PACK_BIT                    ( 0x1  << BLIS_PACK_SHIFT )
#define BLIS_PACK_REV_IF_UPPER_BIT         ( 0x1  << BLIS_PACK_REV_IF_UPPER_SHIFT )
#define BLIS_PACK_REV_IF_LOWER_BIT         ( 0x1  << BLIS_PACK_REV_IF_LOWER_SHIFT )
#define BLIS_PACK_BUFFER_BITS              ( 0x3  << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_STRUC_BITS                    ( 0x3  << BLIS_STRUC_SHIFT )


//
// -- BLIS enumerated type value definitions -----------------------------------
//

#define BLIS_BITVAL_REAL                      0x0
#define BLIS_BITVAL_COMPLEX                   BLIS_DOMAIN_BIT
#define BLIS_BITVAL_SINGLE_PREC               0x0
#define BLIS_BITVAL_DOUBLE_PREC               BLIS_PRECISION_BIT
#define   BLIS_BITVAL_FLOAT_TYPE              0x0
#define   BLIS_BITVAL_SCOMPLEX_TYPE           BLIS_DOMAIN_BIT  
#define   BLIS_BITVAL_DOUBLE_TYPE             BLIS_PRECISION_BIT
#define   BLIS_BITVAL_DCOMPLEX_TYPE         ( BLIS_DOMAIN_BIT | BLIS_PRECISION_BIT )
#define   BLIS_BITVAL_INT_TYPE                0x04
#define   BLIS_BITVAL_CONST_TYPE              0x05
#define BLIS_BITVAL_NO_TRANS                  0x0
#define BLIS_BITVAL_TRANS                     BLIS_TRANS_BIT
#define BLIS_BITVAL_NO_CONJ                   0x0
#define BLIS_BITVAL_CONJ                      BLIS_CONJ_BIT
#define BLIS_BITVAL_CONJ_TRANS              ( BLIS_CONJ_BIT | BLIS_TRANS_BIT )
#define BLIS_BITVAL_ZEROS                     0x0 
#define BLIS_BITVAL_UPPER                   ( BLIS_UPPER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_LOWER                   ( BLIS_LOWER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_DENSE                     BLIS_UPLO_BITS  
#define BLIS_BITVAL_NONUNIT_DIAG              0x0
#define BLIS_BITVAL_UNIT_DIAG                 BLIS_UNIT_DIAG_BIT
#define BLIS_BITVAL_INVERT_DIAG               BLIS_INVERT_DIAG_BIT
#define BLIS_BITVAL_NOT_PACKED                0x0
#define   BLIS_BITVAL_4MI                   ( 0x1  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_3MI                   ( 0x2  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_4MS                   ( 0x3  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_3MS                   ( 0x4  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_RO                    ( 0x5  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_IO                    ( 0x6  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_RPI                   ( 0x7  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_1E                    ( 0x8  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_1R                    ( 0x9  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_PACKED_UNSPEC         ( BLIS_PACK_BIT                                                            )
#define   BLIS_BITVAL_PACKED_ROWS           ( BLIS_PACK_BIT                                                            )
#define   BLIS_BITVAL_PACKED_COLUMNS        ( BLIS_PACK_BIT                                         | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS     ( BLIS_PACK_BIT                   | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS     ( BLIS_PACK_BIT                   | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_4MI ( BLIS_PACK_BIT | BLIS_BITVAL_4MI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_4MI ( BLIS_PACK_BIT | BLIS_BITVAL_4MI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_3MI ( BLIS_PACK_BIT | BLIS_BITVAL_3MI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_3MI ( BLIS_PACK_BIT | BLIS_BITVAL_3MI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_4MS ( BLIS_PACK_BIT | BLIS_BITVAL_4MS | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_4MS ( BLIS_PACK_BIT | BLIS_BITVAL_4MS | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_3MS ( BLIS_PACK_BIT | BLIS_BITVAL_3MS | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_3MS ( BLIS_PACK_BIT | BLIS_BITVAL_3MS | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_RO  ( BLIS_PACK_BIT | BLIS_BITVAL_RO  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_RO  ( BLIS_PACK_BIT | BLIS_BITVAL_RO  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_IO  ( BLIS_PACK_BIT | BLIS_BITVAL_IO  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_IO  ( BLIS_PACK_BIT | BLIS_BITVAL_IO  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_RPI ( BLIS_PACK_BIT | BLIS_BITVAL_RPI | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_RPI ( BLIS_PACK_BIT | BLIS_BITVAL_RPI | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_1E  ( BLIS_PACK_BIT | BLIS_BITVAL_1E  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_1E  ( BLIS_PACK_BIT | BLIS_BITVAL_1E  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define   BLIS_BITVAL_PACKED_ROW_PANELS_1R  ( BLIS_PACK_BIT | BLIS_BITVAL_1R  | BLIS_PACK_PANEL_BIT                    )
#define   BLIS_BITVAL_PACKED_COL_PANELS_1R  ( BLIS_PACK_BIT | BLIS_BITVAL_1R  | BLIS_PACK_PANEL_BIT | BLIS_PACK_RC_BIT )
#define BLIS_BITVAL_PACK_FWD_IF_UPPER         0x0
#define BLIS_BITVAL_PACK_REV_IF_UPPER         BLIS_PACK_REV_IF_UPPER_BIT
#define BLIS_BITVAL_PACK_FWD_IF_LOWER         0x0
#define BLIS_BITVAL_PACK_REV_IF_LOWER         BLIS_PACK_REV_IF_LOWER_BIT
#define BLIS_BITVAL_BUFFER_FOR_A_BLOCK        0x0
#define BLIS_BITVAL_BUFFER_FOR_B_PANEL      ( 0x1 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_C_PANEL      ( 0x2 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_GEN_USE      ( 0x3 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_GENERAL                   0x0
#define BLIS_BITVAL_HERMITIAN               ( 0x1 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_SYMMETRIC               ( 0x2 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_TRIANGULAR              ( 0x3 << BLIS_STRUC_SHIFT )


//
// -- BLIS enumerated type definitions -----------------------------------------
//

// -- Operational parameter types --

typedef enum
{
	BLIS_NO_TRANSPOSE      = 0x0,
	BLIS_TRANSPOSE         = BLIS_BITVAL_TRANS,
	BLIS_CONJ_NO_TRANSPOSE = BLIS_BITVAL_CONJ,
	BLIS_CONJ_TRANSPOSE    = BLIS_BITVAL_CONJ_TRANS
} trans_t;

typedef enum
{
	BLIS_NO_CONJUGATE      = 0x0,
	BLIS_CONJUGATE         = BLIS_BITVAL_CONJ
} conj_t;

typedef enum
{
	BLIS_ZEROS             = BLIS_BITVAL_ZEROS,
	BLIS_LOWER             = BLIS_BITVAL_LOWER,
	BLIS_UPPER             = BLIS_BITVAL_UPPER,
	BLIS_DENSE             = BLIS_BITVAL_DENSE
} uplo_t;

typedef enum
{
	BLIS_LEFT              = 0x0,
	BLIS_RIGHT
} side_t;

typedef enum
{
	BLIS_NONUNIT_DIAG      = 0x0,
	BLIS_UNIT_DIAG         = BLIS_BITVAL_UNIT_DIAG
} diag_t;

typedef enum
{
	BLIS_NO_INVERT_DIAG    = 0x0,
	BLIS_INVERT_DIAG       = BLIS_BITVAL_INVERT_DIAG
} invdiag_t;

typedef enum
{
	BLIS_GENERAL           = BLIS_BITVAL_GENERAL,
	BLIS_HERMITIAN         = BLIS_BITVAL_HERMITIAN,
	BLIS_SYMMETRIC         = BLIS_BITVAL_SYMMETRIC,
	BLIS_TRIANGULAR        = BLIS_BITVAL_TRIANGULAR
} struc_t;


// -- Data type --

typedef enum
{
	BLIS_FLOAT             = BLIS_BITVAL_FLOAT_TYPE,
	BLIS_DOUBLE            = BLIS_BITVAL_DOUBLE_TYPE,
	BLIS_SCOMPLEX          = BLIS_BITVAL_SCOMPLEX_TYPE,
	BLIS_DCOMPLEX          = BLIS_BITVAL_DCOMPLEX_TYPE,
	BLIS_INT               = BLIS_BITVAL_INT_TYPE,
	BLIS_CONSTANT          = BLIS_BITVAL_CONST_TYPE,
	BLIS_DT_LO             = BLIS_FLOAT,
	BLIS_DT_HI             = BLIS_DCOMPLEX
} num_t;

typedef enum
{
	BLIS_REAL              = BLIS_BITVAL_REAL,
	BLIS_COMPLEX           = BLIS_BITVAL_COMPLEX
} dom_t;

typedef enum
{
	BLIS_SINGLE_PREC       = BLIS_BITVAL_SINGLE_PREC,
	BLIS_DOUBLE_PREC       = BLIS_BITVAL_DOUBLE_PREC
} prec_t;


// -- Pack schema type --

typedef enum
{
	BLIS_NOT_PACKED            = BLIS_BITVAL_NOT_PACKED,
	BLIS_PACKED_UNSPEC         = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_VECTOR         = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_ROWS           = BLIS_BITVAL_PACKED_ROWS,
	BLIS_PACKED_COLUMNS        = BLIS_BITVAL_PACKED_COLUMNS,
	BLIS_PACKED_ROW_PANELS     = BLIS_BITVAL_PACKED_ROW_PANELS,
	BLIS_PACKED_COL_PANELS     = BLIS_BITVAL_PACKED_COL_PANELS,
	BLIS_PACKED_ROW_PANELS_4MI = BLIS_BITVAL_PACKED_ROW_PANELS_4MI,
	BLIS_PACKED_COL_PANELS_4MI = BLIS_BITVAL_PACKED_COL_PANELS_4MI,
	BLIS_PACKED_ROW_PANELS_3MI = BLIS_BITVAL_PACKED_ROW_PANELS_3MI,
	BLIS_PACKED_COL_PANELS_3MI = BLIS_BITVAL_PACKED_COL_PANELS_3MI,
	BLIS_PACKED_ROW_PANELS_4MS = BLIS_BITVAL_PACKED_ROW_PANELS_4MS,
	BLIS_PACKED_COL_PANELS_4MS = BLIS_BITVAL_PACKED_COL_PANELS_4MS,
	BLIS_PACKED_ROW_PANELS_3MS = BLIS_BITVAL_PACKED_ROW_PANELS_3MS,
	BLIS_PACKED_COL_PANELS_3MS = BLIS_BITVAL_PACKED_COL_PANELS_3MS,
	BLIS_PACKED_ROW_PANELS_RO  = BLIS_BITVAL_PACKED_ROW_PANELS_RO,
	BLIS_PACKED_COL_PANELS_RO  = BLIS_BITVAL_PACKED_COL_PANELS_RO,
	BLIS_PACKED_ROW_PANELS_IO  = BLIS_BITVAL_PACKED_ROW_PANELS_IO,
	BLIS_PACKED_COL_PANELS_IO  = BLIS_BITVAL_PACKED_COL_PANELS_IO,
	BLIS_PACKED_ROW_PANELS_RPI = BLIS_BITVAL_PACKED_ROW_PANELS_RPI,
	BLIS_PACKED_COL_PANELS_RPI = BLIS_BITVAL_PACKED_COL_PANELS_RPI,
	BLIS_PACKED_ROW_PANELS_1E  = BLIS_BITVAL_PACKED_ROW_PANELS_1E,
	BLIS_PACKED_COL_PANELS_1E  = BLIS_BITVAL_PACKED_COL_PANELS_1E,
	BLIS_PACKED_ROW_PANELS_1R  = BLIS_BITVAL_PACKED_ROW_PANELS_1R,
	BLIS_PACKED_COL_PANELS_1R  = BLIS_BITVAL_PACKED_COL_PANELS_1R
} pack_t;

// We combine row and column packing into one "type", and we start
// with BLIS_PACKED_ROW_PANELS, _COLUMN_PANELS. We also count the
// schema pair for "4ms" (4m separated), because its bit value has
// been reserved, even though we don't use it.
#define BLIS_NUM_PACK_SCHEMA_TYPES 10


// -- Pack order type --

typedef enum
{
	BLIS_PACK_FWD_IF_UPPER = BLIS_BITVAL_PACK_FWD_IF_UPPER,
	BLIS_PACK_REV_IF_UPPER = BLIS_BITVAL_PACK_REV_IF_UPPER,

	BLIS_PACK_FWD_IF_LOWER = BLIS_BITVAL_PACK_FWD_IF_LOWER,
	BLIS_PACK_REV_IF_LOWER = BLIS_BITVAL_PACK_REV_IF_LOWER
} packord_t;


// -- Pack buffer type --

typedef enum
{
	BLIS_BUFFER_FOR_A_BLOCK = BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
	BLIS_BUFFER_FOR_B_PANEL = BLIS_BITVAL_BUFFER_FOR_B_PANEL,
	BLIS_BUFFER_FOR_C_PANEL = BLIS_BITVAL_BUFFER_FOR_C_PANEL,
	BLIS_BUFFER_FOR_GEN_USE = BLIS_BITVAL_BUFFER_FOR_GEN_USE
} packbuf_t;


// -- Partitioning direction --

typedef enum
{
	BLIS_FWD,
	BLIS_BWD
} dir_t;


// -- Subpartition type --

typedef enum
{
	BLIS_SUBPART0,
	BLIS_SUBPART1,
	BLIS_SUBPART2,
	BLIS_SUBPART1T,
	BLIS_SUBPART1B,
	BLIS_SUBPART1L,
	BLIS_SUBPART1R,
	BLIS_SUBPART00,
	BLIS_SUBPART10,
	BLIS_SUBPART20,
	BLIS_SUBPART01,
	BLIS_SUBPART11,
	BLIS_SUBPART21,
	BLIS_SUBPART02,
	BLIS_SUBPART12,
	BLIS_SUBPART22
} subpart_t;


// -- Matrix dimension type --

typedef enum
{
	BLIS_M = 0,
	BLIS_N = 1
} mdim_t;


// -- Machine parameter types --

typedef enum
{
	BLIS_MACH_EPS = 0,
	BLIS_MACH_SFMIN,
	BLIS_MACH_BASE,
	BLIS_MACH_PREC,
	BLIS_MACH_NDIGMANT,
	BLIS_MACH_RND,
	BLIS_MACH_EMIN,
	BLIS_MACH_RMIN,
	BLIS_MACH_EMAX,
	BLIS_MACH_RMAX,
	BLIS_MACH_EPS2
} machval_t;

#define BLIS_NUM_MACH_PARAMS   11
#define BLIS_MACH_PARAM_FIRST  BLIS_MACH_EPS
#define BLIS_MACH_PARAM_LAST   BLIS_MACH_EPS2


// -- Induced method types --

typedef enum
{
	BLIS_3MH = 0,
	BLIS_3M3,
	BLIS_3M2,
	BLIS_3M1,
	BLIS_4MH,
	BLIS_4M1B,
	BLIS_4M1A,
	BLIS_1M,
	BLIS_NAT
} ind_t;

#define BLIS_NUM_IND_METHODS (BLIS_NAT+1)


// -- Kernel ID types --

typedef enum
{
	BLIS_ADDV_KER  = 0,
	BLIS_AMAXV_KER,
	BLIS_AXPBYV_KER,
	BLIS_AXPYV_KER,
	BLIS_COPYV_KER,
	BLIS_DOTV_KER,
	BLIS_DOTXV_KER,
	BLIS_INVERTV_KER,
	BLIS_SCALV_KER,
	BLIS_SCAL2V_KER,
	BLIS_SETV_KER,
	BLIS_SUBV_KER,
	BLIS_SWAPV_KER,
	BLIS_XPBYV_KER
} l1vkr_t;

#define BLIS_NUM_LEVEL1V_KERS 14


typedef enum
{
	BLIS_AXPY2V_KER = 0,
	BLIS_DOTAXPYV_KER,
	BLIS_AXPYF_KER,
	BLIS_DOTXF_KER,
	BLIS_DOTXAXPYF_KER
} l1fkr_t;

#define BLIS_NUM_LEVEL1F_KERS 5


typedef enum
{
	BLIS_PACKM_0XK_KER  = 0,
	BLIS_PACKM_1XK_KER  = 1,
	BLIS_PACKM_2XK_KER  = 2,
	BLIS_PACKM_3XK_KER  = 3,
	BLIS_PACKM_4XK_KER  = 4,
	BLIS_PACKM_5XK_KER  = 5,
	BLIS_PACKM_6XK_KER  = 6,
	BLIS_PACKM_7XK_KER  = 7,
	BLIS_PACKM_8XK_KER  = 8,
	BLIS_PACKM_9XK_KER  = 9,
	BLIS_PACKM_10XK_KER = 10,
	BLIS_PACKM_11XK_KER = 11,
	BLIS_PACKM_12XK_KER = 12,
	BLIS_PACKM_13XK_KER = 13,
	BLIS_PACKM_14XK_KER = 14,
	BLIS_PACKM_15XK_KER = 15,
	BLIS_PACKM_16XK_KER = 16,
	BLIS_PACKM_17XK_KER = 17,
	BLIS_PACKM_18XK_KER = 18,
	BLIS_PACKM_19XK_KER = 19,
	BLIS_PACKM_20XK_KER = 20,
	BLIS_PACKM_21XK_KER = 21,
	BLIS_PACKM_22XK_KER = 22,
	BLIS_PACKM_23XK_KER = 23,
	BLIS_PACKM_24XK_KER = 24,
	BLIS_PACKM_25XK_KER = 25,
	BLIS_PACKM_26XK_KER = 26,
	BLIS_PACKM_27XK_KER = 27,
	BLIS_PACKM_28XK_KER = 28,
	BLIS_PACKM_29XK_KER = 29,
	BLIS_PACKM_30XK_KER = 30,
	BLIS_PACKM_31XK_KER = 31,

	BLIS_UNPACKM_0XK_KER  = 0,
	BLIS_UNPACKM_1XK_KER  = 1,
	BLIS_UNPACKM_2XK_KER  = 2,
	BLIS_UNPACKM_3XK_KER  = 3,
	BLIS_UNPACKM_4XK_KER  = 4,
	BLIS_UNPACKM_5XK_KER  = 5,
	BLIS_UNPACKM_6XK_KER  = 6,
	BLIS_UNPACKM_7XK_KER  = 7,
	BLIS_UNPACKM_8XK_KER  = 8,
	BLIS_UNPACKM_9XK_KER  = 9,
	BLIS_UNPACKM_10XK_KER = 10,
	BLIS_UNPACKM_11XK_KER = 11,
	BLIS_UNPACKM_12XK_KER = 12,
	BLIS_UNPACKM_13XK_KER = 13,
	BLIS_UNPACKM_14XK_KER = 14,
	BLIS_UNPACKM_15XK_KER = 15,
	BLIS_UNPACKM_16XK_KER = 16,
	BLIS_UNPACKM_17XK_KER = 17,
	BLIS_UNPACKM_18XK_KER = 18,
	BLIS_UNPACKM_19XK_KER = 19,
	BLIS_UNPACKM_20XK_KER = 20,
	BLIS_UNPACKM_21XK_KER = 21,
	BLIS_UNPACKM_22XK_KER = 22,
	BLIS_UNPACKM_23XK_KER = 23,
	BLIS_UNPACKM_24XK_KER = 24,
	BLIS_UNPACKM_25XK_KER = 25,
	BLIS_UNPACKM_26XK_KER = 26,
	BLIS_UNPACKM_27XK_KER = 27,
	BLIS_UNPACKM_28XK_KER = 28,
	BLIS_UNPACKM_29XK_KER = 29,
	BLIS_UNPACKM_30XK_KER = 30,
	BLIS_UNPACKM_31XK_KER = 31

} l1mkr_t;

#define BLIS_NUM_PACKM_KERS   32
#define BLIS_NUM_UNPACKM_KERS 32


typedef enum
{
	BLIS_GEMM_UKR = 0,
	BLIS_GEMMTRSM_L_UKR,
	BLIS_GEMMTRSM_U_UKR,
	BLIS_TRSM_L_UKR,
	BLIS_TRSM_U_UKR
} l3ukr_t;

#define BLIS_NUM_LEVEL3_UKRS 5


typedef enum
{
	BLIS_REFERENCE_UKERNEL = 0,
	BLIS_VIRTUAL_UKERNEL,
	BLIS_OPTIMIZED_UKERNEL,
	BLIS_NOTAPPLIC_UKERNEL
} kimpl_t;

#define BLIS_NUM_UKR_IMPL_TYPES 4


#if 0
typedef enum
{
	BLIS_JC_IDX = 0,
	BLIS_PC_IDX,
	BLIS_IC_IDX,
	BLIS_JR_IDX,
	BLIS_IR_IDX,
	BLIS_PR_IDX
} thridx_t;
#endif

#define BLIS_NUM_LOOPS 6


// -- Operation ID type --

typedef enum
{
//
// NOTE: If/when additional type values are added to this enum,
// you must either:
// - keep the level-3 values (starting with _GEMM) beginning at
//   index 0; or
// - if the value range is moved such that it does not begin at
//   index 0, implement something like a BLIS_OPID_LEVEL3_RANGE_START
//   value that can be subtracted from the opid_t value to map it
//   to a zero-based range.
// This is needed because these level-3 opid_t values are used in
// bli_l3_ind.c to index into arrays.
//
	BLIS_GEMM = 0,
	BLIS_HEMM,
	BLIS_HERK,
	BLIS_HER2K,
	BLIS_SYMM,
	BLIS_SYRK,
	BLIS_SYR2K,
	BLIS_TRMM3,
	BLIS_TRMM,
	BLIS_TRSM,

	BLIS_NOID
} opid_t;

#define BLIS_NUM_LEVEL3_OPS 10


// -- Blocksize ID type --

typedef enum
{
	BLIS_KR = 0,
	BLIS_MR,
	BLIS_NR,
	BLIS_MC,
	BLIS_KC,
	BLIS_NC,
	BLIS_M2, // level-2 blocksize in m dimension
	BLIS_N2, // level-2 blocksize in n dimension
	BLIS_AF, // level-1f axpyf fusing factor
	BLIS_DF, // level-1f dotxf fusing factor
	BLIS_XF, // level-1f dotxaxpyf fusing factor

	BLIS_NO_PART  // used as a placeholder when blocksizes are not applicable.
} bszid_t;

#define BLIS_NUM_BLKSZS 11


//
// -- BLIS misc. structure types -----------------------------------------------
//

// -- Pool block type --

typedef struct
{
	void* buf_sys;
	void* buf_align;
} pblk_t;

// -- Pool type --

typedef struct
{
	pblk_t* block_ptrs;
	dim_t   block_ptrs_len;

	dim_t   top_index;
	dim_t   num_blocks;

	siz_t   block_size;
	siz_t   align_size;
} pool_t;

// -- Mutex object type --

#include "bli_mutex.h"
#include "bli_malloc.h"

// -- Memory broker object type --

typedef struct membrk_s
{
	pool_t    pools[3];
	mtx_t     mutex;

	malloc_ft malloc_fp;
	free_ft   free_fp;
} membrk_t;

// -- Memory object type --

typedef struct mem_s
{
	pblk_t    pblk;
	packbuf_t buf_type;
	pool_t*   pool;
	membrk_t* membrk;
	siz_t     size;
} mem_t;

// -- Control tree node type --

struct cntl_s
{
	// Basic fields (usually required).
	opid_t         family;
	bszid_t        bszid;
	void*          var_func;
	struct cntl_s* sub_node;

	// Optional fields (needed only by some operations such as packm).
	// NOTE: first field of params must be a uint64_t containing the size
	// of the struct.
	void*          params;

	// Internal fields that track "cached" data.
	mem_t          pack_mem;
};
typedef struct cntl_s cntl_t;


// -- Blocksize object type --

typedef struct blksz_s
{
	// Primary blocksize values.
	dim_t  v[BLIS_NUM_FP_TYPES];

	// Blocksize extensions.
	dim_t  e[BLIS_NUM_FP_TYPES];

} blksz_t;


// -- Function pointer object type --

typedef struct func_s
{
	// Kernel function address.
	void*  ptr[BLIS_NUM_FP_TYPES];

} func_t;


// -- Multi-boolean object type --

typedef struct mbool_s
{
	bool_t  v[BLIS_NUM_FP_TYPES];

} mbool_t;


// -- Auxiliary kernel info type --

// Note: This struct is used by macro-kernels to package together extra
// parameter values that may be of use to the micro-kernel without
// cluttering up the micro-kernel interface itself.

typedef struct
{
	// The pack schemas of A and B.
	pack_t schema_a;
	pack_t schema_b;

	// Pointers to the micro-panels of A and B which will be used by the
	// next call to the micro-kernel.
	void*  a_next;
	void*  b_next;

	// The imaginary strides of A and B.
	inc_t  is_a;
	inc_t  is_b;

} auxinfo_t;


//
// -- BLIS object type definitions ---------------------------------------------
//

typedef struct obj_s
{
	// Basic fields
	struct obj_s* root;

	dim_t         off[2];
	dim_t         dim[2];
	doff_t        diag_off;

	objbits_t     info;
	siz_t         elem_size;

	void*         buffer;
	inc_t         rs;
	inc_t         cs;
	inc_t         is;

	// Bufferless scalar storage
	atom_t        scalar;

	// Pack-related fields
	dim_t         m_padded; // m dimension of matrix, including any padding
	dim_t         n_padded; // n dimension of matrix, including any padding
	inc_t         ps;       // panel stride (distance to next panel)
	inc_t         pd;       // panel dimension (the "width" of a panel:
	                        // usually MR or NR)
	dim_t         m_panel;  // m dimension of a "full" panel
	dim_t         n_panel;  // n dimension of a "full" panel
} obj_t;


// Define these macros here since they must be updated if contents of
// obj_t changes.

#define bli_obj_init_full_shallow_copy_of( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).off[0]    = (a).off[0]; \
	(b).off[1]    = (a).off[1]; \
	(b).dim[0]    = (a).dim[0]; \
	(b).dim[1]    = (a).dim[1]; \
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
	(b).is        = (a).is; \
\
	(b).scalar    = (a).scalar; \
\
	/*(b).pack_mem  = (a).pack_mem;*/ \
	(b).m_padded  = (a).m_padded; \
	(b).n_padded  = (a).n_padded; \
	(b).ps        = (a).ps; \
	(b).pd        = (a).pd; \
	(b).m_panel   = (a).m_panel; \
	(b).n_panel   = (a).n_panel; \
}

#define bli_obj_init_subpart_from( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).off[0]    = (a).off[0]; \
	(b).off[1]    = (a).off[1]; \
	/* Avoid copying m since it will be overwritten. */ \
	/* Avoid copying n since it will be overwritten. */ \
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
	(b).is        = (a).is; \
\
	(b).scalar    = (a).scalar; \
\
	/* We want to copy the pack_mem field here because this macro is used
	   when creating subpartitions, including those of packed objects. In
	   those situations, we want the subpartition to inherit the pack_mem
	   field of its parent, as well as other related fields such as the
	   padded dimensions. */ \
	/*(b).pack_mem  = (a).pack_mem;*/ \
	(b).m_padded  = (a).m_padded; \
	(b).n_padded  = (a).n_padded; \
	(b).pd        = (a).pd; \
	(b).ps        = (a).ps; \
	(b).m_panel   = (a).m_panel; \
	(b).n_panel   = (a).n_panel; \
}


// -- Context type --

typedef struct cntx_s
{
	blksz_t   blkszs[ BLIS_NUM_BLKSZS ];
	bszid_t   bmults[ BLIS_NUM_BLKSZS ];

	func_t    l3_vir_ukrs[ BLIS_NUM_LEVEL3_UKRS ];
	func_t    l3_nat_ukrs[ BLIS_NUM_LEVEL3_UKRS ];
	mbool_t   l3_nat_ukrs_prefs[ BLIS_NUM_LEVEL3_UKRS ];

	func_t    l1f_kers[ BLIS_NUM_LEVEL1F_KERS ];
	func_t    l1v_kers[ BLIS_NUM_LEVEL1V_KERS ];

	func_t    packm_kers[ BLIS_NUM_PACKM_KERS ];
	func_t    unpackm_kers[ BLIS_NUM_UNPACKM_KERS ];

	ind_t     method;
	pack_t    schema_a_block;
	pack_t    schema_b_panel;
	pack_t    schema_c_panel;

	bool_t    anti_pref;

	dim_t     thrloop[ BLIS_NUM_LOOPS ];

	membrk_t* membrk;
} cntx_t;


// -- Error types --

typedef enum
{
	BLIS_NO_ERROR_CHECKING = 0,
	BLIS_FULL_ERROR_CHECKING
} errlev_t;

typedef enum
{
	// Generic error codes
	BLIS_SUCCESS                               = (  -1),
	BLIS_FAILURE                               = (  -2),

	BLIS_ERROR_CODE_MIN                        = (  -9),

	// General errors
	BLIS_INVALID_ERROR_CHECKING_LEVEL          = ( -10),
	BLIS_UNDEFINED_ERROR_CODE                  = ( -11),
	BLIS_NULL_POINTER                          = ( -12),
	BLIS_NOT_YET_IMPLEMENTED                   = ( -13),

	// Parameter-specific errors
	BLIS_INVALID_SIDE                          = ( -20),
	BLIS_INVALID_UPLO                          = ( -21),
	BLIS_INVALID_TRANS                         = ( -22),
	BLIS_INVALID_CONJ                          = ( -23),
	BLIS_INVALID_DIAG                          = ( -24),
	BLIS_INVALID_MACHVAL                       = ( -25),
	BLIS_EXPECTED_NONUNIT_DIAG                 = ( -26),

	// Datatype-specific errors
	BLIS_INVALID_DATATYPE                      = ( -30),
	BLIS_EXPECTED_FLOATING_POINT_DATATYPE      = ( -31),
	BLIS_EXPECTED_NONINTEGER_DATATYPE          = ( -32),
	BLIS_EXPECTED_NONCONSTANT_DATATYPE         = ( -33),
	BLIS_EXPECTED_REAL_DATATYPE                = ( -34),
	BLIS_EXPECTED_INTEGER_DATATYPE             = ( -35),
	BLIS_INCONSISTENT_DATATYPES                = ( -36),
	BLIS_EXPECTED_REAL_PROJ_OF                 = ( -37),
	BLIS_EXPECTED_REAL_VALUED_OBJECT           = ( -38),

	// Dimension-specific errors
	BLIS_NONCONFORMAL_DIMENSIONS               = ( -40),
	BLIS_EXPECTED_SCALAR_OBJECT                = ( -41),
	BLIS_EXPECTED_VECTOR_OBJECT                = ( -42),
	BLIS_UNEQUAL_VECTOR_LENGTHS                = ( -43),
	BLIS_EXPECTED_SQUARE_OBJECT                = ( -44),
	BLIS_UNEXPECTED_OBJECT_LENGTH              = ( -45),
	BLIS_UNEXPECTED_OBJECT_WIDTH               = ( -46),
	BLIS_UNEXPECTED_VECTOR_DIM                 = ( -47),
	BLIS_UNEXPECTED_DIAG_OFFSET                = ( -48),
	BLIS_NEGATIVE_DIMENSION                    = ( -49),

	// Stride-specific errors
	BLIS_INVALID_ROW_STRIDE                    = ( -50),
	BLIS_INVALID_COL_STRIDE                    = ( -51),
	BLIS_INVALID_DIM_STRIDE_COMBINATION        = ( -52),

	// Structure-specific errors    
	BLIS_EXPECTED_GENERAL_OBJECT               = ( -60),
	BLIS_EXPECTED_HERMITIAN_OBJECT             = ( -61),
	BLIS_EXPECTED_SYMMETRIC_OBJECT             = ( -62),
	BLIS_EXPECTED_TRIANGULAR_OBJECT            = ( -63),

	// Storage-specific errors    
	BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT        = ( -70),

	// Partitioning-specific errors
	BLIS_INVALID_3x1_SUBPART                   = ( -80),
	BLIS_INVALID_1x3_SUBPART                   = ( -81),
	BLIS_INVALID_3x3_SUBPART                   = ( -82),

	// Control tree-specific errors
	BLIS_UNEXPECTED_NULL_CONTROL_TREE          = ( -90),

	// Packing-specific errors
	BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK  = (-100),

	// Buffer-specific errors 
	BLIS_EXPECTED_NONNULL_OBJECT_BUFFER        = (-110),

	// Memory allocator errors
	BLIS_INVALID_PACKBUF                       = (-120),
	BLIS_REQUESTED_CONTIG_BLOCK_TOO_BIG        = (-121),
	BLIS_EXHAUSTED_CONTIG_MEMORY_POOL          = (-122),
	BLIS_INSUFFICIENT_STACK_BUF_SIZE           = (-123),
	BLIS_ALIGNMENT_NOT_POWER_OF_TWO            = (-124),
	BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE        = (-125),

	// Object-related errors
	BLIS_EXPECTED_OBJECT_ALIAS                 = (-130),

	BLIS_ERROR_CODE_MAX                        = (-140)
} err_t;

#endif
