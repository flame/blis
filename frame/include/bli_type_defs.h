/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
  typedef   signed long int  int64_t;
  typedef unsigned long int  uint64_t;
#endif

// -- General-purpose integers --

typedef     int64_t gint_t;  // general signed integer
typedef    uint64_t guint_t; // general unsigned integer

// -- Boolean type --

typedef      gint_t bool_t;

// -- Special-purpose integers --

// This cpp guard provides a temporary hack to allow libflame
// interoperability with BLIS.
#ifndef _DEFINED_DIM_T
#define _DEFINED_DIM_T
//typedef unsigned long int dim_t;  // dimension type
typedef     guint_t dim_t;   // dimension type
#endif
typedef     guint_t inc_t;   // increment/stride type
typedef      gint_t doff_t;  // diagonal offset type
typedef     guint_t siz_t;   // byte size type
typedef     guint_t info_t;  // object information bit field

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
#ifdef BLIS_ENABLE_BLAS2BLIS_INT64
typedef int64_t   f77_int;
#else
typedef int32_t   f77_int;
#endif
typedef char      f77_char;
typedef float     f77_float;
typedef double    f77_double;
typedef scomplex  f77_scomplex;
typedef dcomplex  f77_dcomplex;


//
// -- BLIS info bit field masks ------------------------------------------------
//

#define BLIS_DOMAIN_BIT                    0x01
#define BLIS_PRECISION_BIT                 0x02
#define BLIS_SPECIAL_BIT                   0x04
#define BLIS_DATATYPE_BITS                 0x07
#define BLIS_TRANS_BIT                     0x08
#define BLIS_CONJ_BIT                      0x10
#define BLIS_CONJTRANS_BITS                0x18
#define BLIS_UPPER_BIT                     0x20
#define BLIS_DIAG_BIT                      0x40
#define BLIS_LOWER_BIT                     0x80
#define BLIS_UPLO_BITS                     0xE0
#define BLIS_UNIT_DIAG_BIT                 0x100
#define BLIS_INVERT_DIAG_BIT               0x200
#define BLIS_TARGET_DT_BITS                0x1C00
#define BLIS_EXECUTION_DT_BITS             0xE000
#define BLIS_PACK_BITS                     0x70000
#define BLIS_PACK_REV_IF_UPPER_BIT         0x80000
#define BLIS_PACK_REV_IF_LOWER_BIT         0x100000
#define BLIS_PACK_BUFFER_BITS              0x600000
#define BLIS_STRUC_BITS                    0x1800000


//
// -- BLIS enumerated type value definitions -----------------------------------
//

#define BLIS_BITVAL_REAL                   0x00
#define BLIS_BITVAL_COMPLEX                0x01
#define BLIS_BITVAL_SINGLE_PREC            0x00
#define BLIS_BITVAL_DOUBLE_PREC            0x02
#define   BLIS_BITVAL_FLOAT_TYPE           0x00
#define   BLIS_BITVAL_SCOMPLEX_TYPE        0x01
#define   BLIS_BITVAL_DOUBLE_TYPE          0x02
#define   BLIS_BITVAL_DCOMPLEX_TYPE        0x03
#define   BLIS_BITVAL_INT_TYPE             0x04
#define   BLIS_BITVAL_CONST_TYPE           0x05
#define BLIS_BITVAL_NO_TRANS               0x0
#define BLIS_BITVAL_TRANS                  0x08
#define BLIS_BITVAL_NO_CONJ                0x0
#define BLIS_BITVAL_CONJ                   0x10
#define BLIS_BITVAL_ZEROS                  0x0 
#define BLIS_BITVAL_UPPER                  0x60 
#define BLIS_BITVAL_LOWER                  0xC0
#define BLIS_BITVAL_DENSE                  0xE0
#define BLIS_BITVAL_NONUNIT_DIAG           0x0
#define BLIS_BITVAL_UNIT_DIAG              0x100
#define BLIS_BITVAL_INVERT_DIAG            0x200
#define BLIS_BITVAL_NOT_PACKED             0x0
#define BLIS_BITVAL_PACKED_UNSPEC          0x10000
#define BLIS_BITVAL_PACKED_VECTOR          0x20000
#define BLIS_BITVAL_PACKED_ROWS            0x30000
#define BLIS_BITVAL_PACKED_COLUMNS         0x40000
#define BLIS_BITVAL_PACKED_ROW_PANELS      0x50000
#define BLIS_BITVAL_PACKED_COL_PANELS      0x60000
#define BLIS_BITVAL_PACKED_BLOCKS          0x70000
#define BLIS_BITVAL_PACK_FWD_IF_UPPER      0x0
#define BLIS_BITVAL_PACK_REV_IF_UPPER      0x80000
#define BLIS_BITVAL_PACK_FWD_IF_LOWER      0x0
#define BLIS_BITVAL_PACK_REV_IF_LOWER      0x100000
#define BLIS_BITVAL_BUFFER_FOR_A_BLOCK     0x0
#define BLIS_BITVAL_BUFFER_FOR_B_PANEL     0x200000
#define BLIS_BITVAL_BUFFER_FOR_C_PANEL     0x400000
#define BLIS_BITVAL_BUFFER_FOR_GEN_USE     0x600000
#define BLIS_BITVAL_GENERAL                0x0
#define BLIS_BITVAL_HERMITIAN              0x800000
#define BLIS_BITVAL_SYMMETRIC              0x1000000
#define BLIS_BITVAL_TRIANGULAR             0x1800000

#define BLIS_TARGET_DT_SHIFT               10
#define BLIS_EXECUTION_DT_SHIFT            13
#define BLIS_PACK_BUFFER_SHIFT             21


//
// -- BLIS enumerated type definitions -----------------------------------------
//

// -- Operational parameter types --

typedef enum
{
	BLIS_NO_TRANSPOSE      = 0x0,
	BLIS_TRANSPOSE         = BLIS_TRANS_BIT,
	BLIS_CONJ_NO_TRANSPOSE = BLIS_CONJ_BIT,
	BLIS_CONJ_TRANSPOSE    = BLIS_CONJ_BIT | BLIS_TRANS_BIT
} trans_t;

typedef enum
{
	BLIS_NO_CONJUGATE      = 0x0,
	BLIS_CONJUGATE         = BLIS_CONJ_BIT
} conj_t;

typedef enum
{
	BLIS_ZEROS             = BLIS_BITVAL_ZEROS,
	BLIS_LOWER             = BLIS_LOWER_BIT | BLIS_DIAG_BIT,
	BLIS_UPPER             = BLIS_UPPER_BIT | BLIS_DIAG_BIT,
	BLIS_DENSE             = BLIS_UPPER_BIT | BLIS_DIAG_BIT | BLIS_LOWER_BIT
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
	BLIS_NOT_PACKED        = BLIS_BITVAL_NOT_PACKED,
	BLIS_PACKED_UNSPEC     = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_VECTOR     = BLIS_BITVAL_PACKED_VECTOR,
	BLIS_PACKED_ROWS       = BLIS_BITVAL_PACKED_ROWS,
	BLIS_PACKED_COLUMNS    = BLIS_BITVAL_PACKED_COLUMNS,
	BLIS_PACKED_ROW_PANELS = BLIS_BITVAL_PACKED_ROW_PANELS,
	BLIS_PACKED_COL_PANELS = BLIS_BITVAL_PACKED_COL_PANELS,
	BLIS_PACKED_BLOCKS     = BLIS_BITVAL_PACKED_BLOCKS
} pack_t;


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
	BLIS_BUFFER_FOR_GEN_USE = BLIS_BITVAL_BUFFER_FOR_GEN_USE,
} packbuf_t;


//
// -- BLIS misc. structure types -----------------------------------------------
//

// -- Memory pool type --

typedef struct
{
    void** block_ptrs;
    gint_t top_index;
    siz_t  num_blocks;
    siz_t  block_size;
} pool_t;

// -- Memory object type --

typedef struct mem_s
{
	void*     buf;
	packbuf_t buf_type;
	pool_t*   pool;
	siz_t     size;
} mem_t;

// -- Blocksize object type --

typedef struct blksz_s
{
	// Primary blocksize values.
	dim_t v[BLIS_NUM_FP_TYPES];

	// Blocksize Extensions.
	dim_t e[BLIS_NUM_FP_TYPES];
} blksz_t;


//
// -- BLIS object type definitions ---------------------------------------------
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
           - 13 domain    (0 == real, 1 == complex)
           - 14: precision (0 == single, 1 == double)
           - 15: unused
  18 ~ 16  Packed type/status
           - 0 == not packed
           - 1 == packed (unspecified; row or column)
           - 2 == packed vector
           - 3 == packed by rows
           - 4 == packed by columns
           - 5 == packed by row panels
           - 6 == packed by column panels
           - 7 == unused
       19  Packed panel order if upper-stored
           - 0 == forward order if upper
           - 1 == reverse order if upper
       20  Packed panel order if lower-stored
           - 0 == forward order if lower
           - 1 == reverse order if lower
  22 ~ 21  Packed buffer type
           - 0 == block of A
           - 1 == panel of B
           - 2 == panel of C
           - 3 == general use
  24 ~ 23  Structure type
           - (00 == general, 01 == Hermitian)
           - (10 == symmetric, 11 == triangular)
*/

typedef struct obj_s
{
	// Basic fields
	struct obj_s* root;

	dim_t         offm;
	dim_t         offn;
	dim_t         m;
	dim_t         n;
	doff_t        diag_off;

	info_t        info;
	siz_t         elem_size;

	void*         buffer;
	inc_t         rs;
	inc_t         cs;

	// Bufferless scalar storage
	atom_t        scalar;

	// Pack-related fields
	mem_t         pack_mem; // cached memory region for packing
	dim_t         m_padded; // m dimension of matrix, including any padding
	dim_t         n_padded; // n dimension of matrix, including any padding
	inc_t         ps;       // panel stride (distance to next panel)
	inc_t         pd;       // panel dimension (the "width" of a panel:
	                        // usually MR or NR)

	//mem_t         cast_mem; // cached memory region for casting

} obj_t;


// Define these macros here since they must be updated if contents of
// obj_t changes.
#define bli_obj_init_as_copy_of( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).offm      = (a).offm; \
	(b).offn      = (a).offn; \
	(b).m         = (a).m; \
	(b).n         = (a).n; \
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
\
	/* We must NOT copy pack_mem field since this macro forms the basis of
	   bli_obj_alias_to(), which is used in packm_init(). There, we want to
	   copy the basic fields of the obj_t but PRESERVE the pack_mem field
	   (and the corresponding dimensions and stride) of the destination
	   object since it holds the cached mem_t object and buffer. */ \
}

#define bli_obj_init_subpart_from( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).offm      = (a).offm; \
	(b).offn      = (a).offn; \
\
\
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
\
	/* We want to copy the pack_mem field here because this macro is used
	   when creating subpartitions, including those of packed objects. In
	   those situations, we want the subpartition to inherit the pack_mem
	   field, and the corresponding packed dimensions, of its parent. */ \
	(b).pack_mem  = (a).pack_mem; \
	(b).m_padded  = (a).m_padded; \
	(b).n_padded  = (a).n_padded; \
	(b).pd        = (a).pd; \
	(b).ps        = (a).ps; \
\
	/*(b).cast_mem  = (a).cast_mem;*/ \
}


//
// -- Other BLIS enumerated type definitions -----------------------------------
//

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


// -- Error types --

typedef enum
{
	BLIS_NO_ERROR_CHECKING = 0,
	BLIS_FULL_ERROR_CHECKING,
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

	// Memory allocator errors
	BLIS_INVALID_PACKBUF                       = (-110),
	BLIS_REQUESTED_CONTIG_BLOCK_TOO_BIG        = (-111),
	BLIS_EXHAUSTED_CONTIG_MEMORY_POOL          = (-112),

	BLIS_ERROR_CODE_MAX                        = (-120)
} err_t;

#endif
