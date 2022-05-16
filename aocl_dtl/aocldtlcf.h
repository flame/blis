/*===================================================================
 * File Name :  aocldtlcf.h
 *
 * Description : This is configuration file for debug and trace
 *               libaray, all debug features (except auto trace)
 *               can be enabled/disabled in this file.
 *
 * Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/

#ifndef _AOCLDTLCF_H_
#define _AOCLDTLCF_H_

/* Macro for tracing the log If the user wants to enable tracing he has to
   enable this macro by making it to 1 else 0 */
#define AOCL_DTL_TRACE_ENABLE       0

/* Macro for dumping the log If the user wants to enable dumping he has to
   enable this macro by making it to 1 else 0 */
#define AOCL_DTL_DUMP_ENABLE        0

/* Macro for dumping the log If the user wants to enable input logs he has to
   enable this macro by making it to 1 else 0 */
#define AOCL_DTL_LOG_ENABLE         0

/* Select the trace level till which you want to log the data */
/* By default it will log for all levels */
#define AOCL_DTL_TRACE_LEVEL         AOCL_DTL_LEVEL_TRACE_5

/* user has to explicitly use the below macros to identify
   ciriticality of the logged message */
#define AOCL_DTL_LEVEL_ALL          (15)
#define AOCL_DTL_LEVEL_TRACE_9      (14)
#define AOCL_DTL_LEVEL_TRACE_8      (13)
#define AOCL_DTL_LEVEL_TRACE_7      (12)     /* Kernels */
#define AOCL_DTL_LEVEL_TRACE_6      (11)
#define AOCL_DTL_LEVEL_TRACE_5      (10)
#define AOCL_DTL_LEVEL_TRACE_4      (9)
#define AOCL_DTL_LEVEL_TRACE_3      (8)
#define AOCL_DTL_LEVEL_TRACE_2      (7)
#define AOCL_DTL_LEVEL_TRACE_1      (6)       /* BLIS/BLAS API */
#define AOCL_DTL_LEVEL_VERBOSE      (5)
#define AOCL_DTL_LEVEL_INFO         (4)
#define AOCL_DTL_LEVEL_MINOR        (3)
#define AOCL_DTL_LEVEL_MAJOR        (2)
#define AOCL_DTL_LEVEL_CRITICAL     (1)


#define AOCL_DTL_TRACE_FILE         "aocldtl_trace.txt"
#define AOCL_DTL_AUTO_TRACE_FILE    "aocldtl_auto_trace.rawfile"
#define AOCL_DTL_LOG_FILE           "aocldtl_log.txt"

/* The use can use below three macros for different data type while dumping data
 * or specify the size of data type in bytes macro for character data type */
#define AOCL_CHAR_DATA_TYPE         (1)

/* macro for short data type */
#define AOCL_UINT16_DATA_TYPE       (2)

/* macro for String data type */
#define AOCL_STRING_DATA_TYPE       (3)

/* macro for uint32 data type */
#define AOCL_UINT32_DATA_TYPE       (4)

/* macro for printing Hex values */
#define AOCL_LOG_HEX_VALUE          ('x')

/* macro for printing Decimal values */
#define AOCL_LOG_DECIMAL_VALUE      ('d')



#endif /* _AOCLDTLCF_H_ */

/* --------------- End of aocldtlcf.h ----------------- */
