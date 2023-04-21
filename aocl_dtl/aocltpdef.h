
/*===================================================================
 * File Name :  aocltpdef.h
 *
 * Description : Abstraction for various datatypes used by DTL.
 *
 * Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/
#ifndef AOCL_TYPEDEF_H_
#define AOCL_TYPEDEF_H_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#ifndef _WIN32
#include <sys/types.h>
#else
typedef int pid_t;
#endif

typedef double                  Double;
typedef float                   Float;
typedef void                    Void;
typedef unsigned char           uint8;
typedef unsigned short int      uint16;
typedef unsigned int            uint32;
typedef unsigned long           uint64;
typedef uint8                   *STRING;
typedef unsigned char           Bool;
typedef char                    int8;
typedef signed long int         int32;
typedef short int               int16;

typedef Void                    *AOCL_HANDLE;
typedef pid_t                   AOCL_TID;

#endif /*AOCL_TYPEDEF_H_ */

/* --------------- End of aocltpdef.h ----------------- */
