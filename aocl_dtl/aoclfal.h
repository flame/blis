/*===================================================================
 * File Name :  aoclfal.h
 * 
 * Description : Interfaces for platform/os independed file 
 *               handling API's
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/

#ifndef _AOCL_FAL_H_
#define _AOCL_FAL_H_

/* The possible error values of FAL */
#define AOCL_FAL_SUCCESS             0
#define AOCL_FAL_CLOSE_ERROR        -1
#define AOCL_FAL_READ_ERROR         -2
#define AOCL_FAL_WRITE_ERROR        -3
#define AOCL_FAL_EOF_ERROR          -6
#define AOCL_FAL_FERROR             -7

/* The type definition for FILE */
#define AOCL_FAL_FILE FILE

/* The FAL function declaration */
int32 AOCL_FAL_Close(
    AOCL_FAL_FILE *fpFilePointer);

int32 AOCL_FAL_Error(
    AOCL_FAL_FILE *fpFilePointer);

AOCL_FAL_FILE *AOCL_FAL_Open(
    const int8 *pchFileName,
    const int8 *pchMode);

int32 AOCL_FAL_Read(
    void *pvBuffer,
    int32 i32Size,
    int32 i32Count,
    AOCL_FAL_FILE *fpFilePointer);

int32 AOCL_FAL_Write(
    const void *pvBuffer,
    int32 i32Size,
    int32 iCount,
    AOCL_FAL_FILE *fpFilePointer);

#endif /* _AOCL_FAL_H_ */

/* --------------- End of aoclfal.h ----------------- */
