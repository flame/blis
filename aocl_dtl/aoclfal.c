/*===================================================================
 * File Name :  aoclfal.c
 * 
 * Description : Platform/os independed file handling API's
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/

#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"



/* Disable instrumentation for following function, since they are called from 
 * Auto Generated execution trace handlers. */

/* The FAL function declaration */
int32 AOCL_FAL_Close(
    AOCL_FAL_FILE *fpFilePointer) __attribute__((no_instrument_function));

int32 AOCL_FAL_Error(
    AOCL_FAL_FILE *fpFilePointer) __attribute__((no_instrument_function));

AOCL_FAL_FILE *AOCL_FAL_Open(
    const int8 *pchFileName,
    const int8 *pchMode) __attribute__((no_instrument_function));

int32 AOCL_FAL_Read(
    void *pvBuffer,
    int32 i32Size,
    int32 i32Count,
    AOCL_FAL_FILE *fpFilePointer) __attribute__((no_instrument_function));

int32 AOCL_FAL_Write(
    const void *pvBuffer,
    int32 i32Size,
    int32 iCount,
    AOCL_FAL_FILE *fpFilePointer) __attribute__((no_instrument_function));

/*=============================================================================
*  Function Name       :   AOCL_FAL_Open
*  Description         :   Used for opening a file specified by name
*  Input Parameter(s)  :   int8 *pchFileName - Stores the file name (path)
*                          int8 *pchMode - Specify the mode for opening file
*  Output Parameter(s) :   None
*  Return parameter(s) :   AOCL_FAL_FILE - If the file is opened successfully
*                          NULL - If there is any error while opening file
*============================================================================*/
AOCL_FAL_FILE *AOCL_FAL_Open(
    const int8 *pchFileName,
    const int8 *pchMode)
{
    AOCL_FAL_FILE *fpFileOpen = NULL;
    /* Open the file with provided by specified path and mode in which it should
      be opened. Refer to FILE I/O operation help for getting mode types */
    fpFileOpen = fopen(pchFileName, pchMode);
    /* If the file is not opened then NULL value should be returned */
    if (NULL == fpFileOpen)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Cannot open file: AOCL_FAL_Open()");
    }
    return fpFileOpen;
} /* end of AOCL_FAL_Open */

/*=============================================================================
*  Function Name       :   AOCL_FAL_Close
*  Description         :   Used for closing a file specified by file pointer
*  Input Parameter(s)  :   AOCL_FAL_FILE *fpFilePointer - File pointer
*  Output Parameter(s) :   None
*  Return parameter(s) :   0 - If the file is closed successfully
*                          AOCL_FAL_CLOSE_ERROR - For any error while closing file
*
*============================================================================*/
int32 AOCL_FAL_Close(
    AOCL_FAL_FILE *fpFilePointer)
{
    /* Return value for the file close */
    int32 i32RetVal;
    i32RetVal = AOCL_FAL_CLOSE_ERROR;

    /* Check whether the file pointer passed is valid or not */
    if (NULL == fpFilePointer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Can not close file: AOCL_FAL_Close()");
        return i32RetVal;
    }

    /* Close the file using the FILE pointer passed */
    i32RetVal = fclose(fpFilePointer);

    /* If the return value is non zero then it indicates an error */
    if (i32RetVal)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR,
                     "Can't close file, Invalid file pointer passed");
        return i32RetVal;
    }

    /* On successful closing of the file, function should return 0 */
    return i32RetVal;

} /* End of AOCL_FAL_Close */

/*=============================================================================
*  Function Name       :   AOCL_FAL_Read
*  Description         :   Used for reading a file specified by file pointer.
*                          This function reads the specified number of bytes
*                          from the file into the buffer specified. The bytes
*                          read are returned by this function.
*  Input Parameter(s)  :   int32 i32Size - Item size in bytes
*                          int32 i32Count - Maximum number of items to be read
*                          AOCL_FAL_FILE *fpFilePointer - File ptr to read from
*  Output Parameter(s) :   void *pvBuffer - Storage location of data
*  Return parameter(s) :   i32RetVal - Number of bytes read if successful
*                          AOCL_FAL_READ_ERROR - In case of error while reading
*============================================================================*/
int32 AOCL_FAL_Read(
    void *pvBuffer,
    int32 i32Size,
    int32 i32Count,
    AOCL_FAL_FILE *fpFilePointer)
{
    /* Return value for the file read */
    int32 i32RetVal;
    i32RetVal = AOCL_FAL_READ_ERROR;

    /* Check pointer used for pointing the storage location data is valid */
    if (NULL == pvBuffer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR,
                     "Can not read the file, Buffer pointer is NULL");
        return i32RetVal;
    }

    /* Check whether file pointer passed is valid */
    if (NULL == fpFilePointer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR,
                     "Can not read the file, Buffer pointer is NULL");
        return i32RetVal;
    }

    /* Read the file using file pointer */
    i32RetVal = fread(pvBuffer, i32Size, i32Count, fpFilePointer);

    if (i32RetVal != i32Count)
    {
        /* Check whether this is an end of file The AOCL_FAL_Error() will return
         non-zero value to indicate an error */
        if (AOCL_FAL_Error(fpFilePointer)) /* AOCL_FAL_EndOfFile (fpFilePointer) */
        {
            AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR,
                         "There is an error condition while file read");
            i32RetVal = AOCL_FAL_READ_ERROR;
        }
        /* This is condition where file read has encountered an end of file */
        else
        {
            AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "End of file...");
        }
    }

    /* The number of bytes read by the file read operation.
    * This value may be less than the actual count, due to end of file
    * or an error while reading the file */
    return i32RetVal;

} /* End of AOCL_FAL_Read */

/*=============================================================================
*  Function Name       :   AOCL_FAL_Write
*  Description         :   Used for writing data to a file specified by file
*                          pointer. The number of bytes written to file are
*                          written by this function.
*  Input Parameter(s)  :   const void *pvBuffer - Pointer to data location from
*                                                 where the data to be copied
                           int32 i32Size - Item size in bytes
*                          int32 i32Count - Maximum number of items to be
*                                           written
*                          AOCL_FAL_FILE *fpFilePointer - File pointer to write to
*  Output Parameter(s) :   None
*  Return parameter(s) :   i32RetVal - Number of bytes written if successful
*                          AOCL_FAL_WRITE_ERROR - In case of error while writing
*============================================================================*/
int32 AOCL_FAL_Write(
    const void *pvBuffer,
    int32 i32Size,
    int32 iCount,
    AOCL_FAL_FILE *fpFilePointer)
{
    /* Return value for write operation */
    int32 i32RetVal;
    i32RetVal = AOCL_FAL_WRITE_ERROR;
    /* Check pointer used for pointing the storage location data is valid */
    if (NULL == pvBuffer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Can not perform file write");
        return i32RetVal;
    }

    /* Check whether the file pointer passed is valid or not */
    if (NULL == fpFilePointer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Can not perform file write");
        return i32RetVal;
    }

    /* Write into the file specified by the file pointer */
    i32RetVal = fwrite(pvBuffer, i32Size, iCount, fpFilePointer);

    /* If the number of bytes written into the file are less than specified
    * bytes then it is an error while file writing */
    if (i32RetVal != iCount)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "File write operation error");
        i32RetVal = AOCL_FAL_WRITE_ERROR;
    }

    /* The return value of the file write operation */
    return i32RetVal;

} /* End of AOCL_FAL_Write */

/*=============================================================================
*  Function Name       :   AOCL_FAL_Error
*  Description         :   Used for testing an error on the file specified
*  Input Parameter(s)  :   AOCL_FAL_FILE *fpFilePointer - File pointer
*  Output Parameter(s) :   None
*  Return parameter(s) :   non-zero - Indicates an end of file
*                          0 - Indicates that function is successful
*                          non-zero - Indicates that there is some error
*                          AOCL_FAL_ERROR - Indicates error during the operation
*============================================================================*/
int32 AOCL_FAL_Error(
    AOCL_FAL_FILE *fpFilePointer)
{
    /* Used for storing the return value for ferror function */
    int32 i32RetVal;
    i32RetVal = AOCL_FAL_FERROR;

    /* Check whether the file pointer is NULL */
    if (NULL == fpFilePointer)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Invalid file pointer is passed");
        return i32RetVal;
    }

    /* Call the ferror function to get an error on the file */
    i32RetVal = ferror(fpFilePointer);

    /* Check for the return value, it non-zero there is an error */
    if (i32RetVal)
    {
        AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "The file has some error");
        i32RetVal = AOCL_FAL_FERROR;
    }

    /* In case of success, this function should return 0 */
    return i32RetVal;

} /* End of AOCL_FAL_Error */

/* ------------------- End of aoclfal.c ----------------------- */
