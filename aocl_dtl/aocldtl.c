/*===================================================================
 * File Name :  aocldtl.c
 * 
 * Description : This file contains main logging functions.
 *               These functions are invoked though macros by
 *               end user.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/

#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"
#include "aocldtlcf.h"
#include "aoclflist.h"
#include "aoclos.h"

#ifdef AOCL_DTL_AUTO_TRACE_ENABLE
#if defined(__linux__)
#define __USE_GNU
#include <dlfcn.h>
#endif
#endif

/* By default the trace level will be set to ALL User can configure this 
      parameter at run time using command line argument */
uint32 gui32TraceLogLevel = AOCL_DTL_TRACE_LEVEL;

/* The user can configure the file name in which he wants to dump the data */
#if AOCL_DTL_TRACE_ENABLE
/* The file name for storing traced log added manually in the code */
static char *pchDTL_TRACE_FILE = AOCL_DTL_TRACE_FILE;

/* Global file pointer for trace logging */
AOCL_FLIST_Node *gpTraceFileList = NULL;

#endif

#if AOCL_DTL_LOG_ENABLE
/* The file name for storing log data */
static char *pchDTL_LOG_FILE = AOCL_DTL_LOG_FILE;

/* Global file pointer for logging the results */
AOCL_FLIST_Node *gpLogFileList = NULL;
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE

/* The file name for storing execution trace, 
   These files are used by compiler assisted execution testing */
static char *pchDTL_AUTO_TRACE_FILE = AOCL_DTL_AUTO_TRACE_FILE;

/* Global file pointer for logging the results */
AOCL_FLIST_Node *gpAutoTraceFileList = NULL;
#endif

/*===================================================================
*  Function Name           :  DTL_Initialize
*  Description             :  Creates/Opens log file and initializes the 
*                             global trace log level 
*  Input Parameter(s)      :  ui32CurrentLogLevel - current log level 
*                             which user can configure at run time 
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#ifdef AOCL_DTL_INITIALIZE_ENABLE

void DTL_Initialize(
    uint32 ui32CurrentLogLevel)
{
    /* If user selects invalid trace log level then the dafault trace log level 
      will be AOCL_DTL_LEVEL_ALL */
    if ((ui32CurrentLogLevel < 1) || (ui32CurrentLogLevel > AOCL_DTL_LEVEL_ALL))
    {
        gui32TraceLogLevel = AOCL_DTL_LEVEL_ALL;
    }
	else
	{
		/* Assign the user requested log level to the global trace log level */
		gui32TraceLogLevel = ui32CurrentLogLevel;
	}

#if AOCL_DTL_TRACE_ENABLE
    /* Create/Open the file to log the traced data */
    AOCL_FLIST_AddFile(pchDTL_TRACE_FILE, &gpTraceFileList, AOCL_gettid());

    if (NULL == gpTraceFileList)
    {
        /* Unable to open the specified file.*/
        AOCL_DEBUGPRINT("Unable to create the trace file %s\n", pchDTL_TRACE_FILE);
        return;
    }
#endif

#if AOCL_DTL_LOG_ENABLE
    /* Create/Open the file to log the log data */
    AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, AOCL_gettid());

    if (NULL == gpLogFileList)
    {
        /* Unable to open the specified file.*/
        AOCL_DEBUGPRINT("Unable to create the log file %s\n", pchDTL_LOG_FILE);
        return;
    }
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE
    /* Create/Open the file to log the log data */
    AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

    if (NULL == gpAutoTraceFileList)
    {
        /* Unable to open the specified file.*/
        AOCL_DEBUGPRINT("Unable to create the log file %s\n", pchDTL_AUTO_TRACE_FILE);
        return;
    }
#endif

} /* DTL_Initialize */
#endif

/*===================================================================
*  Function Name           :  DTL_Uninitialize
*  Description             :  Close all the log files
*  Input Parameter(s)      :  void
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#ifdef AOCL_DTL_INITIALIZE_ENABLE
void DTL_Uninitialize(void)
{
#if AOCL_DTL_TRACE_ENABLE
    /* Close the trace file */
    AOCL_FLIST_CloseAll(gpTraceFileList);
#endif

#if AOCL_DTL_LOG_ENABLE
    /* Close the log file */
    AOCL_FLIST_CloseAll(gpLogFileList);
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE
    /* Close the log file */
    AOCL_FLIST_CloseAll(gpAutoTraceFileList);
#endif
    return;
} /* DTL_Uninitialise */
#endif

/*===================================================================
*  Function Name           :  DTL_Trace
*  Description             :  This is common lowest level function
*                             to log the event to a file, This function
*                             will take case of choosing correct file
*                             according to the current thread and
*                             log the event as per format requested.

*  Input Parameter(s)      :  ui8LogLevel - Log Level
*                             ui8LogType - Identify log type (entry, exit etc)
*                             pi8FileName.- File name 
*                             pi8FunctionName - Function Name
*                             ui32LineNumber - Line number 
*                             pi8Message - Message to be printed
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#if (AOCL_DTL_TRACE_ENABLE || AOCL_DTL_LOG_ENABLE)
void DTL_Trace(
    uint8 ui8LogLevel,
    uint8 ui8LogType,
    const int8 *pi8FileName,
    const int8 *pi8FunctionName,
    uint32 ui32LineNumber,
    const int8 *pi8Message)
{
    uint8 i = 0;
    AOCL_FAL_FILE *pOutFile = NULL;

    if (ui8LogType == TRACE_TYPE_LOG || ui8LogType == TRACE_TYPE_RAW)
    {
#if AOCL_DTL_LOG_ENABLE
        pOutFile = AOCL_FLIST_GetFile(gpLogFileList, AOCL_gettid());

        /* If trace file pointer is equal to NULL then return with out dumping data 
         to the file */
        if (NULL == pOutFile)
        {
            /* It might be the first call from the current thread, try to create
         new trace for this thread. */
            pOutFile = AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, AOCL_gettid());

            if (NULL == pOutFile)
            {
                AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
                return;
            }
        }
#endif /* Logging enabled */
    } 
    else
    {
#if AOCL_DTL_TRACE_ENABLE
	 pOutFile = AOCL_FLIST_GetFile(gpTraceFileList, AOCL_gettid());

        /* If trace file pointer is equal to NULL then return with out dumping data
         to file */
        if (NULL == pOutFile)
        {
            /* It might be the first call from the current thread, try to create
         new trace for this thread. */
            pOutFile = AOCL_FLIST_AddFile(pchDTL_TRACE_FILE, &gpTraceFileList, AOCL_gettid());

            if (NULL == pOutFile)
            {
                AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
                return;
            }
        }
#endif /* Trace Enabled */
    }

    /* Log the message only if the log level is less than or equal to global log
      level set while initialization */
    if (ui8LogLevel <= gui32TraceLogLevel)
    {
		
		/* Indent as per level if is function call trace */
		if ((ui8LogLevel >= AOCL_DTL_LEVEL_TRACE_1) &&
			(ui8LogLevel <= AOCL_DTL_LEVEL_TRACE_8))
		{
			/* this loop is for formating the output log file */
			for (i = 0; i < (ui8LogLevel - AOCL_DTL_LEVEL_TRACE_1); i++)
			{
				/* print tabs in the output file */
				fprintf(pOutFile, "\t");
			}
		}

        switch (ui8LogType)
        {
        case TRACE_TYPE_FENTRY:
            fprintf(pOutFile, "In %s()...\n", pi8FunctionName);
            break;

        case TRACE_TYPE_FEXIT:
            if (pi8Message == NULL)
            { /* Function returned successfully */
                fprintf(pOutFile, "Out of %s()\n", pi8FunctionName);
            }
            else
            { /* Function failed to complete, use message to get error */
                fprintf(pOutFile, "Out of %s() with error %s\n", pi8FunctionName, pi8Message);
            }
            break;

        case TRACE_TYPE_LOG:
            fprintf(pOutFile, "%s:%d:%s\n", pi8FileName, ui32LineNumber, pi8Message);
            break;

        case TRACE_TYPE_RAW:
            fprintf(pOutFile, "%s\n", pi8Message);
            break;
        }
    }
} /* DTL_Data_Trace_Entry */
#endif

/*===================================================================
*  Function Name           :  DTL_DumpData
*  Description             :  This function is mainly used for dumping 
*                             the data into the file
*  Input Parameter(s)      :  pui8Buffer - the buffer to be dumped
*                             ui32BufferSize.- the no. of bytes to be dumped
*                             ui8DataType - the data type char/int32/int32
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#if AOCL_DTL_DUMP_ENABLE
void DTL_DumpData(
    uint8 ui8LogLevel,
    void *pvBuffer,
    uint32 ui32BufferSize,
    uint8 ui8DataType,
    int8 *pi8Message,
    int8 i8OutputType)
{
    uint32 j;

    /* Pointer to store the buffer */
    uint32 *pui32Array, ui32LocalData;
    uint16 *pui16Array;
    uint8 *pui8CharArray;
    int8 *pi8CharString;

    /* If dump (log) file pointer is equal to NULL return with out dumping data to file */
    AOCL_FAL_FILE *pDumpFile = AOCL_FLIST_GetFile(gpLogFileList, AOCL_gettid());
    /* Log the message only if the log level is less than or equal to global log
      level set while initialization */
    if (ui8LogLevel > gui32TraceLogLevel)
    {
        return;
    }

    /* The string message */
    if (pi8Message != NULL)
    {
        fprintf(pDumpFile, "%s :", pi8Message);
    }

    /* Assuming that if the Data type for character = 1
   * the Data type for uint32 = 2
   * the data type for uint32 = 4
   * the data type for string = 3
   */
    if (ui8DataType == AOCL_STRING_DATA_TYPE)
    {
        /* Typecast the void buffer to character buffer */
        pi8CharString = (int8 *)pvBuffer;
        fprintf(pDumpFile, "%s", pi8CharString);
        fprintf(pDumpFile, "\n");
    }

    if (ui8DataType == AOCL_CHAR_DATA_TYPE)
    {
        /* Typecast the void buffer to character buffer */
        pui8CharArray = (uint8 *)pvBuffer;

        for (j = 0; j < ui32BufferSize; j++)
        {
            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, pui8CharArray[j]);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, pui8CharArray[j]);
            }
        }
        fprintf(pDumpFile, "\n");
    }

    if (ui8DataType == AOCL_UINT16_DATA_TYPE)
    {
        /* Typecast the void buffer to uint32 bit buffer */
        pui16Array = (uint16 *)pvBuffer;

        /* dump the data in the file line by line */
        for (j = 0; j < ui32BufferSize; j++)
        {
            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, pui16Array[j]);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, pui16Array[j]);
            }
        }
        fprintf(pDumpFile, "\n");

    } /* End of if */

    if (ui8DataType == AOCL_UINT32_DATA_TYPE)
    {
        /* Typecast the void buffer to uint32 buffer */
        pui32Array = (uint32 *)pvBuffer;

        /* dump the data in the file line by line */
        for (j = 0; j < ui32BufferSize; j++)
        {
            ui32LocalData = pui32Array[j];

            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, ui32LocalData);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, ui32LocalData);
            }
        }
        fprintf(pDumpFile, "\n");
    } /* End of if */

} /* DTL_DumpData */
#endif

/* This is enabled by passing ETRACE_ENABLE=1 to make */
#ifdef AOCL_DTL_AUTO_TRACE_ENABLE

/* 
    Disable intrumentation for these functions as they will also be
    called from compiler generated instumation code to trace 
   function execution.

    It needs to be part of declration in the C file so can't be 
    moved to header file.

    WARNING: These functions are automatically invoked. however any function
    called from this should have instumtation disable to avoid recursive 
    calls which results in hang/crash.
   */
void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

/*===================================================================
*  Function Name           :  __cyg_profile_func_enter
*  Description             :  This function is automatically invoked
*                             by compiler instrumntation when the flow
*                             enters a function.
*  Input Parameter(s)      :  pvThisFunc - Address of function entered. 
*                             call_site.- Address of the caller
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
void __cyg_profile_func_enter(void *pvThisFunc, void *pvCaller)
{
    Dl_info info;
    dladdr(pvThisFunc, &info);

    AOCL_FAL_FILE *pOutFile = NULL;

    pOutFile = AOCL_FLIST_GetFile(gpAutoTraceFileList, AOCL_gettid());

    /* If trace file pointer is equal to NULL then return with out dumping data 
        to the file */
    if (NULL == pOutFile)
    {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        pOutFile = AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return;
        }
    }

    fprintf(pOutFile, "\n%lu:+:%p",
            AOCL_getTimestamp(),
            (void *)(pvThisFunc - info.dli_fbase));
}

/*===================================================================
*  Function Name           :  __cyg_profile_func_exit
*  Description             :  This function is automatically invoked
*                             by compiler before returing from a 
*                             function.
*  Input Parameter(s)      :  pvThisFunc - Address of function to be existed. 
*                             call_site.- Address of the caller
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
void __cyg_profile_func_exit(void *pvThisFunc, void *pvCaller)
{
    Dl_info info;
    dladdr(pvThisFunc, &info);
    AOCL_FAL_FILE *pOutFile = NULL;

    pOutFile = AOCL_FLIST_GetFile(gpAutoTraceFileList, AOCL_gettid());

    /* If trace file pointer is equal to NULL then return with out dumping data 
        to the file */
    if (NULL == pOutFile)
    {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        pOutFile = AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return;
        }
    }

    fprintf(pOutFile, "\n%lu:-:%p",
            AOCL_getTimestamp(),
            (void *)(pvThisFunc - info.dli_fbase));
}

#endif /* AOCL_AUTO_TRACE_ENABLE */

/* ------------------ End of aocldtl.c ---------------------- */
