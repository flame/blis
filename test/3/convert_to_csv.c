#define CONVERT_VERSION 3.0
//============================================================
// This Code Presented to you courtesy of Oracle Labs
// By default, the blis test/3 suite outputs performance
// data in MatLab format.
// This utility converts the output until Excel compatible
// csv files, with the special ability to combine a user
// specified set of runs into a single excel spreadsheet
// for easy graph making in Excel. :)
//============================================================
//
// To build: Currently we use c++ just for the bool type..
// gcc -O2 convert_to_csv.c -o convert.x
//
// NEW:  Optional: if first argument ends in .csv, it will use
//      that name for the summary file.
//
// usage: leverage shell to group experiments
// Example, create a single spreadsheet called SingleSocketDP.csv
//   that contains all the 1s D and Z results....
// 
// ./convert.x SingleSocketDP.csv output_1s_d*blis.m output_1s_z*blis.m
//
//============================================================
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
//============================================================
#define MAX_ROWS 8192
#define MAX_FILES 256

const bool debug = false; // set to true if something goes wrong.

typedef struct { int size; float perf; } DataPoint_t;

typedef struct { char szOp[8]; char szScale[8]; char szLib[64];
  int nrows; DataPoint_t aData[MAX_ROWS]; } Experiment_t;

int gMaxRows = 0;

bool ConvertFile(const char* pszFile, Experiment_t* pStoreAll)
  {
  char szOut[256];
  sprintf(szOut, "%s.csv", pszFile);
    
  // Let's do an advanced header!
  const char* psz = pszFile;
  char szScale[3], szOp[8], szLib[64];
    
  while (*(psz++) != '_'); // scan past 1st underscore

  char* pszDst = szScale; while (*psz != '_') *pszDst++ = *psz++;  *pszDst = '\0'; psz++;
  pszDst = szOp; while (*psz != '_') *pszDst++ = *psz++;  *pszDst = '\0'; psz++;
  pszDst = szLib; while (*psz != '.') *pszDst++ = *psz++;  *pszDst = '\0';

    
  
  FILE* fpIn = fopen(pszFile, "r");
  FILE* fpOut = fopen(szOut, "w");
    
  // header:
  fprintf(fpOut, "# %s\n", pszFile);
  fprintf(fpOut, ",%s\n", szOp);
  fprintf(fpOut, ",%s\n", szScale);
  fprintf(fpOut, "size,%s\n", szLib);
    
  // Add to experiment list:
  strcpy(pStoreAll->szOp, szOp);
  strcpy(pStoreAll->szLib, szLib);
  strcpy(pStoreAll->szScale, szScale);

  char szLine[256];
  fgets(szLine, 256, fpIn); // skip first line
  int entry = 0;
    
  while (fgets(szLine, 256, fpIn))
    {
    char* pszPerf = NULL, *pszSize = NULL;
    // Real line from the right:
    char* p = szLine + strlen(szLine) - 1;
      
    while (*(--p) != ' '); // skip ];
    while (*(--p) != ' '); // Get to beginning of performance...
    pszPerf = p + 1;
    while (*(--p) == ' '); // skip space
    while (*(--p) != ' '); // Get to beginning of size...
    pszSize = p + 1;
      
    int size = atoi(pszSize);
    double perf = atof(pszPerf);
      
    // Store in data
    pStoreAll->aData[entry].perf = perf;
    pStoreAll->aData[entry].size = size;
    entry++;
    if (entry >= MAX_ROWS)
    	{
     	printf("CONVERT ERROR! File %s has more than %d rows -> increase MAX_ROWS and recompile convert.x!\n", pszFile, MAX_ROWS);
      fflush(stdout);
      break;
    	}
      
    fprintf(fpOut,"%d, %g\n", size, perf);
    }
    
  pStoreAll->nrows = entry;
  if (entry > gMaxRows) gMaxRows = entry;
    
  fclose(fpOut);
  fclose(fpIn);
    
  if (debug) printf("Wrote out file %s.\n", szOut);
    
  return true;
  }


int main(int argc, const char* argv[])
  {
  if (debug) printf("# of args = %d:\n", argc);
  int argOffset = 1; // Number of args that are NOT experiments...
  
  char szSummaryFile[32] = "summary.csv"; // the default name
  
  if (argc == 1) // no arguments gets help
  	{
    printf("==================================================\n");
		printf("Convert Version %.02f Usage: leverage shell to group experiments:\n", CONVERT_VERSION);
		printf("\te.g.: ./convert.x *_st_*blis.m will create a single chart with those experiments as columns.\n\n");
  	printf("-------------------------------------------------- Optional V3 options must be before file list:\n\n");
    printf("[MyFileName.csv] \t - filename ending in .csv defines the summary filename.\n");
  	printf("every=[n] \t\t - downsample summary, taking every nth point.\n");
    printf("offset=[k] \t\t - sample every n points starting with k, default is n.\n");
    printf("==================================================\n");
		exit(0);
  	}
  
  // NEW:  Optional initial arguments:
	int every = 1, offset = -1;
	
ProcessArgs:

  if (argc > argOffset) // do we have a known argument?
  	{
  	long len = strlen(argv[argOffset]);
   
   	if (len > 4)
    	{
    	if (strcasecmp(&(argv[argOffset][len-4]), ".csv") == 0)
     		{
        strcpy(szSummaryFile, argv[argOffset]);
     		printf("Writing output summary file to %s\n", szSummaryFile);
        argOffset++; // now more args
        goto ProcessArgs;  // get next arg
     		}
      else if (len > 6)
        {
        if (strncmp(argv[argOffset], "every=", 6) == 0)
          {
          every = atoi(argv[argOffset] + 6);
          if (debug) printf("User set every to %d\n", every);
          argOffset++; // now more args
          goto ProcessArgs;  // get next arg
          }
        else if (len > 7)
          {
          if (strncmp(argv[argOffset], "offset=", 7) == 0)
            {
            offset = atoi(argv[argOffset] + 7);
            if (debug) printf("User set offset to %d\n", offset);
            argOffset++; // now more args
            goto ProcessArgs;  // get next arg
            }
      		}
        }
      }
  	}
    
  // fallthrough - no valid options found, parse files.
  if (offset == -1) { offset = every;
    printf("offset set to %d\n", offset); }// offset not set
  
  int numExperiments = argc - argOffset;
  if (numExperiments < 1)
   	{
    printf("No input files found.\n");
    return -1; // no args for files...
  	}
    
printf("Running experiments...\n"); fflush(stdout);

  // Allocate space for entire run...
  Experiment_t aExperiments[numExperiments];
  
  // Assume all but the first are a file (.m) to convert...
  for (int i = argOffset; i < argc; i++)
    {
    if (i > MAX_FILES)
    	{
     	printf("CONVERT ERROR: Combining more than %d files!  Increase MAX_FILES and recompile convert.x!\n", MAX_FILES);
      fflush(stdout);
      break;
    	}
    printf("Processing File: %s\n", argv[i]); fflush(stdout);
    ConvertFile(argv[i], &aExperiments[i-argOffset]);
    }
    
  // Now output the UEBER file!
  FILE* fpOut = fopen(szSummaryFile, "w");
  
  // First, output the headers:
  for (int col = 0; col < numExperiments; col++)  fprintf(fpOut, ",%s,", aExperiments[col].szOp); fprintf(fpOut, "\n");
  for (int col = 0; col < numExperiments; col++)  fprintf(fpOut, ",%s,", aExperiments[col].szScale); fprintf(fpOut, "\n");
  for (int col = 0; col < numExperiments; col++)  fprintf(fpOut, "size,%s,", aExperiments[col].szLib); fprintf(fpOut, "\n");
    
  // And now, the data...
  for (int row = offset; row < gMaxRows; row += every)
    {
    for (int col = 0; col < numExperiments; col++)
      {
      if (row < aExperiments[col].nrows)
        fprintf(fpOut, "%d,%g,", aExperiments[col].aData[row].size, aExperiments[col].aData[row].perf);
      else fprintf(fpOut, ",,");
      }
    fprintf(fpOut, "\n");
    }
    
  fclose(fpOut);
  if (debug) printf("Wrote out summary file %s.\n", szSummaryFile);
  return 0;
  }
