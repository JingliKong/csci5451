// Slightly more advanced hello world. Each processor reports its
// proc_id and the host machine on which it is running. Includes two
// utilities for printing messages prepended with the proc_id and
// printing only on the root.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <mpi.h>

int total_procs, proc_id, name_len;
char proc_name[256];

// Utility to print messages with the proc # and host name prepended
// to the message. Works like printf().
void pprintf(const char* format, ...) { 
  va_list args;
  char buffer[2048];
  int len = sprintf(buffer,"P%04d [%s]: ",proc_id, proc_name);
  va_start (args, format);
  vsprintf(buffer+len, format, args);
  va_end (args);
  printf("%s",buffer);
}


int main (int argc, char *argv[]){
  MPI_Init (&argc, &argv);                      // starts MPI 
  MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
  MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
  MPI_Get_processor_name(proc_name, &name_len); // get the symbolic host name 

  pprintf("Hello world from process %4d of %d \n",
          proc_id, total_procs);

  MPI_Finalize();
  return 0;
}
