// comm_split.c: Demonstrates use of MPI_Comm_split() by repeatedly
// dividing a communicator until each processor is in its own.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <mpi.h>

// Initialize MPI and relevant variables then have each proc say hello  
int main (int argc, char *argv[]){
  MPI_Init (&argc, &argv);                      // starts MPI 

  int total_procs, proc_id;                     // overall proc name
  MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
  MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 

  int cur_proc_id, cur_total_procs;
  MPI_Comm cur_comm, new_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &cur_comm);

  for(int p=total_procs, i=0; p>=1; p/=2, i++){
    MPI_Comm_rank (cur_comm, &cur_proc_id);     // get current process id 
    MPI_Comm_size (cur_comm, &cur_total_procs); // get number of processes 
    printf("P%02d Iter: %d: Cur proc_id: %d / %d\n",
           proc_id, i, cur_proc_id, cur_total_procs);

    int color = cur_proc_id < (cur_total_procs / 2);
    MPI_Comm_split(cur_comm, color, cur_proc_id, &new_comm);
    MPI_Comm_dup(new_comm, &cur_comm);
    // Memory leak as MPI_Comm_free() should be used to free communicators
  }

  MPI_Finalize();
  return 0;
}
