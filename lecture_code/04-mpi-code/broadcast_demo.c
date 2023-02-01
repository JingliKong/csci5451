// broadcast_demo.c: Demonstration of the MPI_Bcast function to
// broadcast data from a single source to all other procs
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int procid, total_procs, *data, i;
  int root_proc = 0, num_elements=10;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &procid); 

  // Everyone allocates
  data = malloc(sizeof(int) * num_elements);

  // Root fills data by reading from file/computation
  if(procid == root_proc){
    for(i=0; i<num_elements; i++){
      data[i] = i*i;
    }
  }

  // Everyone calls broadcast, root proc sends, others receive
  MPI_Bcast(data, num_elements, MPI_INT, root_proc,
            MPI_COMM_WORLD);
  // data[] now filled with same elements on each proc

  // Each proc reports its contents
  printf("Proc %d elements: ",procid);
  for(i=0; i<num_elements; i++){
    printf("%d ",data[i]);
  }
  printf("\n");

  // Everyone frees
  free(data);
  MPI_Finalize();
  return 0;
}
