// scatter_demo.c: Demonstration of the MPI_Scatter function to spread
// distinct data to individual procs
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int procid, total_procs, i, *data;
  int root_proc = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &procid); 

  int total_elements = 16;
  int elements_per_proc = total_elements / total_procs;

  int *root_data = NULL;
  // Root allocates/fills root_data by reading from file/computation
  if(procid == root_proc){
    root_data = malloc(sizeof(int) * total_elements);
    for(i=0; i<total_elements; i++){
      root_data[i] = i*i;
    }
  }

  // Everyone allocates for their share of data including root
  data = malloc(sizeof(int) * elements_per_proc);

  // Everyone calls scatter, root proc sends, others receive
  MPI_Scatter(root_data, elements_per_proc, MPI_INT,
              data,      elements_per_proc, MPI_INT,
              root_proc, MPI_COMM_WORLD); 
  // data[] now filled with unique portion from root_data[]

  // Each proc reports its contents
  printf("Proc %d elements: ",procid);
  for(i=0; i<elements_per_proc; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  // Everyone frees data
  free(data);

  // Root frees root_data
  if(procid == root_proc){
    free(root_data);
  }
  MPI_Finalize();
  return 0;
}
