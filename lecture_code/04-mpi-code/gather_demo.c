// gather_demo.c: Demonstration of the MPI_Gather function collect
// distinct data from individual procs onto a single proc
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int procid, total_procs, i, *data, *root_data;
  int root_proc = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &procid); 

  int total_elements = 16;
  int elements_per_proc = total_elements / total_procs;

  // Everyone allocates for their share of data including root
  data = malloc(sizeof(int) * elements_per_proc);

  /* Each proc fills data[] with unique values */
  int x = 1;
  for(i=0; i<elements_per_proc; i++){
    data[i] = x;
    x *= (procid+2);
  }
  // data[] now filled with unique values on each proc

  // Each proc reports its contents
  printf("proc %d elements: ",procid);
  for(i=0; i<elements_per_proc; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  // Root allocates root_data to be filled with gathered data
  if(procid == root_proc){
    root_data = malloc(sizeof(int) * total_elements);
  }

  // Everyone calls gather, root proc receives, others send
  MPI_Gather(data,      elements_per_proc, MPI_INT,
             root_data, elements_per_proc, MPI_INT,
             root_proc, MPI_COMM_WORLD); 
  // root_data[] now contains each procs data[] in order


  // Everyone frees data
  free(data);

  // Root reports gathere data, frees root_data
  if(procid == root_proc){
    printf("ROOT %d elements: ",procid);
    for(i=0; i<total_elements; i++){
      printf("%4d ",root_data[i]);
    }
    printf("\n");
    free(root_data);
  }

  MPI_Finalize();
  return 0;
}
