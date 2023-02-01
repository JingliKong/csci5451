// reduce_demo.c: Demonstration of the MPI_Reduce function.  Sums /
// Products / Maxes / Mins distinct data from individual procs onto a
// single proc
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

  // Everyone allocates for their share of data including root
  data = malloc(sizeof(int) * total_elements);

  /* Each proc fills data[] with unique values */
  int x = 1;
  for(i=0; i<total_elements; i++){
    data[i] = x;
    x *= (procid+2);
  }
  // data[] now filled with unique values on each proc

  // Each proc reports its contents
  printf("proc %d elements: ",procid);
  for(i=0; i<total_elements; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  // Root allocates root_data to be filled with reduced data
  if(procid == root_proc){
    root_data = malloc(sizeof(int) * total_elements);
  }

  // Everyone calls reduce, root proc receives, others send and operate
  MPI_Reduce(data, root_data, total_elements, MPI_INT,
             MPI_SUM, // operation to perform on each element
             root_proc, MPI_COMM_WORLD); 
  // root_data[] now contains each procs data[] summed up

  // Root reports gathere data
  if(procid == root_proc){
    printf("SUM ROOT %d elements: ",procid);
    for(i=0; i<total_elements; i++){
      printf("%6d ",root_data[i]);
    }
    printf("\n");
  }


  // Reduce again but this time find the MIN
  MPI_Reduce(data, root_data, total_elements, MPI_INT,
             MPI_MIN, // operation to perform on each element
             root_proc, MPI_COMM_WORLD); 
  // root_data[] now contains each procs data[], min elements


  // Root reports gathere data
  if(procid == root_proc){
    printf("MIN ROOT %d elements: ",procid);
    for(i=0; i<total_elements; i++){
      printf("%6d ",root_data[i]);
    }
    printf("\n");
  }

  // Everyone frees data
  free(data);
  // root frees data
  if(procid == root_proc){
    free(root_data);
  }

  MPI_Finalize();
  return 0;
}
