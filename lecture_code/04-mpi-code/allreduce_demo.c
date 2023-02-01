/* Demonstration of the MPI_Reduce function; sums/products/maxes/mins
   distinct data from individual procs onto a single proc */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int procid, total_procs, i, *data, *reduced_data;
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

  // Everyon allocates reduced_data to be filled with reduced data
  reduced_data = malloc(sizeof(int) * total_elements);

  MPI_Barrier(MPI_COMM_WORLD);  // barrier to make printing saner

  // Everyone calls reduce, everyone sends and receives
  MPI_Allreduce(data, reduced_data, total_elements, MPI_INT,
                MPI_SUM, // operation to perform on each element
                MPI_COMM_WORLD); 
  // reduced_data[] now contains each procs data[] summed up

  // Each proc reports its reduced data
  printf("proc %d SUM     : ",procid);
  for(i=0; i<total_elements; i++){
    printf("%4d ",reduced_data[i]);
  }
  printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);  // barrier to make printing saner

  // Reduce again but this time find the MIN and do it in place with
  // option MPI_IN_PLACE 
  MPI_Allreduce(MPI_IN_PLACE, data, total_elements, MPI_INT,
                MPI_MIN, // operation to perform on each element
                MPI_COMM_WORLD); 
  // data[] now contains each procs data[], min elements

  // Each proc reports its reduced data
  printf("proc %d MIN     : ",procid);
  for(i=0; i<total_elements; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  // Everyone frees data
  free(data);
  free(reduced_data);

  MPI_Finalize();
  return 0;
}
