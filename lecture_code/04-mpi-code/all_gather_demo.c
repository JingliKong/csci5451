// all_gather_demo.c: Demonstration of the MPI_Allgather function;
// collects distinct data from individual procs onto ALL procs
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int proc_id, total_procs, i, *data, *all_data;
  int root_proc = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id); 

  int total_elements = 16;
  int elements_per_proc = total_elements / total_procs;

  // Everyone allocates for their share of data including root
  data = malloc(sizeof(int) * elements_per_proc);

  // Each proc fills data[] with unique values 
  int x = 1;
  for(i=0; i<elements_per_proc; i++){
    data[i] = x;
    x *= (proc_id+2);
  }
  // data[] now filled with unique values on each proc

  // Each proc reports its contents
  printf("proc %d elements: ",proc_id);
  for(i=0; i<elements_per_proc; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);  // attempt to sync output

  // Everyone allocates all_data to be filled with gathered data
  all_data = malloc(sizeof(int) * total_elements);

  // Everyone calls all-gather, everyone sends and receives
  MPI_Allgather(data,     elements_per_proc, MPI_INT,
                all_data, elements_per_proc, MPI_INT,
                MPI_COMM_WORLD); 
  // all_data[] now contains each procs data[] in order on 
  // all procs

  // buffer for printing 1-line message
  char buf[1024], *pos;         

  // Everyone reports their data and frees memory
  pos = buf;
  for(i=0; i<total_elements; i++){
    pos += sprintf(pos,"%2d ",all_data[i]);
  }
  printf("Proc %d all_data[]: %s\n",proc_id,buf);

  // Everyone frees data
  free(data);
  free(all_data);

  MPI_Finalize();
  return 0;
}
