/* Demonstration of the MPI_Scatterv function to spread distinct data
   to individual procs but each proc may have a different count of the
   data. */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);

  int proc_id, total_procs, i, *data, *root_data;
  int root_proc = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id); 

  int total_elements = 16;

  // Determine how many elements each proc will have and their
  // displacements. Not strictly necessary in a scatterv for all procs
  // to know how many elements will be on all other procs, but for any
  // subsequent all-gatherv operation it is needed so for simplicity
  // just have everyone calculate it here.
  int *counts = malloc(total_procs * sizeof(int));
  int *displs = malloc(total_procs * sizeof(int));

  // Divide total_elements as evenly as possible: lower numbered
  // processors get one extra element each.
  int elements_per_proc = total_elements / total_procs;
  int surplus           = total_elements % total_procs;
  for(i=0; i<total_procs; i++){
    counts[i] = (i < surplus) ? elements_per_proc+1 : elements_per_proc;
    displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
  }
  // counts[] and displs[] now contain relevant data for a scatterv,
  // gatherv, all-gatherv calls

  // Root allocates/fills root_data by reading from file/computation
  if(proc_id == root_proc){
    root_data = malloc(sizeof(int) * total_elements);
    for(i=0; i<total_elements; i++){
      root_data[i] = i*i;
    }
  }

  // Everyone allocates for their share of data including root. Use
  // the above counts[] array for individual sizes.
  data = malloc(sizeof(int) * counts[proc_id]);

  // Everyone calls scatterv, root proc sends, others receive,
  // quantities dictated by the contents of counts[]
  MPI_Scatterv(root_data, counts, displs,  MPI_INT,
               data,      counts[proc_id], MPI_INT,
               root_proc, MPI_COMM_WORLD); 
  // data[] now filled with unique portion from root_data[]

  // Each proc reports its contents
  printf("Proc %d has %d elements: ",
         proc_id, counts[proc_id]);
  for(i=0; i<counts[proc_id]; i++){
    printf("%4d ",data[i]);
  }
  printf("\n");

  // Everyone frees data
  free(data);
  free(counts);
  free(displs);

  // Root frees root_data
  if(proc_id == root_proc){
    free(root_data);
  }
  MPI_Finalize();
  return 0;
}
