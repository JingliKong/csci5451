// C Example of hello world with MPI.  Compile and run as
// > mpicc -o mpi_hello mpi_hello.c
// > mpirun ./mpi_hello      # use number of processors equal to total machine procs
// > mpirun -np 2 mpi_hello  # use 2 processors
// > mpirun -np 8 mpi_hello  # use 8 processors

#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[]){
  int rank;                     // the id of this processor
  int size;                     // the number of processors being used

  MPI_Init (&argc, &argv);               // starts MPI
  MPI_Comm_rank (MPI_COMM_WORLD, &rank); // get current process id
  MPI_Comm_size (MPI_COMM_WORLD, &size); // get number of processes
  // Say hello multiple times on each processor
  for(int i=0; i<8; i++){
    printf( "Hello world #%d from process %d of %d\n", i, rank, size );
  }
  MPI_Finalize();
  return 0;
}
