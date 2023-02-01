// send_bugs.c: buggy implementations that show send/send or recv/recv
// problems. Demo runs are
//
// > mpirun -np 2 ./a.out ss 32
// > mpirun -np 2 ./a.out ss 1024
// > mpirun -np 2 ./a.out rr 32

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define NAME_LEN 255

int main (int argc, char *argv[]){
  MPI_Init (&argc, &argv);	// starts MPI

  if(argc < 3){
    printf("usage: mpirun -np <int> %s <mode> <size>\n", argv[0]);
    printf("  <mode>: 'ss' for send/send, 'rr' for recv/recv\n");
    printf("  <size>: number of integes in buffer to send\n");
    return 1;
  }

  char *mode = argv[1];
  int size = atoi(argv[2]);

  // basic info
  int npes, myrank, name_len;   
  char processor_name[NAME_LEN];
  MPI_Comm_size(MPI_COMM_WORLD, &npes); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
  MPI_Get_processor_name(processor_name, &name_len);

  if(myrank == 0 && (npes % 2)!=0){ // Warn about odd numbers of procs
    printf("Warning: odd number of procs %d in use, failures likely\n ",
           npes);
  }
  int *a = malloc(sizeof(int)*size);
  int *b = malloc(sizeof(int)*size);

  // initialize a[] with proc num, b received from other proc
  memset(a, myrank, sizeof(int)*size);

  // up one for evens, down one for odds, only works for even # of
  // procs in an mpirun
  int partner = (myrank%2 == 0) ? (myrank+1) : (myrank-1);

  if(0){}
  else if( strcmp(mode, "ss")==0 ){  // send/send buggy
    printf("Proc %d Sending to %d\n", myrank, partner);
    MPI_Send(a, size, MPI_INT, partner, 1, MPI_COMM_WORLD);

    printf("Proc %d Receiving from %d\n", myrank, partner);
    MPI_Recv(b, size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  }
  else if( strcmp(mode, "rr")==0 ){  // recv/recv buggy
    printf("Proc %d Receiving from %d\n", myrank, partner);
    MPI_Recv(b, size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

    printf("Proc %d Sending to %d\n", myrank, partner);
    MPI_Send(a, size, MPI_INT, partner, 1, MPI_COMM_WORLD);
  }
  else{
    printf("Unknown mode '%s' : aborting\n", mode);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Compute totals
  int total = 0;
  for(int i=0; i<size; i++){
    total += a[i]+b[i];
  }
  printf("Proc %d (%s) with %d: Total = %d\n",myrank,processor_name,partner,total);

  free(a); free(b);

  MPI_Finalize();
  return 0;
}
