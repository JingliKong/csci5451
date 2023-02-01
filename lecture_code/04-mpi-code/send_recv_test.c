// send_recv_test.c: Simple example code to demonstrate send/receive
// communication.  Has odd/even processors exchanging arrays and
// summing them.

#include <stdio.h>
#include <mpi.h>

#define NAME_LEN 255

int main (int argc, char *argv[]){
  MPI_Init (&argc, &argv);	// starts MPI

  // Basic info
  int npes, myrank, name_len;   
  char processor_name[NAME_LEN];
  MPI_Comm_size(MPI_COMM_WORLD, &npes); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
  MPI_Get_processor_name(processor_name, &name_len);

  // Fill a with powers of proc rank
  int count=5;
  int a[count], b[count];
  for(int i=0; i<count; i++){
    a[i] = i + 10*myrank;
  }
  // P0: has 0, 1, 2, 3, 4
  // P1: has 10, 11, 12, 13, 14
  // P3: has 30, 31, 32, 33, 34 etc.
  
  // Exchange messages with "adjacent" procs
  int tag = 1;
  int partner =-1;              // used later on after conditionals
  if (myrank%2 == 0) {          // Evens look up, send first, then receive
    partner = (myrank+1)%npes;
    MPI_Send(a, count, MPI_INT, partner, tag, MPI_COMM_WORLD);
    MPI_Recv(b, count, MPI_INT, partner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  }
  else{                         // Odds look down, receive first, then send
    partner = (myrank-1+npes)%npes;
    MPI_Recv(b, count, MPI_INT, partner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    MPI_Send(a, count, MPI_INT, partner, tag, MPI_COMM_WORLD);
  }
  // Note that the above code breaks when run on an odd number of procs


  // Compute totals
  int total = 0;
  for(int i=0; i<count; i++){
    total += a[i]+b[i];
  }
  printf("Proc %d (%s) with %d: Total = %d\n",
         myrank,processor_name,partner,total);

  MPI_Finalize();
  return 0;
}


