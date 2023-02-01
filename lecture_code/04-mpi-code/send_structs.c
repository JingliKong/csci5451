// send_structs.c: Demonstrates the simple, platform-dependent method
// of sending structs

#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define NAME_LEN 255

typedef struct {
  double x;
  int a, b;
} dint_t;

int main (int argc, char *argv[]){
  MPI_Init (&argc, &argv);	// starts MPI

  // Basic info
  int npes, myrank, name_len;   
  char processor_name[NAME_LEN];
  MPI_Comm_size(MPI_COMM_WORLD, &npes); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
  MPI_Get_processor_name(processor_name, &name_len);


  // Fill a with powers of proc rank - values are arbitrary
  dint_t mine[10], yours[10];
  for(int i=0; i<10; i++){
    mine[i].a = i*myrank;
    mine[i].b = i*i*myrank / 2;
    mine[i].x = sqrt(mine[i].a * mine[i].b);
  }
  
  // Exchange messages with "adjacent" procs
  int partner = -1;             // save partner number for later
  if (myrank%2 == 0) {          // Evens look up, send first, then receive
    partner = (myrank+1)%npes;
    MPI_Send(mine,  10*sizeof(dint_t), MPI_BYTE, // calculate data sizes "manually"
             partner, 1, MPI_COMM_WORLD);        // just as is done in a malloc()
    MPI_Recv(yours, 10*sizeof(dint_t), MPI_BYTE,
             partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  }
  else{                         // Odds look down, receive first, then send
    partner = (myrank-1+npes)%npes;
    MPI_Recv(yours, 10*sizeof(dint_t), MPI_BYTE,
             partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    MPI_Send(mine,  10*sizeof(dint_t), MPI_BYTE,
             partner, 1, MPI_COMM_WORLD);
  }
  // Note that the above code breaks if the number of processors being run 

  // Compute totals
  double total = 0;
  for(int i=0; i<10; i++){
    total += mine[i].x+yours[i].x;
  }
  printf("Proc %d (%s) with %d: Total = %f\n",
         myrank,processor_name,partner,total);

  MPI_Finalize();
  return 0;
}
