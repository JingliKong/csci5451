#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv){
    if(argc < 4){
        printf("usage: %s max_time width print\n", argv[0]);
        printf("  max_time: int\n");
        printf("  width: int\n");
        printf("  print: 1 print output, 0 no printing\n");
        return 0;
    }

    int max_time = atoi(argv[1]); // Number of time steps to simulate
    int width = atoi(argv[2]);    // Number of cells in the rod
    int print = atoi(argv[3]);
    double initial_temp = 50.0;   // Initial temp of internal cells 
    double L_bound_temp = 20.0;   // Constant temp at Left end of rod
    double R_bound_temp = 10.0;   // Constant temp at Right end of rod
    double k = 0.5;               // thermal conductivity constant
    double **H;                   // 2D array of temps at times/locations

    //Begin MPI code:
    MPI_Init (&argc, &argv); 

    //MPI variables
    int proc_id, total_procs, cols_per_proc;
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     // get current process id 
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); // get number of processes 
    int root_processor = 0; // have the root processor be processor 0

    //Confirm unsupported configurations fail
    if ((width % total_procs) != 0) {
        printf("Number of columns not divisible by procs\n");
        return 0;

    } else if ((cols_per_proc = width / total_procs) < 3) {
        printf("Not emough columns per proc\n");
        return 0;
    }

    // Divide and assign the columns
    // 3 * cols_per_proc is start, end at 3 * (cols_per_proc + 1) 

    // Allocate memory 
    H = malloc(sizeof(double*)*max_time); 
    int t,p;
    for(t=0;t<max_time;t++){
        H[t] = malloc(sizeof(double)*width);
    }

    // Initialize constant left/right boundary temperatures
    for(t=0; t<max_time; t++){
        H[t][0] = L_bound_temp;
        H[t][width-1] = R_bound_temp;
    }

    // Initialize temperatures at time 0
    t = 0;
    for(p=1; p<width-1; p++){
        H[t][p] = initial_temp;
    }


    // PARALLELIZE THIS PART BELOW?

    // Simulate the temperature changes for internal cells
    for(t=0; t<max_time-1; t++){
        for(p=1; p<width-1; p++){
            double left_diff  = H[t][p] - H[t][p-1];
            double right_diff = H[t][p] - H[t][p+1];
            double delta = -k*( left_diff + right_diff );
            H[t+1][p] = H[t][p] + delta;
        }
    }

    double** root_data;

    //Gather all of the results into the root processor
    if (proc_id == root_processor) {
        //allocate space on the root processor for all times
        root_data = malloc(sizeof(double*)*max_time); 
        for(int i = 0; i < max_time; i++){
            H[i] = malloc(sizeof(double)*width);
        }
    } 

    // double send[10], recv[10]; int partner;
    // partner = (procid % 2 == 1) ? procid-1 : procid+1;
    // MPI_Sendrecv(send, 10, MPI_DOUBLE, partner, 1,
    // recv, 10, MPI_DOUBLE, partner, 1,
    // MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Gather(H, cols_per_proc, MPI_INT,
                root_data, cols_per_proc, MPI_INT,
                root_processor, MPI_COMM_WORLD); 



    // Print the results
    if(print){
        // Column headers
        printf("%3s| ","");
        for(p=0; p<width; p++){
            printf("%5d ",p);
        }
        printf("\n");
        printf("%3s+-","---");
        for(p=0; p<width; p++){
            printf("------");
        }
        printf("\n");
            // Row headers and data
            for(t=0; t<max_time; t++){
                printf("%3d| ",t);
                for(p=0; p<width; p++){
                printf("%5.1f ",H[t][p]);
                }
                printf("\n");
            }
    }


    // Clean up and deallocation
    for(t=0; t<max_time; t++){
        free(H[t]);
        free(root_data[t]);
    }
    free(H);
    free(root_data);
    MPI_Finalize();

    return 0;
}