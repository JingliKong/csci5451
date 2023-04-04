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
    // Allocate space for local data
    double** proc_data = malloc(sizeof(double*) * max_time);
    for (int i = 0; i < max_time; i++){
        proc_data[i] = malloc(sizeof(double) * cols_per_proc);
    }
    
    volatile int t = 0; //initalize t to 0
    volatile int p;

    if (proc_id == 0) {
        // Allocate memory for collective data
        H = malloc(sizeof(double*)*max_time); 
        for(t = 0; t < max_time; t++){
            // H[t] = malloc(sizeof(double)*width);
            H[t] = calloc(width, sizeof(double)); 
        }

        // Initialize constant left/right boundary temperatures
        for(t=0; t < max_time; t++){
            H[t][0] = L_bound_temp;
            H[t][width-1] = R_bound_temp;
        }
        // Initialize temperatures at time 0
        t=0;
        for(p=1; p < width-1; p++){
            H[t][p] = initial_temp;
        }

        // Proc 0 initializes partition of H to starting conditions 
        proc_data[t][0] = L_bound_temp; 
        for (int i = 1; i < cols_per_proc; i++) {
            proc_data[t][i] = initial_temp; 
        }
        // Initialize constant left boundary temperatures
        for(t=0; t < max_time; t++){
            proc_data[t][0] = L_bound_temp;
        }
    }

    if (proc_id == total_procs - 1) { // we are the leftmost portion of the pipe 
        // proc_data[t][cols_per_proc - 1] = R_bound_temp; 
        t = 0;
        for (int i = 1; i < cols_per_proc; i++) {
            proc_data[t][i] = initial_temp; 
        }
        // Proc n-1 initializes partition of H to starting conditions 
        proc_data[t][0] = L_bound_temp; 
        int i = (total_procs == 1) ? 0 : 1;
        for (; i < cols_per_proc - 1; i++) {
            proc_data[t][i] = initial_temp; 
        }

        // Initialize constant right boundary temperatures
        for(t=0; t < max_time; t++){
            proc_data[t][cols_per_proc-1] = R_bound_temp;
        }
    } else { // we are in the middle all the initial temps are just 0 
        for (int i = 0; i < cols_per_proc; i++) {
            proc_data[t][i] = initial_temp; 
        }
    }

    // Simulate the temperature changes for internal cells
    for(t = 0; t < max_time-1; t++){
        for(p = 1; p < cols_per_proc - 1; p++){
            double left_diff  = proc_data[t][p] - proc_data[t][p-1];
            double right_diff = proc_data[t][p] - proc_data[t][p+1];
            double delta = -k*( left_diff + right_diff );
            proc_data[t+1][p] = proc_data[t][p] + delta;
        }

        // Communicate accross processors
        double recv_left, recv_right;
        double send_left, send_right; 

        if (total_procs <= 1) {
            continue;
        } else if (proc_id == 0) { // Leftmost processor case
            send_right = proc_data[t][cols_per_proc - 1];
            MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, 1, 1,
                          &recv_right, 1, MPI_DOUBLE, 1, 1, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            recv_left = proc_data[t][cols_per_proc - 2];
            proc_data[t+1][cols_per_proc - 1] = proc_data[t][cols_per_proc - 1] + -k*(2*proc_data[t][cols_per_proc - 1] - recv_left - recv_right);
    
        } else if ((proc_id == (total_procs - 1))) { // Rightmost processor case
            send_left = proc_data[t][0];
            MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                          &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            recv_right = proc_data[t][1];              
            proc_data[t+1][0] = proc_data[t][0] + -k*(2*proc_data[t][0] - recv_left - recv_right);          
            
        } else { // Middle processor case 
            // We need 2 MPI_Sendrecv one to the proc to the right and one to the left 
            // sending my right most element to the recv_left because we are sending 

            if (proc_id % 2 == 0) {
                send_right = proc_data[t][cols_per_proc - 1];
                MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, proc_id + 1, 1, 
                            &recv_right, 1, MPI_DOUBLE, proc_id + 1, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_left = proc_data[t][0];               
                MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                            &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            } else {
                send_left = proc_data[t][0];               
                MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, proc_id - 1, 1,
                            &recv_left, 1, MPI_DOUBLE, proc_id - 1, 1, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_right = proc_data[t][cols_per_proc - 1];
                MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, proc_id + 1, 1, 
                            &recv_right, 1, MPI_DOUBLE, proc_id + 1, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }


            // Then we just calculate the ends of our local array using the values we just recieved 
            // calculating the end 
            double left_diff = proc_data[t][0] - recv_left;  
            double right_diff = proc_data[t][0] - proc_data[t][1];
            double delta = -k*( left_diff + right_diff );
            proc_data[t+1][0] = proc_data[t][0] + delta;

            // now computing for the rightmost element of local array 
            left_diff = proc_data[t][0] - proc_data[t][cols_per_proc - 2];
            right_diff =  proc_data[t][0] - recv_right; 
            delta = -k*( left_diff + right_diff );
            proc_data[t+1][cols_per_proc - 1] = proc_data[t][cols_per_proc] + delta;
        }
    }

    // Gather all of the results
    MPI_Gather(proc_data, cols_per_proc, MPI_DOUBLE,
                H, cols_per_proc, MPI_DOUBLE,
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
        printf("%x ------ %x\n", H[t], proc_data[t]);
        if (proc_id == 0) {
            free(H[t]);
        }
            
        free(proc_data[t]);
    }
   
    free(proc_data);

    if (proc_id == 0) {
        free(H);
    }
    

    MPI_Finalize();

    return 0;
}